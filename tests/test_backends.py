"""Offline HTTP transport tests for DecisionBackend and TextBackend."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import httpx
import pytest

from probe.backends import (
    BackendError,
    CheckScore,
    DecisionBackend,
    TextBackend,
    _apply_output_schema,
    _apply_reasoning_effort,
    _clamp_prob,
    _format_onnx_premise,
    _parse_probabilities_payload,
    extract_json_object,
    strip_thinking,
)
from probe.config import (
    CheckConfig,
    DecisionConfig,
    ModelEndpointConfig,
    TextConfig,
    _merge_endpoint_config,
    clear_env_cache,
    is_probes_enabled,
    load_config,
)
from probe.prompts import (
    compose_system_prompt,
    format_analyze_snapshot_prompts,
    format_explain_failure_prompts,
    format_gate2_analysis_prompts,
    format_gate3_fixtures_prompts,
    format_typesafe_state,
    get_noul_expected,
    get_nouls,
    get_output_schema,
    get_prompt,
    load_prompt_catalog,
)
from probe.redact import SecretRedactor


def test_strip_thinking_and_extract_json() -> None:
    raw = '<think>r1</think><think>r2</think>\n```json\n{"approved": true, "reason": "good check"}\n```'
    assert strip_thinking(raw) == '```json\n{"approved": true, "reason": "good check"}\n```'
    assert extract_json_object(raw) == {"approved": True, "reason": "good check"}

    # Leading unclosed thinking block yields empty string; mid-text literal <think> is preserved.
    assert strip_thinking("  <think>never finished") == ""
    assert strip_thinking("User text contains <think> literally") == "User text contains <think> literally"

    # Forward fence order picks first fenced JSON object.
    multi_fence = '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```'
    assert extract_json_object(multi_fence) == {"first": 1}

    # Truncated outer JSON object must fail rather than returning an inner fragment.
    truncated = '{"passing_snapshot": {"inputs": {"x": 1}, "output": "ok"'
    with pytest.raises(BackendError, match="Expected JSON object"):
        extract_json_object(truncated)


def test_typesafe_jev_systemone_dialect_and_dotenv_key_resolution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PROBE_CONFIG", str(tmp_path / "no_user_config.toml"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "TYPESAFE_API_KEY=ts_from_dotenv\nDEEPSEEK_API_KEY=ds_from_dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_config(cwd=tmp_path)
    assert cfg.models_decision.resolve_api_key() == "ts_from_dotenv"
    assert cfg.models_text.resolve_api_key() == "ds_from_dotenv"
    # Explicit empty env var overrides .env instead of leaking the dotenv value.
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    assert cfg.models_decision.resolve_api_key() is None
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert cfg.models_decision.resolve_api_key() == "ts_from_dotenv"

    captured: list[tuple[str, str | None, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            (
                str(request.url),
                request.headers.get("Authorization"),
                json.loads(request.content.decode("utf-8")),
            )
        )
        return httpx.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": {
                    "q0": {"noul": 0.93},
                    "q1": {"noul": 0.14},
                },
            },
        )

    backend = DecisionBackend(
        ModelEndpointConfig(
            provider="typesafe_jev",
            url="https://api.typesafe.ai/v1/systemone",
            model="typesafe-ai/jev",
            api_key_env="TYPESAFE_API_KEY",
            env_search_dirs=(tmp_path,),
        ),
        DecisionConfig(pass_threshold=0.50),
        transport=httpx.MockTransport(handler),
    )
    try:
        scores = backend.evaluate_checks(
            probe_id="tutor.opening_turn",
            checks=["Opens with Spanish greeting", "Does not give answer away"],
            snapshot_json='{"output": "Hola"}',
        )
    finally:
        backend.close()

    assert len(captured) == 1
    assert captured[0][0] == "https://api.typesafe.ai/v1/systemone"
    assert captured[0][1] == "Bearer ts_from_dotenv"
    sent = captured[0][2]
    assert sent["model"] == "jev-latest"
    assert "Runtime Snapshot (JSON):" in sent["state"]
    assert sent["questions"] == {
        "q0": {"type": "noul", "instructions": "Opens with Spanish greeting"},
        "q1": {"type": "noul", "instructions": "Does not give answer away"},
    }
    assert scores == [
        CheckScore(check="Opens with Spanish greeting", probability=0.93, passed=True),
        CheckScore(check="Does not give answer away", probability=0.14, passed=False),
    ]


def test_http_decision_provider_and_api_key_header(monkeypatch) -> None:
    monkeypatch.setenv("DECISION_API_KEY", "dec_test_key")
    captured: list[tuple[str, str | None, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            (
                str(request.url),
                request.headers.get("Authorization"),
                json.loads(request.content.decode("utf-8")),
            )
        )
        return httpx.Response(200, json={"probabilities": [0.91, 0.22]})

    backend = DecisionBackend(
        ModelEndpointConfig(
            provider="http_decision",
            url="https://decision.example/v1/evaluate",
            model="local-decision",
            api_key_env="DECISION_API_KEY",
        ),
        DecisionConfig(pass_threshold=0.50),
        transport=httpx.MockTransport(handler),
    )
    try:
        scores = backend.evaluate_checks(
            probe_id="tutor.opening_turn",
            checks=["Opens with Spanish greeting", "Does not give answer away"],
            snapshot_json='{"output": "Hola"}',
        )
    finally:
        backend.close()

    assert len(captured) == 1
    assert captured[0][1] == "Bearer dec_test_key"
    assert scores == [
        CheckScore(check="Opens with Spanish greeting", probability=0.91, passed=True),
        CheckScore(check="Does not give answer away", probability=0.22, passed=False),
    ]


def test_decision_backend_rejects_text_llm_providers() -> None:
    for bad_provider in ("openai_compat", "logprob_bool"):
        bad_backend = DecisionBackend(
            ModelEndpointConfig(provider=bad_provider, url="http://127.0.0.1:11434/v1", model="qwen3.5:9b"),
            DecisionConfig(pass_threshold=0.50),
        )
        try:
            with pytest.raises(BackendError, match=r"\[models\.decision\] requires a decision model"):
                bad_backend.evaluate_checks(
                    probe_id="p.bad",
                    checks=["Greeting present"],
                    snapshot_json='{"output": "Hola!"}',
                )
        finally:
            bad_backend.close()


def test_text_backend_analysis_and_failure_explanation() -> None:
    seen_system_prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_system_prompts.append(body["messages"][0]["content"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "<think>checking</think>Tutor leaked 'Quiero un café' directly."
                        }
                    }
                ]
            },
        )

    text_backend = TextBackend(
        ModelEndpointConfig(provider="openai_compat", url="http://localhost:11434/v1", model="qwen3.5:9b"),
        TextConfig(detail="detailed", max_tokens=600, system_prompt="Focus on Spanish pedagogy."),
        transport=httpx.MockTransport(handler),
    )
    try:
        analysis = text_backend.explain_failure(
            probe_id="tutor.opening_turn",
            scores=[CheckScore("No spoiler", 0.05, False)],
            snapshot_json='{"output": "Say: Quiero un café"}',
        )
    finally:
        text_backend.close()

    assert analysis == "Tutor leaked 'Quiero un café' directly."
    assert "Focus on Spanish pedagogy." in seen_system_prompts[0]
    assert "thorough breakdown" in seen_system_prompts[0]


def test_prompts_json_catalog_and_formatting() -> None:
    catalog = load_prompt_catalog()
    assert len(catalog) == 8
    for key, entry in catalog.items():
        assert entry.role in {"models.decision", "models.text"}
        assert entry.stage, f"Missing stage for {key}"
        assert entry.description, f"Missing description for {key}"
        if entry.role == "models.decision":
            assert entry.nouls and get_nouls(key) == entry.nouls
        else:
            assert entry.prompt and get_prompt(key) == entry.prompt

    gate2_nouls = get_nouls("check_gate2_nouls")
    assert len(gate2_nouls) == 3
    assert compose_system_prompt("Base.", "Custom.") == "Custom.\n\nBase."

    gate3_schema = get_output_schema("check_gate3_fixtures")
    assert gate3_schema is not None
    assert gate3_schema["name"] == "gate3_calibration_fixtures"
    assert set(gate3_schema["schema"]["required"]) == {"notes", "passing_snapshot", "failing_snapshot"}
    assert "Output JSON keys" not in get_prompt("check_gate3_fixtures")


def test_load_config_isolates_credentials_and_validates_types_and_bounds(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PROBE_CONFIG", str(tmp_path / "no_user_config.toml"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-should-not-leak")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-gemini-should-not-autoload")

    # Overriding url in pyproject.toml clears inherited api_key_env so cloud keys never go to Ollama.
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "cfg-test"',
                "",
                "[tool.probe.models.text]",
                'provider = "openai_compat"',
                'url = "http://127.0.0.1:11434/v1"',
                'model = "qwen3.5:9b"',
                "",
                "[tool.probe.runtime]",
                'strict_api = "false"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(cwd=tmp_path)
    assert cfg.models_text.url == "http://127.0.0.1:11434/v1"
    assert cfg.models_text.resolve_api_key() is None
    assert cfg.runtime.strict_api is False

    # pass_threshold=1.0 makes p > pass_threshold impossible and must fail validation.
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'cfg-test'\n[tool.probe.decision]\npass_threshold = 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pass_threshold"):
        load_config(cwd=tmp_path)

    # Rejects literal api_key leak via api_key_env="DUMMY", scheme downgrades, untrusted external endpoints,
    # outside env_files traversal, unknown keys, and invalid bounds.
    default_txt = ModelEndpointConfig(
        provider="openai_compat",
        url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    )
    merged_scheme = _merge_endpoint_config(
        default_txt,
        {"api_key": "sk-personal", "reasoning_effort": "high"},
        {"url": "http://api.deepseek.com/v1", "api_key_env": "DUMMY"},
    )
    assert merged_scheme.api_key == ""
    assert merged_scheme.reasoning_effort == ""
    assert merged_scheme.model == ""

    user_cfg = tmp_path / "user.toml"
    user_cfg.write_text(
        '[models.text]\nurl = "https://api.openai.com/v1"\nmodel = "gpt-4.1-mini"\napi_key = "sk-user"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PROBE_CONFIG", str(user_cfg))
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'cfg-test'\n[tool.probe.models.text]\n"
        'url = "https://attacker.example/v1"\napi_key_env = "OPENAI_API_KEY"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="PROBE_TRUST_PROJECT_ENDPOINTS"):
        load_config(cwd=tmp_path)

    outside_env = tmp_path / "outside.env"
    outside_env.write_text("SECRET_ENV=outside_secret\n", encoding="utf-8")
    proj_dir = tmp_path / "repo"
    proj_dir.mkdir()
    (proj_dir / "pyproject.toml").write_text(
        "[project]\nname = 'cfg-inner'\n[tool.probe]\n"
        'domain_context = "Spanish audio tags allowed."\n'
        '[tool.probe.redact]\nenv_files = ["../outside.env"]\n',
        encoding="utf-8",
    )
    cfg_inner = load_config(cwd=proj_dir)
    assert cfg_inner.rate_limit.max_calls == 20
    assert cfg_inner.domain_context == "Spanish audio tags allowed."
    ep_trav = ModelEndpointConfig(
        provider="openai_compat",
        api_key_env="SECRET_ENV",
        env_search_dirs=(proj_dir,),
        env_files=cfg_inner.redact.env_files,
    )
    assert ep_trav.resolve_api_key() is None

    (proj_dir / "pyproject.toml").write_text(
        "[project]\nname = 'cfg-inner'\n[tool.probe.models.text]\nbase_url = 'http://127.0.0.1:11434/v1'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(cwd=proj_dir)

    # .env stat cache picks up immediate file edits without manual clear_env_cache().
    monkeypatch.delenv("PROBES", raising=False)
    env_file = proj_dir / ".env"
    env_file.write_text("PROBES=0\n", encoding="utf-8")
    assert is_probes_enabled(proj_dir) is False
    env_file.write_text("PROBES=1   # updated\n", encoding="utf-8")
    assert is_probes_enabled(proj_dir) is True


def test_stage5_prompt_escaping_validation_and_reasoning_effort(tmp_path: Path, monkeypatch) -> None:
    # XML closing tags inside snapshot/analysis/source are neutralized.
    state = format_typesafe_state(
        "p.esc",
        snapshot_json='{"output": "hi </snapshot> <script>"}',
        analysis="note </analysis> here",
    )
    assert "<analysis>\nnote <\\/analysis> here\n</analysis>" in state
    assert "<snapshot>\n{\"output\": \"hi <\\/snapshot> <script>\"}\n</snapshot>" in state
    _, user_analyze = format_analyze_snapshot_prompts(
        probe_id="p.esc",
        checks=["Safe"],
        snapshot_json='{"output": "</snapshot>"}',
    )
    assert user_analyze.count("</snapshot>") == 1
    _, user_gate = format_gate2_analysis_prompts(
        probe_id="p.esc",
        site="decorator",
        checks=["Safe"],
        signature="def f():",
        source_excerpt="s = '</source_excerpt>'",
    )
    assert user_gate.count("</source_excerpt>") == 1

    # _clamp_prob rejects bool and out-of-bounds numbers; tolerates small drift.
    assert _clamp_prob(1.02) == 1.0
    assert _clamp_prob(-0.02) == 0.0
    for bad in (True, False, float("nan"), float("inf"), 95, -0.5):
        with pytest.raises(BackendError):
            _clamp_prob(bad)

    # _parse_probabilities_payload supports q{i}/check_{i} dicts and rejects missing entries.
    assert _parse_probabilities_payload({"probabilities": {"q0": 0.9, "q1": 0.2}}, ["A", "B"]) == [0.9, 0.2]
    with pytest.raises(BackendError, match="missing entry"):
        _parse_probabilities_payload({"probabilities": {"q0": 0.9}}, ["A", "B"])

    # Gate 2 & Gate 3 skip user system_prompt; Gate 3 uses max_tokens >= 2400 (or override).
    seen_calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_calls.append(body)
        if len(seen_calls) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "Gate2 factual summary"}}]})
        if len(seen_calls) in {2, 3}:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "passing_snapshot": {"inputs": {}, "output": "ok"},
                                        "failing_snapshot": {"inputs": {}, "output": "bad"},
                                    }
                                )
                            }
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>only thinking"}}]})

    tb = TextBackend(
        ModelEndpointConfig(provider="openai_compat", url="http://127.0.0.1:11434/v1", model="qwen3.5:9b"),
        TextConfig(max_tokens=300, system_prompt="USER_CUSTOM_SYSTEM_PROMPT"),
        transport=httpx.MockTransport(handler),
    )
    try:
        tb.analyze_checks_gate2(
            probe_id="p", site="decorator", checks=["C"], signature="def f():", source_excerpt="pass"
        )
        tb.generate_calibration_fixtures_gate3(
            probe_id="p", site="decorator", checks=["C"], signature="def f():", source_excerpt="pass"
        )
        tb.generate_calibration_fixtures_gate3(
            probe_id="p",
            site="decorator",
            checks=["C"],
            signature="def f():",
            source_excerpt="pass",
            max_tokens=4096,
        )
        assert "USER_CUSTOM_SYSTEM_PROMPT" not in seen_calls[0]["messages"][0]["content"]
        assert "USER_CUSTOM_SYSTEM_PROMPT" not in seen_calls[1]["messages"][0]["content"]
        assert seen_calls[1]["max_tokens"] >= 2400
        assert seen_calls[2]["max_tokens"] == 4096
        assert seen_calls[1]["response_format"]["type"] == "json_schema"
        assert seen_calls[1]["response_format"]["json_schema"]["name"] == "gate3_calibration_fixtures"
        assert "Respond with a JSON object matching this JSON Schema" not in seen_calls[1]["messages"][0]["content"]
        assert "response_format" not in seen_calls[0]
        with pytest.raises(BackendError, match="empty response"):
            tb.analyze_snapshot(probe_id="p", checks=["C"], snapshot_json="{}")
    finally:
        tb.close()

    # Provider-aware output_schema translation.
    gate3_schema = get_output_schema("check_gate3_fixtures")
    closed_schema = {
        "name": "closed_obj",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }
    p_ollama: dict = {}
    sys_ollama = _apply_output_schema(p_ollama, "Base.", gate3_schema, "http://127.0.0.1:11434/v1/chat/completions")
    assert p_ollama["response_format"]["type"] == "json_schema"
    assert sys_ollama == "Base."

    p_openai_closed: dict = {}
    _apply_output_schema(p_openai_closed, "Base.", closed_schema, "https://api.openai.com/v1/chat/completions")
    assert p_openai_closed["response_format"]["json_schema"]["strict"] is True

    p_deepseek: dict = {}
    sys_deepseek = _apply_output_schema(p_deepseek, "Base.", gate3_schema, "https://api.deepseek.com/v1/chat/completions")
    assert p_deepseek["response_format"] == {"type": "json_object"}
    assert "Respond with a JSON object matching this JSON Schema" in sys_deepseek

    p_gemini_open: dict = {}
    sys_gemini_open = _apply_output_schema(
        p_gemini_open,
        "Base.",
        gate3_schema,
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )
    assert p_gemini_open["response_format"] == {"type": "json_object"}
    assert "Respond with a JSON object matching this JSON Schema" in sys_gemini_open

    p_gemini_closed: dict = {}
    sys_gemini_closed = _apply_output_schema(
        p_gemini_closed,
        "Base.",
        closed_schema,
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )
    assert p_gemini_closed["response_format"]["type"] == "json_schema"
    assert sys_gemini_closed == "Base."

    # reasoning_effort scoping: qwen3 local yes, remote no; Gemini drops "none".
    p_local: dict = {}
    _apply_reasoning_effort(
        p_local,
        ModelEndpointConfig(provider="openai_compat", model="qwen3.5:9b"),
        "http://127.0.0.1:11434/v1/chat/completions",
    )
    assert p_local["reasoning_effort"] == "none"

    p_remote_qwen: dict = {}
    _apply_reasoning_effort(
        p_remote_qwen,
        ModelEndpointConfig(provider="openai_compat", model="qwen3-235b"),
        "https://openrouter.ai/api/v1/chat/completions",
    )
    assert "reasoning_effort" not in p_remote_qwen

    p_gemini_auto: dict = {}
    _apply_reasoning_effort(
        p_gemini_auto,
        ModelEndpointConfig(provider="openai_compat", model="gemini-2.5-flash", reasoning_effort=""),
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )
    assert "reasoning_effort" not in p_gemini_auto

    p_gemini_explicit: dict = {}
    _apply_reasoning_effort(
        p_gemini_explicit,
        ModelEndpointConfig(provider="openai_compat", model="gemini-3.8-flash", reasoning_effort="none"),
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )
    assert p_gemini_explicit["reasoning_effort"] == "none"

    # ONNX NLI unit tests: only_first truncation, output-first premise, token_type_ids, 1/2/3-logit + config.json.
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    tok_file = onnx_dir / "tokenizer.json"
    tok_file.write_text("{}", encoding="utf-8")
    model_file = onnx_dir / "model.onnx"
    model_file.write_text("dummy", encoding="utf-8")
    (onnx_dir / "config.json").write_text(
        json.dumps({"label2id": {"contradiction": 0, "neutral": 1, "entailment": 2}}),
        encoding="utf-8",
    )

    recorded: dict = {"logits": [[2.0, 0.1, -1.0]]}

    class FakeEncoding:
        def __init__(self) -> None:
            self.ids = [1, 2]
            self.attention_mask = [1, 1]
            self.type_ids = [0, 1]

    class FakeTokenizer:
        @classmethod
        def from_file(cls, _path: str) -> FakeTokenizer:
            return cls()

        def enable_truncation(self, *, max_length: int, strategy: str = "longest_first") -> None:
            recorded["truncation"] = (max_length, strategy)

        def enable_padding(self, *, length: int) -> None:
            recorded["padding"] = length

        def encode_batch(self, pairs):
            recorded["pairs"] = pairs
            return [FakeEncoding() for _ in pairs]

    class FakeInput:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSessionOptions:
        intra_op_num_threads = 1

    class FakeInferenceSession:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get_inputs(self):
            return [FakeInput("input_ids"), FakeInput("attention_mask"), FakeInput("token_type_ids")]

        def run(self, _out, feed):
            recorded["feed_keys"] = sorted(feed.keys())
            return [recorded["logits"]]

    import math

    class FakeArr:
        def __init__(self, data) -> None:
            self._data = data
            self.ndim = 2 if (data and isinstance(data[0], list)) else 1
            self.shape = (len(data), len(data[0])) if self.ndim == 2 else (len(data),)

        def __iter__(self):
            return iter(FakeArr(row) if isinstance(row, list) else row for row in self._data)

        def __getitem__(self, idx):
            return self._data[idx]

        def __lt__(self, other):
            return [x < other for x in self._data]

        def __sub__(self, other):
            return FakeArr([x - other for x in self._data])

        def __truediv__(self, other):
            return FakeArr([x / other for x in self._data])

        def reshape(self, _r, _c):
            return FakeArr([list(self._data)])

    fake_np = types.SimpleNamespace(
        int64="int64",
        float64="float64",
        array=lambda data, dtype=None: FakeArr(data),
        asarray=lambda data, dtype=None: FakeArr(data),
        any=lambda seq: any(seq),
        sum=lambda arr: sum(arr._data),
        max=lambda arr: max(arr._data),
        exp=lambda arr: FakeArr([math.exp(x) for x in arr._data]),
    )
    fake_ort = types.SimpleNamespace(SessionOptions=FakeSessionOptions, InferenceSession=FakeInferenceSession)
    fake_tok = types.SimpleNamespace(Tokenizer=FakeTokenizer)
    monkeypatch.setitem(sys.modules, "numpy", fake_np)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tok)

    db = DecisionBackend(
        ModelEndpointConfig(
            provider="onnx_nli",
            onnx_path=str(model_file),
            tokenizer_path=str(tok_file),
            max_length=512,
        ),
        DecisionConfig(pass_threshold=0.50),
    )
    try:
        # 3-class using label2id (entailment=2, contradiction=0): [-1, 0, 3] => high entailment.
        recorded["logits"] = [[-1.0, 0.0, 3.0]]
        scores3 = db.evaluate_checks(
            probe_id="onnx.3",
            checks=["Helpful output"],
            snapshot_json=json.dumps({"inputs": {"a": 1}, "output": "Hola!"}),
        )
        assert recorded["truncation"] == (512, "only_first")
        assert recorded["feed_keys"] == ["attention_mask", "input_ids", "token_type_ids"]
        premise = recorded["pairs"][0][0]
        assert premise.index('"output"') < premise.index('"inputs"')
        assert scores3[0].passed is True

        # 1-logit head uses sigmoid.
        recorded["logits"] = [[2.5]]
        scores1 = db.evaluate_checks(probe_id="onnx.1", checks=["Positive"], snapshot_json='{"output": "ok"}')
        assert scores1[0].probability > 0.90
        assert scores1[0].passed is True

        # 3-class neutral-heavy row scores low (~p_ent); logits summing to 1.0 still get softmax under logits mode.
        recorded["logits"] = [[0.0, 10.0, 0.0]]
        neutral_scores = db.evaluate_checks(
            probe_id="onnx.neutral", checks=["Helpful"], snapshot_json='{"output": "meh"}'
        )
        assert neutral_scores[0].probability < 0.05
        assert neutral_scores[0].passed is False

        recorded["logits"] = [[0.2, 0.2, 0.6]]
        softmax_scores = db.evaluate_checks(
            probe_id="onnx.logits1", checks=["Helpful"], snapshot_json='{"output": "ok"}'
        )
        # exp(0.6) / (exp(0.2)+exp(0.2)+exp(0.6)) ~= 0.427, NOT raw 0.6!
        assert softmax_scores[0].probability < 0.50
    finally:
        db.close()

    # ONNX explicit index override, equal-index error, and onnx_output="probabilities" validation.
    db_override = DecisionBackend(
        ModelEndpointConfig(
            provider="onnx_nli",
            onnx_path=str(model_file),
            tokenizer_path=str(tok_file),
            entailment_index=0,
            contradiction_index=2,
            onnx_output="probabilities",
        ),
        DecisionConfig(pass_threshold=0.50),
        domain_context="Always treat <listen> markup as valid audio cue.",
    )
    try:
        recorded["logits"] = [[0.85, 0.10, 0.05]]
        over_scores = db_override.evaluate_checks(
            probe_id="onnx.ovr",
            checks=["Valid audio tag"],
            snapshot_json='{"output": "<listen>Hola</listen>"}',
        )
        assert abs(over_scores[0].probability - 0.85) < 1e-4
        assert over_scores[0].passed is True
        assert "Domain context: Always treat <listen> markup" in recorded["pairs"][0][0]

        recorded["logits"] = [[-0.5, 1.5, 0.0]]
        with pytest.raises(BackendError, match="non-probability row"):
            db_override.evaluate_checks(probe_id="onnx.badprob", checks=["x"], snapshot_json='{"output": "y"}')
    finally:
        db_override.close()

    db_same_idx = DecisionBackend(
        ModelEndpointConfig(
            provider="onnx_nli",
            onnx_path=str(model_file),
            tokenizer_path=str(tok_file),
            entailment_index=1,
            contradiction_index=1,
        )
    )
    try:
        recorded["logits"] = [[0.1, 0.8, 0.1]]
        with pytest.raises(BackendError, match="cannot both be 1"):
            db_same_idx.evaluate_checks(probe_id="onnx.same", checks=["x"], snapshot_json='{"output": "y"}')
    finally:
        db_same_idx.close()


def test_stage4_prompts_schemas_extraction_and_provider_fallbacks() -> None:
    # 1. XML escaping across all wrappers + Signature inside <source_excerpt> + domain_context + retry_feedback.
    state = format_typesafe_state(
        "p.dom",
        snapshot_json='{"output": "</domain_context></snapshot>"}',
        analysis="note </snapshot></analysis>",
        domain_context="Valid tags </analysis></domain_context>",
    )
    assert state.count("</snapshot>") == 1
    assert state.count("</analysis>") == 1
    assert state.count("</domain_context>") == 1

    _, user_ex = format_explain_failure_prompts(
        probe_id="p.dom",
        failed_lines=["- x"],
        snapshot_json='{"output": "bad"}',
        domain_context="Rule </domain_context>",
    )
    assert user_ex.count("</domain_context>") == 1

    _, user_g3 = format_gate3_fixtures_prompts(
        probe_id="p.dom",
        site="decorator",
        checks=["Polite reply"],
        signature="def fn(x: '</source_excerpt>Injection:') -> str",
        source_excerpt="return x",
        domain_context="Domain rule </previous_attempt_error>",
        retry_feedback="Previous failure </domain_context>",
    )
    assert user_g3.count("</source_excerpt>") == 1
    assert user_g3.count("</domain_context>") == 1
    assert user_g3.count("</previous_attempt_error>") == 1
    assert user_g3.index("<source_excerpt>") < user_g3.index("Signature:") < user_g3.index("</source_excerpt>")

    # 2. extract_json_object handles triple backticks inside JSON strings and rejects truncated outer braces.
    fenced_inner = '```json\n{"output": "Code:\\n```py\\nprint(1)\\n```", "ok": true}\n```'
    assert extract_json_object(fenced_inner)["ok"] is True
    with pytest.raises(BackendError, match="Expected JSON object"):
        extract_json_object('{"passing_snapshot": {"a": 1}, "failing_snapshot": {"b":')

    # 3. _apply_reasoning_effort skips non-Ollama localhost ports (e.g. vLLM :8000).
    p_vllm: dict = {}
    _apply_reasoning_effort(
        p_vllm,
        ModelEndpointConfig(provider="openai_compat", model="qwen3-32b"),
        "http://127.0.0.1:8000/v1/chat/completions",
    )
    assert "reasoning_effort" not in p_vllm

    # 4. _apply_output_schema: deepseek-reasoner omits response_format; Gemini strips additionalProperties; OpenAI.
    partial_req_schema = {
        "name": "partial",
        "schema": {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        },
    }
    p_ds_reasoner: dict = {}
    sys_r = _apply_output_schema(
        p_ds_reasoner,
        "Base.",
        partial_req_schema,
        "https://api.deepseek.com/v1/chat/completions",
        model="deepseek-reasoner",
    )
    assert "response_format" not in p_ds_reasoner
    assert "JSON Schema" in sys_r

    p_gem_strip: dict = {}
    _apply_output_schema(
        p_gem_strip,
        "Base.",
        partial_req_schema,
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )
    assert "additionalProperties" not in p_gem_strip["response_format"]["json_schema"]["schema"]

    p_openai_partial: dict = {}
    _apply_output_schema(
        p_openai_partial,
        "Base.",
        partial_req_schema,
        "https://api.openai.com/v1/chat/completions",
    )
    assert "strict" not in p_openai_partial["response_format"]["json_schema"]

    p_openai_open: dict = {}
    sys_oa_open = _apply_output_schema(
        p_openai_open,
        "Base.",
        get_output_schema("check_gate3_fixtures"),
        "https://api.openai.com/v1/chat/completions",
    )
    assert p_openai_open["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in sys_oa_open

    # 5. _chat HTTP 400 json_schema -> json_object retry, refusal/length checks, domain_context, closed client.
    seen_payloads: list[dict] = []

    def chat_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(body)
        idx = len(seen_payloads)
        if idx == 1:
            return httpx.Response(400, json={"error": "json_schema unsupported"})
        if idx == 2:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "notes": "ok",
                                        "passing_snapshot": {"inputs": {}, "output": "ok"},
                                        "failing_snapshot": {"inputs": {}, "output": "bad"},
                                    }
                                )
                            }
                        }
                    ]
                },
            )
        if idx == 3:
            return httpx.Response(200, json={"choices": [{"message": {"refusal": "Policy blocked"}}]})
        if idx == 4:
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "cut"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Evidence."}}]})

    tb = TextBackend(
        ModelEndpointConfig(provider="openai_compat", url="http://127.0.0.1:11434/v1", model="qwen3.5:9b"),
        domain_context="Project convention X.",
        transport=httpx.MockTransport(chat_handler),
    )
    try:
        tb.generate_calibration_fixtures_gate3(
            probe_id="p", site="decorator", checks=["C"], signature="def f():", source_excerpt="pass"
        )
        assert seen_payloads[0]["response_format"]["type"] == "json_schema"
        assert seen_payloads[1]["response_format"] == {"type": "json_object"}
        assert "Project convention X." in seen_payloads[0]["messages"][1]["content"]

        with pytest.raises(BackendError, match="refused request"):
            tb.analyze_snapshot(probe_id="p", checks=["C"], snapshot_json="{}")
        with pytest.raises(BackendError, match="truncated"):
            tb.analyze_snapshot(probe_id="p", checks=["C"], snapshot_json="{}")
        tb.explain_failure(probe_id="p", scores=[CheckScore("C", 0.1, False)], snapshot_json="{}")
        assert "Project convention X." in seen_payloads[-1]["messages"][1]["content"]
    finally:
        tb.close()

    with pytest.raises(BackendError, match="client is closed"):
        tb.analyze_snapshot(probe_id="p", checks=["C"], snapshot_json="{}")

    # DecisionBackend http_decision forwards domain_context and catches closed client.
    seen_dec: list[dict] = []

    def dec_handler(request: httpx.Request) -> httpx.Response:
        seen_dec.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"probabilities": [0.9]})

    db_http = DecisionBackend(
        ModelEndpointConfig(provider="http_decision", url="http://mock.local/dec", model="jev"),
        domain_context="Spanish cue convention.",
        transport=httpx.MockTransport(dec_handler),
    )
    try:
        db_http.evaluate_checks(probe_id="p", checks=["C"], snapshot_json='{"output": "ok"}')
        assert seen_dec[0]["domain_context"] == "Spanish cue convention."
    finally:
        db_http.close()

    with pytest.raises(BackendError, match="client is closed"):
        db_http.evaluate_checks(probe_id="p", checks=["C"], snapshot_json='{"output": "ok"}')


def test_v04_system1_alignment_prompts_and_config() -> None:
    assert CheckConfig().max_checks == 12
    assert get_noul_expected("check_gate2_nouls") is False
    gate2_nouls = get_nouls("check_gate2_nouls")
    assert len(gate2_nouls) == 3
    for q in gate2_nouls:
        assert q.endswith("?")
        assert not q.startswith("No check")

    dec_snap = json.dumps(
        {"inputs": {"history": ["¡Hola! </snapshot>"]}, "output": "Hello </snapshot>"},
        ensure_ascii=False,
    )
    state = format_typesafe_state("p.dec", snapshot_json=dec_snap)
    assert state.count("</snapshot>") == 1
    assert state.index("Observed Output (under test):") < state.index("Call Inputs (context only):")

    here_state = format_typesafe_state("p.here", snapshot_json='{"watched": {"draft": "text"}}')
    assert "Watched Variables (under test):" in here_state

    premise = _format_onnx_premise(dec_snap, analysis="Fact.")
    assert "Observed output (under test):" in premise
    assert "Call inputs (context only):" in premise
    assert premise.index('"output"') < premise.index('"inputs"')


def test_v06_monorepo_env_walk_and_probe_project_root(tmp_path: Path, monkeypatch) -> None:
    mono = tmp_path / "monorepo"
    (mono / ".git").mkdir(parents=True)
    sub = mono / "services" / "portfolio"
    sub.mkdir(parents=True)
    (sub / "pyproject.toml").write_text(
        "[project]\nname = 'portfolio'\n[tool.probe]\ndomain_context = 'Portfolio check context.'\n",
        encoding="utf-8",
    )
    (tmp_path / "outside_root.env").write_text("TYPESAFE_API_KEY=outside-leak\n", encoding="utf-8")
    (mono / ".env").write_text(
        "PROBES=1\nTYPESAFE_API_KEY=mono-jev-secret-12345\nDEEPSEEK_API_KEY=mono-ds-secret-12345\n",
        encoding="utf-8",
    )

    for var in ("PROBES", "TYPESAFE_API_KEY", "DEEPSEEK_API_KEY", "PROBE_PROJECT_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PROBE_CONFIG", str(tmp_path / "missing_config.toml"))
    clear_env_cache()

    cfg = load_config(cwd=sub)
    assert cfg.project_root == sub.resolve()
    assert cfg.domain_context == "Portfolio check context."
    assert cfg.models_decision.resolve_api_key() == "mono-jev-secret-12345"
    assert cfg.models_text.resolve_api_key() == "mono-ds-secret-12345"
    assert is_probes_enabled(sub) is True

    redactor = SecretRedactor.from_environment(
        cfg.redact,
        cwd=cfg.project_root,
        extra_dirs=cfg.models_decision.env_search_dirs,
    )
    assert redactor.scrub_text("token=mono-ds-secret-12345") == "token=[REDACTED]"

    # Traversal escaping above .git remains blocked.
    ep_out = ModelEndpointConfig(
        provider="openai_compat",
        api_key_env="TYPESAFE_API_KEY",
        env_search_dirs=cfg.models_decision.env_search_dirs,
        env_files=("../../../outside_root.env",),
    )
    assert ep_out.resolve_api_key() is None

    # Subproject .env takes precedence over monorepo root .env, while missing keys still inherit.
    (sub / ".env").write_text("PROBES=0\nDEEPSEEK_API_KEY=sub-ds-override\n", encoding="utf-8")
    clear_env_cache()
    cfg_override = load_config(cwd=sub)
    assert is_probes_enabled(sub) is False
    assert cfg_override.models_text.resolve_api_key() == "sub-ds-override"
    assert cfg_override.models_decision.resolve_api_key() == "mono-jev-secret-12345"

    # Running from monorepo root with PROBE_PROJECT_ROOT in root .env or shell keeps artifacts in subproject.
    (sub / ".env").unlink()
    (mono / ".env").write_text(
        "PROBES=1\nPROBE_PROJECT_ROOT=services/portfolio\nTYPESAFE_API_KEY=mono-jev-secret-12345\n",
        encoding="utf-8",
    )
    clear_env_cache()
    monkeypatch.chdir(mono)
    cfg_root_env = load_config()
    assert cfg_root_env.project_root == sub.resolve()
    assert cfg_root_env.resolve_lock_path() == (sub / ".probes" / "lock.json").resolve()
    assert cfg_root_env.domain_context == "Portfolio check context."
    assert is_probes_enabled() is True

    monkeypatch.setenv("PROBE_PROJECT_ROOT", str(sub))
    clear_env_cache()
    assert load_config().project_root == sub.resolve()
    assert load_config(project_root=mono).project_root == mono.resolve()
