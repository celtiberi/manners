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
    _apply_reasoning_effort,
    _clamp_prob,
    _parse_probabilities_payload,
    extract_json_object,
    strip_thinking,
)
from probe.config import DecisionConfig, ModelEndpointConfig, TextConfig, load_config
from probe.prompts import (
    compose_system_prompt,
    format_analyze_snapshot_prompts,
    format_gate2_analysis_prompts,
    format_typesafe_state,
    get_nouls,
    get_prompt,
    load_prompt_catalog,
)


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

    # Gate 2 & Gate 3 do not prepend user system_prompt; Gate 3 sets max_tokens >= 1200; empty fails.
    seen_calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_calls.append(body)
        if len(seen_calls) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "Gate2 factual summary"}}]})
        if len(seen_calls) == 2:
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
        assert "USER_CUSTOM_SYSTEM_PROMPT" not in seen_calls[0]["messages"][0]["content"]
        assert "USER_CUSTOM_SYSTEM_PROMPT" not in seen_calls[1]["messages"][0]["content"]
        assert seen_calls[1]["max_tokens"] >= 1200
        assert seen_calls[1]["response_format"] == {"type": "json_object"}
        assert "response_format" not in seen_calls[0]
        with pytest.raises(BackendError, match="empty response"):
            tb.analyze_snapshot(probe_id="p", checks=["C"], snapshot_json="{}")
    finally:
        tb.close()

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
    finally:
        db.close()
