"""Live end-to-end integration tests against TypeSafe Jev, DeepSeek/Gemini, and local Qwen via Ollama."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from probe import ProbeHaltError, probe, reset_for_testing
from probe.backends import DecisionBackend
from probe.check import run_probe_check
from probe.cli import app
from probe.config import (
    CheckConfig,
    ModelEndpointConfig,
    ProbeConfig,
    RuntimeConfig,
    _parse_dotenv_key,
    load_config,
)
from probe.store import ProbeStore

PROBE_ROOT = Path(__file__).resolve().parents[1]
PROBE_DOTENV = PROBE_ROOT / ".env"

pytestmark = pytest.mark.live


def _resolve_live_keys() -> tuple[str, str, str]:
    ts = os.environ.get("TYPESAFE_API_KEY", "").strip()
    ds = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    gm = os.environ.get("GEMINI_API_KEY", "").strip()
    if PROBE_DOTENV.is_file():
        ts = ts or _parse_dotenv_key(PROBE_DOTENV, "TYPESAFE_API_KEY")
        ds = ds or _parse_dotenv_key(PROBE_DOTENV, "DEEPSEEK_API_KEY")
        gm = gm or _parse_dotenv_key(PROBE_DOTENV, "GEMINI_API_KEY")
    return ts, ds, gm


_TYPESAFE_KEY, _DEEPSEEK_KEY, _GEMINI_KEY = _resolve_live_keys()
_HAS_LIVE_KEYS = bool(_TYPESAFE_KEY and (_DEEPSEEK_KEY or _GEMINI_KEY))
_DEEPSEEK_CHECK_CACHE: bool | None = None
_LOCAL_QWEN_CACHE: bool | None = None


def _deepseek_has_balance() -> bool:
    global _DEEPSEEK_CHECK_CACHE
    if _DEEPSEEK_CHECK_CACHE is not None:
        return _DEEPSEEK_CHECK_CACHE
    if not _DEEPSEEK_KEY:
        _DEEPSEEK_CHECK_CACHE = False
        return False
    try:
        resp = httpx.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_DEEPSEEK_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "OK"}],
                "max_tokens": 2,
            },
            timeout=10.0,
        )
        _DEEPSEEK_CHECK_CACHE = resp.status_code != 402 and resp.is_success
    except httpx.HTTPError:
        _DEEPSEEK_CHECK_CACHE = False
    return _DEEPSEEK_CHECK_CACHE


def _has_local_qwen() -> bool:
    global _LOCAL_QWEN_CACHE
    if _LOCAL_QWEN_CACHE is not None:
        return _LOCAL_QWEN_CACHE
    try:
        resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        _LOCAL_QWEN_CACHE = any(
            isinstance(m, dict) and str(m.get("name", "")).startswith("qwen3.5:9b")
            for m in models
        )
    except Exception:
        _LOCAL_QWEN_CACHE = False
    return _LOCAL_QWEN_CACHE


def _seed_live_project_config(tmp_path: Path, *, skip_lock_require: bool = False) -> ProbeConfig:
    env_lines = [
        f"TYPESAFE_API_KEY={_TYPESAFE_KEY}",
        "SAMPLE_APP_SECRET=sk_live_probe_secret_987654321",
    ]
    use_deepseek = _deepseek_has_balance()
    if use_deepseek and _DEEPSEEK_KEY:
        env_lines.append(f"DEEPSEEK_API_KEY={_DEEPSEEK_KEY}")
    if _GEMINI_KEY:
        env_lines.append(f"GEMINI_API_KEY={_GEMINI_KEY}")
    (tmp_path / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'live-probe-check'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )
    cfg = load_config(cwd=tmp_path)
    if not use_deepseek:
        if not _GEMINI_KEY:
            pytest.skip("DEEPSEEK_API_KEY returned HTTP 402 Insufficient Balance and GEMINI_API_KEY is unset")
        cfg.models_text = ModelEndpointConfig(
            provider="openai_compat",
            url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-3.8-flash",
            api_key_env="GEMINI_API_KEY",
            reasoning_effort="none",
            env_search_dirs=(tmp_path,),
            env_files=cfg.redact.env_files,
        )

    cfg.check = CheckConfig(require_check=not skip_lock_require)
    cfg.runtime = RuntimeConfig(
        runs_dir=str(tmp_path / ".probes" / "runs"),
        lock_path=str(tmp_path / ".probes" / "lock.json"),
    )
    return cfg


@pytest.mark.skipif(
    not _HAS_LIVE_KEYS,
    reason="Requires TYPESAFE_API_KEY and (DEEPSEEK_API_KEY or GEMINI_API_KEY) in env or probe/.env",
)
def test_live_typesafe_jev_spanish_tutor_checks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PROBE_CONFIG", str(tmp_path / "no_user_config.toml"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / ".env").write_text(f"TYPESAFE_API_KEY={_TYPESAFE_KEY}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'live-jev-checks'\n", encoding="utf-8")
    cfg = load_config(cwd=tmp_path)

    assert cfg.models_decision.provider == "typesafe_jev"
    assert cfg.models_decision.resolve_api_key(), "Expected TYPESAFE_API_KEY from seeded .env"

    spec_checks = [
        "Tutor opens with a short Spanish greeting or phrase",
        "Tutor gives brief teaching or instructions to help the student",
        "Tutor asks the student to say something in Spanish",
        "Tutor avoids giving away the exact complete Spanish sentence that the student is asked to say",
    ]

    backend = DecisionBackend(cfg.models_decision, cfg.decision)
    try:
        good_snapshot = json.dumps(
            {
                "inputs": {"lesson": "Ordering coffee politely", "history": []},
                "output": (
                    "¡Hola! When ordering at a café in Spanish, you can start politely with "
                    "'Quisiera...' (I would like) and add 'por favor' at the end. "
                    "Now you try asking for a coffee with milk in Spanish!"
                ),
            }
        )
        good_scores = backend.evaluate_checks(
            probe_id="tutor.opening_turn",
            checks=spec_checks,
            snapshot_json=good_snapshot,
        )
        assert all(s.passed for s in good_scores), f"Unexpected fail on good turn: {good_scores}"

        spoiler_snapshot = json.dumps(
            {
                "inputs": {"lesson": "Ordering coffee politely", "history": []},
                "output": (
                    "¡Hola! Remember to add 'por favor' to sound polite. "
                    "Say this exact sentence in Spanish right now: Quiero un café con leche por favor."
                ),
            }
        )
        spoiler_scores = backend.evaluate_checks(
            probe_id="tutor.opening_turn",
            checks=spec_checks,
            snapshot_json=spoiler_snapshot,
        )
        assert spoiler_scores[3].passed is False, f"Expected spoiler check #4 to fail: {spoiler_scores}"
    finally:
        backend.close()


@pytest.mark.skipif(
    not _HAS_LIVE_KEYS,
    reason="Requires TYPESAFE_API_KEY and (DEEPSEEK_API_KEY or GEMINI_API_KEY) in env or probe/.env",
)
def test_live_probe_check_gates_reject_math_and_calibrate_semantic_probe(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PROBE_CONFIG", str(tmp_path / "no_user_config.toml"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = _seed_live_project_config(tmp_path)

    src_dir = tmp_path / "app"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")

    target_file = src_dir / "tutor.py"
    target_file.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe("Return value equals a + b", id="math.add")
            def add(a: int, b: int) -> int:
                return a + b
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    ok_bad, res_bad = run_probe_check([target_file], config=cfg)
    assert ok_bad is False
    assert len(res_bad) == 1
    assert res_bad[0].status == "failed"
    assert res_bad[0].gate == "gate2"

    target_file.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe(
                [
                    "Tutor opens with a short Spanish greeting or phrase",
                    "Tutor gives brief instructions without spoiling the exact target Spanish sentence",
                ],
                id="tutor.opening_turn",
            )
            def generate_opening_turn(lesson: str, student_level: str) -> str:
                \"\"\"Generate an introductory Spanish tutoring turn for a student.\"\"\"
                return ""
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    ok_good, res_good = run_probe_check([target_file], config=cfg)
    assert ok_good is True, f"Live check calibration failed: {res_good}"
    assert res_good[0].status == "passed"
    assert cfg.resolve_lock_path().is_file()


@pytest.mark.skipif(
    not _HAS_LIVE_KEYS,
    reason="Requires TYPESAFE_API_KEY and (DEEPSEEK_API_KEY or GEMINI_API_KEY) in env or probe/.env",
)
def test_live_runtime_decision_text_redaction_and_cli_report(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PROBE_CONFIG", str(tmp_path / "no_user_config.toml"))
    monkeypatch.setenv("PROBES", "1")
    monkeypatch.setenv("PROBE_NO_PDB", "1")
    monkeypatch.setenv("PROBE_SKIP_LOCK", "1")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    cfg = _seed_live_project_config(tmp_path, skip_lock_require=True)
    reset_for_testing(config=cfg)

    @probe(
        [
            "Output includes a Spanish greeting",
            "Output invites the student to respond in Spanish",
        ],
        id="tutor.live_turn",
        kind=probe.type.decision,
    )
    def live_tutor_turn(student: str, debug_secret: str) -> str:
        return "¡Hola! Today let's practice ordering breakfast. How would you ask for eggs in Spanish?"

    try:
        out = live_tutor_turn("Sam", "sk_live_probe_secret_987654321")
        assert "¡Hola!" in out

        probe.here(
            [
                "Draft answer matches the price stated in retrieved_chunks without inventing numbers",
            ],
            id="rag.live_draft",
            kind=probe.type.text,
            retrieved_chunks=["Menu note: Espresso at Cafe Sol costs $3.75."],
            draft="Espresso at Cafe Sol costs $3.75.",
        )

        @probe(
            [
                "Tutor avoids giving the student the exact full Spanish sentence verbatim",
            ],
            id="tutor.live_fail",
            kind=probe.type.decision,
        )
        def live_spoiler_turn(topic: str) -> str:
            return "Say this exact sentence in Spanish verbatim: Quiero un café con leche por favor."

        with pytest.raises(ProbeHaltError):
            live_spoiler_turn("coffee")
    finally:
        rt = probe.flush_for_testing()
        reset_for_testing()

    assert rt is not None
    store = ProbeStore(cfg.resolve_runs_dir(), create=False)
    try:
        run = store.get_run(rt.run_id)
        events = store.list_events(rt.run_id)
    finally:
        store.close()

    assert run is not None
    assert run.status == "halted"
    assert run.halt_call_seq == 3
    assert [e.call_seq for e in events] == [1, 2, 3]
    assert [e.verdict for e in events] == ["pass", "pass", "fail"]

    # Redaction check: secret value collected from .env must never appear in persisted snapshot.
    assert "[REDACTED]" in events[0].payload_json
    assert "sk_live_probe_secret_987654321" not in events[0].payload_json
    assert (_TYPESAFE_KEY not in events[0].payload_json), "API key leaked into snapshot"
    if _DEEPSEEK_KEY:
        assert (_DEEPSEEK_KEY not in events[0].payload_json), "API key leaked into snapshot"
    if _GEMINI_KEY:
        assert (_GEMINI_KEY not in events[0].payload_json), "API key leaked into snapshot"

    # Text pipeline + decision failure explanation both produced analysis strings.
    assert events[1].text_analysis
    assert events[2].text_analysis

    # CLI report & log against live DB.
    runner = CliRunner()
    rep = runner.invoke(app, ["report", "--all"])
    assert rep.exit_code == 0
    assert "#001" in rep.stdout and "#002" in rep.stdout and "#003" in rep.stdout
    assert rep.stdout.index("  #001  ") < rep.stdout.index("  #002  ") < rep.stdout.index("  #003  ")

    log_res = runner.invoke(app, ["log", "tutor.*", "rag.*"])
    assert log_res.exit_code == 0
    assert "tutor.live_turn" in log_res.stdout
    assert "rag.live_draft" in log_res.stdout
    assert "tutor.live_fail" in log_res.stdout


@pytest.mark.skipif(
    not _TYPESAFE_KEY,
    reason="Requires TYPESAFE_API_KEY in env or probe/.env",
)
def test_live_jev_decision_with_local_qwen_text_pipeline(tmp_path: Path, monkeypatch) -> None:
    if not _has_local_qwen():
        pytest.skip("Requires local Ollama at http://127.0.0.1:11434 with qwen3.5:9b")
    monkeypatch.setenv("PROBE_CONFIG", str(tmp_path / "no_user_config.toml"))
    monkeypatch.setenv("PROBES", "1")
    monkeypatch.setenv("PROBE_NO_PDB", "1")
    monkeypatch.setenv("PROBE_SKIP_LOCK", "1")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    (tmp_path / ".env").write_text(f"TYPESAFE_API_KEY={_TYPESAFE_KEY}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'jev-qwen-check'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )
    cfg = load_config(cwd=tmp_path)
    cfg.models_decision = ModelEndpointConfig(
        provider="typesafe_jev",
        url="https://api.typesafe.ai/v1/systemone",
        model="jev-latest",
        api_key_env="TYPESAFE_API_KEY",
        env_search_dirs=(tmp_path,),
        env_files=cfg.redact.env_files,
    )
    cfg.models_text = ModelEndpointConfig(
        provider="openai_compat",
        url="http://127.0.0.1:11434/v1",
        model="qwen3.5:9b",
        timeout_s=60.0,
    )
    cfg.check = CheckConfig(require_check=False)
    cfg.runtime = RuntimeConfig(
        runs_dir=str(tmp_path / ".probes" / "runs"),
        lock_path=str(tmp_path / ".probes" / "lock.json"),
    )

    # 1. Pre-run check: Gate 2 rejects deterministic math & mixed lists; Gates 2-3 pass semantic @probe + probe.here.
    src_dir = tmp_path / "app"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    target = src_dir / "tutor.py"

    target.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe("Return value equals a + b", id="math.add")
            def add(a: int, b: int) -> int:
                return a + b
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    ok_math, res_math = run_probe_check([target], config=cfg)
    assert ok_math is False
    assert res_math[0].gate == "gate2"

    target.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe(["Summary captures key risks from filing", "len(output) > 0"], id="summary.mixed")
            def summarize(filing: str) -> str:
                return filing
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    ok_mixed, res_mixed = run_probe_check([target], config=cfg)
    assert ok_mixed is False
    assert res_mixed[0].gate == "gate2"

    target.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe(
                [
                    "Tutor opens with a short Spanish greeting or phrase",
                    "Tutor prompts student to respond in Spanish without spoiling the exact sentence",
                ],
                id="tutor.opening_turn",
            )
            def generate_opening_turn(lesson: str) -> str:
                \"\"\"Generate a brief introductory Spanish tutoring turn.\"\"\"
                return ""


            def check_policy_draft(draft: str) -> str:
                \"\"\"Verify policy draft stays aligned with source exclusion rules.\"\"\"
                probe.here(
                    "Draft stays faithful to source_rule without flipping exclusions",
                    id="rag.draft",
                    kind=probe.type.text,
                    source_rule="Tier B is excluded",
                    draft=draft,
                )
                return draft
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    ok_sem, res_sem = run_probe_check([target], config=cfg)
    assert ok_sem is True, f"Jev + local qwen3.5:9b calibration failed: {res_sem}"
    assert [r.status for r in res_sem] == ["passed", "passed"]
    assert cfg.resolve_lock_path().is_file()

    # 2. Live runtime: decision pass, text pipeline pass, and decision spoiler halt with Qwen explanation.
    reset_for_testing(config=cfg)

    @probe(
        [
            "Output opens with a short Spanish greeting",
            "Output asks the student to order in Spanish",
        ],
        id="tutor.local_pass",
        kind=probe.type.decision,
    )
    def local_good_turn(topic: str) -> str:
        return (
            "¡Hola! When ordering breakfast, you can begin politely with 'Quisiera...'. "
            "Now try asking for eggs and coffee in Spanish!"
        )

    try:
        assert "¡Hola!" in local_good_turn("breakfast")

        probe.here(
            ["Draft states that tier B is excluded instead of saying tier B qualifies"],
            id="rag.local_draft",
            kind=probe.type.text,
            retrieved_chunks=["Policy rule: Tier A qualifies for reimbursement; Tier B is excluded."],
            draft="Tier B is excluded from reimbursement under this policy.",
        )

        @probe(
            ["Tutor does NOT give away the exact complete Spanish sentence for the student to say"],
            id="tutor.local_fail",
            kind=probe.type.decision,
        )
        def local_bad_turn(topic: str) -> str:
            return "¡Hola! Say this exact Spanish sentence verbatim: Quiero un café con leche por favor."

        with pytest.raises(ProbeHaltError):
            local_bad_turn("coffee")
    finally:
        rt = probe.flush_for_testing()
        reset_for_testing()

    assert rt is not None
    store = ProbeStore(cfg.resolve_runs_dir(), create=False)
    try:
        run = store.get_run(rt.run_id)
        events = store.list_events(rt.run_id)
    finally:
        store.close()

    assert run is not None and run.status == "halted" and run.halt_call_seq == 3
    assert [(e.call_seq, e.verdict) for e in events] == [(1, "pass"), (2, "pass"), (3, "fail")]
    assert events[1].text_analysis
    assert events[2].text_analysis
