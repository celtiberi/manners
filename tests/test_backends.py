"""Offline HTTP transport tests for DecisionBackend and TextBackend."""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx

from probe.backends import CheckScore, DecisionBackend, TextBackend, extract_json_object, strip_thinking
from probe.config import DecisionConfig, ModelEndpointConfig, TextConfig, load_config


def test_strip_thinking_and_extract_json() -> None:
    raw = '<think>internal reasoning</think>\n```json\n{"approved": true, "reason": "good check"}\n```'
    assert strip_thinking(raw) == '```json\n{"approved": true, "reason": "good check"}\n```'
    assert extract_json_object(raw) == {"approved": True, "reason": "good check"}

    unclosed = "Some <think>never finished"
    assert strip_thinking(unclosed) == "Some"


def test_typesafe_jev_systemone_dialect_and_dotenv_key_resolution(tmp_path: Path, monkeypatch) -> None:
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


def test_logprob_bool_provider_and_missing_logprob_fallback() -> None:
    lp_yes = math.log(0.80)
    lp_no = math.log(0.20)
    calls = {"n": 0}

    def handler_logprob(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "YES"},
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": "YES", "logprob": lp_yes},
                                        {"token": "NO", "logprob": lp_no},
                                    ]
                                }
                            ]
                        },
                    }
                ]
            },
        )

    backend = DecisionBackend(
        ModelEndpointConfig(
            provider="logprob_bool",
            url="http://localhost:11434/v1",
            model="qwen3.5:9b",
        ),
        DecisionConfig(pass_threshold=0.50),
        transport=httpx.MockTransport(handler_logprob),
    )
    try:
        scores = backend.evaluate_checks(
            probe_id="p.bool",
            checks=["Greeting present"],
            snapshot_json='{"output": "Hola!"}',
        )
    finally:
        backend.close()

    assert len(scores) == 1
    assert scores[0].passed is True
    assert abs(scores[0].probability - 0.80) < 1e-3

    # Fallback when server returns choices without logprobs.
    def handler_fallback(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if body.get("logprobs"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "YES"}, "logprobs": None}]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"probabilities": [0.15, 0.88]}'}}]},
        )

    fb_backend = DecisionBackend(
        ModelEndpointConfig(provider="logprob_bool", url="http://localhost:11434/v1", model="local"),
        DecisionConfig(pass_threshold=0.50),
        transport=httpx.MockTransport(handler_fallback),
    )
    try:
        fb_scores = fb_backend.evaluate_checks(
            probe_id="p.fb",
            checks=["c1", "c2"],
            snapshot_json="{}",
        )
    finally:
        fb_backend.close()

    assert [s.passed for s in fb_scores] == [False, True]
    assert [s.probability for s in fb_scores] == [0.15, 0.88]


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
