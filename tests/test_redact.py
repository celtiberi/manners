"""Tests for runtime secret collection, request-field masking, and bounded serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from probe import probe
from probe.config import RedactConfig
from probe.redact import (
    REDACTED,
    SecretRedactor,
    serialize_call_inputs,
    serialize_call_snapshot,
    serialize_here_snapshot,
)


def test_secret_redactor_collects_dotenv_and_env_ignoring_examples_and_noise(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "# comment",
                "export DEEPSEEK_API_KEY=sk-secret-value-12345",
                "CUSTOM_SAAS_CRED='custom_secret_token_999'",
                'DB_PASS="test-password-123" # inline comment',
                "DATABASE_URL=postgres://appuser:dsnsecret99@db.internal:5432/prod",
                "LOG_LEVEL=production",
                "ENVIRONMENT=production",
                "PORT=8080",
                "DEBUG=true",
                "DATA_DIR=/var/lib/myapp/data",
                "PUBLIC_URL=https://api.example.com/v1",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "WEBHOOK_HOOK=https://hooks.example.com/emit?token=hooktok123456\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.example").write_text(
        "OPENAI_API_KEY=sk-example-not-real-12345\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_live_secret_key_abcdef")
    monkeypatch.setenv("RUNTIME_SECRET_SHORTER", "supersecret")
    monkeypatch.setenv("RUNTIME_SECRET_LONGER", "supersecret_extended")
    monkeypatch.setenv("APP_HOST", "localhost")

    redactor = SecretRedactor.from_environment(RedactConfig(env_files=(".env",)), cwd=tmp_path)

    sample = (
        "k1=sk-secret-value-12345 "
        "k2=custom_secret_token_999 "
        "k3=test-password-123 "
        "k4=ts_live_secret_key_abcdef "
        "long=supersecret_extended "
        "short=supersecret "
        "dsn=postgres://appuser:dsnsecret99@db.internal:5432/prod "
        "pw=dsnsecret99 "
        "hook=hooktok123456 "
        "example=sk-example-not-real-12345 "
        "env=production port=8080 dir=/var/lib/myapp/data url=https://api.example.com/v1"
    )
    scrubbed = redactor.scrub_text(sample)

    for leaked in (
        "sk-secret-value-12345",
        "custom_secret_token_999",
        "test-password-123",
        "ts_live_secret_key_abcdef",
        "supersecret_extended",
        "supersecret",
        "dsnsecret99",
        "hooktok123456",
        "postgres://appuser:dsnsecret99@db.internal:5432/prod",
    ):
        assert leaked not in scrubbed
    assert "_extended" not in scrubbed  # longest-first prevented partial leftover

    # Examples and ordinary non-secret values stay readable for debugging.
    assert "sk-example-not-real-12345" in scrubbed
    assert "env=production" in scrubbed
    assert "port=8080" in scrubbed
    assert "dir=/var/lib/myapp/data" in scrubbed
    assert "https://api.example.com/v1" in scrubbed


def test_request_field_masking_and_positional_binding() -> None:
    redactor = SecretRedactor(secrets=(), extra_keys=("x_session_id", "ssn"))

    def login(username: str, password: str, headers: dict[str, str]) -> dict[str, str]:
        return {"user": username, "cookie": headers.get("Cookie", "")}

    inputs = serialize_call_inputs(
        login,
        ("alice", "hunter2"),
        {"headers": {"Cookie": "sid=abc", "Authorization": "Bearer xyz", "X-Session-ID": "sess-1", "Accept": "json"}},
        redactor,
    )
    assert inputs["username"] == "alice"
    assert inputs["password"] == REDACTED
    assert inputs["headers"]["Cookie"] == REDACTED
    assert inputs["headers"]["Authorization"] == REDACTED
    assert inputs["headers"]["X-Session-ID"] == REDACTED
    assert inputs["headers"]["Accept"] == "json"

    snapshot_json = serialize_call_snapshot(
        inputs,
        redactor=redactor,
        return_value={"user": "alice", "ssn": "123-45-6789", "ok": True},
        has_return=True,
    )
    parsed = json.loads(snapshot_json)
    assert parsed["output"]["ssn"] == REDACTED
    assert parsed["output"]["ok"] is True


def test_bounded_serialization_and_oversized_secret_scrub() -> None:
    huge_secret = "SECRET_" + ("X" * 5000) + "_END"
    redactor = SecretRedactor(secrets=(huge_secret,))

    @dataclass
    class Node:
        value: str
        items: list[int]

    circular: list[object] = []
    circular.append(circular)

    payload_json = serialize_here_snapshot(
        {
            "leaked": huge_secret,
            "nested": {"a": {"b": {"c": {"d": "too_deep"}}}},
            "many": list(range(30)),
            "node": Node(value="ok", items=[1, 2]),
            "loop": circular,
        },
        redactor,
        max_depth=4,
        max_items=5,
        max_str_bytes=200,
    )
    parsed = json.loads(payload_json)["watched"]
    assert parsed["leaked"] == REDACTED
    assert huge_secret[:20] not in payload_json
    assert len(parsed["many"]) == 6  # 5 items + truncation marker
    assert "25 more items" in str(parsed["many"][-1])
    assert "<circular>" in json.dumps(parsed["loop"])
    assert probe.REDACTED if hasattr(probe, "REDACTED") else True
