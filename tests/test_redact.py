"""Tests for runtime secret collection, request-field masking, and bounded serialization."""

from __future__ import annotations

import collections
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

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


def test_redactor_edge_cases_webhooks_depth_namedtuple_slots_lock_and_subdirs(
    tmp_path: Path, monkeypatch
) -> None:
    proj = tmp_path / "proj"
    subdir = proj / "sub"
    subdir.mkdir(parents=True)
    (proj / ".env").write_text(
        "\n".join(
            [
                "POSTGRES_USER=postgres",
                "APP_NAME=spanish-tutor",
                "SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T000/B000/SecretHookToken123",
                'MULTILINE_KEY="line_one_secret\nline_two_secret"',
                'QUOTE_SECRET="abc\\"def\\\\ghi"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (subdir / ".env.sub").write_text("SUB_API_TOKEN=sub_token_987654\n", encoding="utf-8")

    redactor = SecretRedactor.from_environment(
        RedactConfig(env_files=(".env",)),
        cwd=proj,
        extra_dirs=(subdir,),
    )

    # Ordinary non-secret config values stay unredacted while webhook/subdir/JSON-escaped secrets scrub.
    text = (
        "postgres spanish-tutor "
        "https://hooks.slack.com/services/T000/B000/SecretHookToken123 "
        "sub_token_987654 "
        'line_one_secret\nline_two_secret json={"k":"abc\\"def\\\\ghi"}'
    )
    scrubbed = redactor.scrub_text(text)
    assert "postgres" in scrubbed
    assert "spanish-tutor" in scrubbed
    assert "SecretHookToken123" not in scrubbed
    assert "sub_token_987654" not in scrubbed
    assert "line_one_secret" not in scrubbed
    assert 'abc\\"def' not in scrubbed

    class CredTuple(NamedTuple):
        user: str
        x_api_key: str

    class SlottedCreds:
        __slots__ = ("label", "access_token", "_internal")

        def __init__(self, label: str, access_token: str) -> None:
            self.label = label
            self.access_token = access_token
            self._internal = "hidden"

    @dataclass
    class GuardedCtx:
        lock: threading.Lock
        password: str

    snap_json = serialize_here_snapshot(
        {
            "deep": {"a": {"b": {"c": {"password": "must_not_leak_at_max_depth"}}}},
            "nt": CredTuple("bob", "secret_key_val"),
            "slots": SlottedCreds("svc", "tok_secret_99"),
            "ctx": GuardedCtx(lock=threading.Lock(), password="pw_secret"),
        },
        redactor,
        max_depth=4,
    )
    assert "must_not_leak_at_max_depth" not in snap_json
    parsed = json.loads(snap_json)["watched"]
    assert parsed["deep"]["a"]["b"]["c"] == "<dict len=1 max_depth>"
    assert parsed["nt"] == {"user": "bob", "x_api_key": REDACTED}
    assert parsed["slots"] == {"label": "svc", "access_token": REDACTED}
    assert parsed["ctx"]["password"] == REDACTED
    assert "_thread.lock" in str(parsed["ctx"]["lock"]) or "lock" in str(parsed["ctx"]["lock"]).lower()

    # 1. Secret shrinkage near truncation boundary & multibyte bytes slicing never leak partial secrets.
    long_secret = "sk_live_" + ("A" * 92)
    r_shrink = SecretRedactor(secrets=(long_secret,))
    out_shrink = json.loads(
        serialize_here_snapshot(
            {"text": ((long_secret + " ") * 42) + long_secret},
            r_shrink,
            max_str_bytes=4096,
        )
    )["watched"]["text"]
    assert "sk_live_" not in out_shrink
    assert out_shrink.endswith("...[truncated]")

    mb_secret = "SECRET_" + ("秘" * 100) + "!"
    r_mb = SecretRedactor(secrets=(mb_secret,))
    out_mb = json.loads(
        serialize_here_snapshot(
            {"raw": (b"x" * 4050) + mb_secret.encode("utf-8")},
            r_mb,
            max_str_bytes=4096,
        )
    )["watched"]["raw"]
    assert "SECRET_" not in out_mb
    assert out_mb.endswith("...[truncated]")

    # 2. Passwordless postgres DSN leaves username intact; deque & field(init=False) serialize safely;
    #    URL-encoded secrets are scrubbed.
    dsn_env = tmp_path / "dsn"
    dsn_env.mkdir()
    (dsn_env / ".env").write_text(
        "DATABASE_URL=postgres://postgres@localhost:5432/mydb\n"
        "API_TOKEN=tok+plus/slash=123\n",
        encoding="utf-8",
    )
    r_dsn = SecretRedactor.from_environment(RedactConfig(), cwd=dsn_env)
    scrub_dsn = r_dsn.scrub_text("postgres q=tok%2Bplus%2Fslash%3D123")
    assert "postgres" in scrub_dsn
    assert "tok%2B" not in scrub_dsn

    @dataclass
    class PartialDataclass:
        x: int
        uninit: str = field(init=False)

    snap_cont = json.loads(
        serialize_here_snapshot(
            {
                "history": collections.deque([{"password": "dynamic_secret_pw"}]),
                "partial": PartialDataclass(x=7),
            },
            r_dsn,
        )
    )["watched"]
    assert snap_cont["history"] == [{"password": REDACTED}]
    assert snap_cont["partial"] == {"x": 7}
