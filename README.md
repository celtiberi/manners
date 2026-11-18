# `probe`

Plain-English runtime behavioral checks (`@probe` and `probe.here`) and call-ordered debug logs for Python. See [`SPEC.md`](./SPEC.md) for the full design.

## Why This Exists

Python `assert`s, type checks, and regexes catch structural bugs (`x is not None`, `len(items) > 0`, valid JSON), and you should keep using them for that. They don't catch behavioral bugs where a function returns a valid string or dict whose content is wrong—especially in code that calls LLMs, formats prompts, or combines retrieved context.

Two examples where normal assertions come up short:

- An assistant endpoint returns `200 OK` and a valid reply, so tests pass—even though the reply tells the user an action was completed (like issuing a refund or updating an account) when no tool call actually ran.
- A RAG endpoint summarizes a policy doc using the exact words and numbers from the retrieved chunks, yet flips the meaning—telling a user they qualify when the chunk says their tier is excluded.

Normally you only spot those by manually clicking through your app or reading through raw prompt/response logs later. `probe` lets you put checks right on the function (`@probe`) or mid-function line (`probe.here`) so execution stops on the exact call that broke them.

## How It Works

- **Off by default (`PROBES=0` or unset):** `@probe` returns the original function unwrapped and `probe.here` does nothing. Probes can stay in application code with zero latency or API cost in production.
- **On when testing or debugging (`PROBES=1` in shell or `.env`):** Start your web server, CLI script, or `pytest` run with `PROBES=1`. When a probed function or `probe.here` line runs, `probe` snapshots inputs, return values, and watched variables, scrubs secrets collected from `.env*` and `os.environ` (plus `password`, `authorization`, and `cookie` fields), and evaluates the checks inline.
- **Stops on the live frame:** If any check fails (or an optional `when="..."` breakpoint condition holds), `probe` logs the snapshot and per-check scores to `.probes/probes.db` and stops before the function returns—opening `pdb` on that caller frame in an interactive terminal, or raising `ProbeHaltError` in tests and scripts.
- **Guardrails against misuse and noisy grading:**
  - **`probe check`:** Run on new or edited probes (approved hashes are saved in `.probes/lock.json`). It rejects checks that belong in regular Python `assert`s or regexes, and dry-runs each probe against a passing and failing sample before running live code.
  - **Loop protection (`PROBE_MISUSE`):** Each probe is guarded by a sliding-window limit (`5` calls per `10.0` seconds by default) so accidentally calling a probe inside a tight loop halts the process before wasting API calls.
  - **Two configurable model roles (`decision` + `text`):** `probe.type.decision` (default) uses a decision model (such as TypeSafe Jev or a local endpoint) that scores each check in `0.0–1.0` -> `pass` or `fail`. `probe.type.text` runs a chat LLM first when open-ended analysis is helpful, then sends that analysis to the decision model for the final `pass` or `fail`.

## Install

```sh
uv tool install git+https://github.com/celtiberi/probe.git
```

In a project virtual environment:

```sh
uv add git+https://github.com/celtiberi/probe.git
```

## Quick Start

1. Attach checks to a function or intermediate step:

```python
from probe import probe

@probe(
    [
        "Tutor opens with a short Spanish greeting or phrase",
        "Tutor gives brief teaching or instructions to help the student",
        "Tutor asks the student to say something in Spanish",
        "Tutor does NOT give away the exact Spanish sentence the student is expected to say",
    ],
    id="tutor.opening_turn",
)
def generate_opening_turn(lesson, history) -> str:
    ...
```

Mid-function watchpoint:

```python
probe.here(
    [
        "Draft answer stays strictly grounded in retrieved chunks",
        "Draft answer directly addresses the user question",
    ],
    id="rag.draft_answer",
    kind=probe.type.text,
    when="draft answers confidently even though retrieved_chunks do not cover user_question",
    user_question=question,
    retrieved_chunks=chunks,
    draft=draft_text,
)
```

2. Run the pre-run check (populates `.probes/lock.json`):

```sh
probe check
```

3. Run your server or tests with probes enabled (either in the shell or by setting `PROBES=1` in `.env`):

```sh
PROBES=1 uvicorn app:app --reload
# or
PROBES=1 pytest
```

4. Inspect results in call-entry order (works while a server is running too, via SQLite WAL):

```sh
probe report --all
probe log tutor.opening_turn rag.*
probe runs
```

## CLI

| Command | Purpose |
|---|---|
| `probe check [paths...]` | Audit checks (Gate 1–2), run pass/fail calibration (Gate 3), and update `.probes/lock.json`. |
| `probe list [paths...]` | List discovered `@probe` and `probe.here` sites with lock status. |
| `probe report [--run ID] [--all]` | Show call-ordered summary of failures/errors (or all events with `--all`). |
| `probe log <probe_id>... [--payload]` | Interleave chronological events across one or more probe IDs or globs. |
| `probe runs` | List recorded runs and pass/fail/error counts. |
| `probe clean [--keep N]` | Prune older runs from `.probes/probes.db` (default keeps last 20). |

## Configuration (`~/.config/probe/config.toml`)

Project overrides can also be placed under `[tool.probe]` in `pyproject.toml`.

```toml
[models.decision]
provider = "typesafe_jev"
url = "https://api.typesafe.ai/v1/systemone"
model = "jev-latest"
api_key_env = "TYPESAFE_API_KEY"

[models.text]
provider = "openai_compat"
url = "https://api.deepseek.com/v1"
model = "deepseek-chat"
api_key_env = "DEEPSEEK_API_KEY"

# Local profile example:
# [models.decision]
# provider = "http_decision"
# url = "http://<host>.<tailnet>.ts.net:8780/v1/evaluate"
# model = "local-decision"
#
# [models.text]
# provider = "openai_compat"
# url = "http://<host>.<tailnet>.ts.net:11434/v1"
# model = "qwen3.5:9b"

[decision]
pass_threshold = 0.50

[text]
detail = "standard"
max_tokens = 400
explain_decision_failures = true
system_prompt = ""

[rate_limit]
max_calls = 5
window_seconds = 10.0
max_calls_per_run = -1

[check]
max_checks = 5
max_check_chars = 200
require_check = true

[redact]
env_files = [".env"]
extra_keys = []

[runtime]
db_path = ".probes/probes.db"
lock_path = ".probes/lock.json"
```

## License

Functional Source License, Version 1.1, with an Apache 2.0 future license (`FSL-1.1-Apache-2.0`). See [`LICENSE`](./LICENSE).
