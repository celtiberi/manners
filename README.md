# `probe`

Plain-English runtime behavioral checks (`@probe` and `probe.here`) and call-ordered debug logs for Python. See [`SPEC.md`](./SPEC.md) for the full design.

## Why This Exists

Python `assert`s, type checks, and regexes catch structural bugs (`x is not None`, `len(items) > 0`, valid JSON), and you should keep using them for that. They don't catch behavioral bugs where a function returns a valid string or dict whose content is wrong—especially in code that calls LLMs, formats prompts, or combines retrieved context.

Two examples where normal assertions come up short:

- An assistant endpoint returns `200 OK` and a valid reply, so tests pass—even though the reply tells the user an action was completed (like issuing a refund or updating an account) when no tool call actually ran.
- A RAG endpoint summarizes a policy doc using the exact words and numbers from the retrieved chunks, yet flips the meaning—telling a user they qualify when the chunk says their tier is excluded.

Normally you only spot those by manually clicking through your app or reading through raw prompt/response logs later. `probe` lets you put checks right on the function (`@probe`) or mid-function line (`probe.here`) so execution stops on the exact call that broke them.

## How It Works

When probes are enabled during local development or testing, `probe` captures the inputs and outputs at each `@probe` or `probe.here` call (redacting secrets first) and checks whether the runtime behavior matches your plain-English rules—halting execution right there if a check fails, and staying completely inactive when turned off in production.

Instead of asking a chat LLM to output `"pass"` or `"fail"` directly—which tends to hedge and drift—`probe` splits the work across two configurable model roles:

- **Decision model (`probe.type.decision`, default):** Evaluates discrete yes/no checks directly and returns a calibrated probability per check (`pass` or `fail`). Fast and deterministic enough for most checks.
- **Text LLM + Decision model (`probe.type.text`):** When a check requires open-ended reasoning first (such as comparing a generated answer against source documents), a text LLM analyzes the snapshot and hands its findings to the decision model for the final `pass` or `fail` verdict.

Both roles are configured once in `config.toml` and can point at cloud APIs or local models without changing your code. Before probes run, `probe check` validates your check prompts against synthetic pass/fail cases and rejects checks that belong as normal Python `assert`s.

## Install

From a GitHub Release wheel (recommended in containers/CI — no `git` binary required):

```sh
uv tool install https://github.com/celtiberi/probe/releases/download/v0.4.0/probe-0.4.0-py3-none-any.whl
```

In a project virtual environment (wheel or git):

```sh
uv add https://github.com/celtiberi/probe/releases/download/v0.4.0/probe-0.4.0-py3-none-any.whl
# or via pip:
# pip install https://github.com/celtiberi/probe/releases/download/v0.4.0/probe-0.4.0-py3-none-any.whl
# or from git main:
# uv add git+https://github.com/celtiberi/probe.git
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
        "Draft does NOT answer confidently when retrieved_chunks do not cover user_question",
    ],
    id="rag.draft_answer",
    kind=probe.type.text,
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
# or in CI / non-interactive test runs:
PROBES=1 PROBE_NO_PDB=1 pytest
```

To skip `.probes/lock.json` enforcement during quick local iteration, set `PROBE_SKIP_LOCK=1`. Setting `PROBES=""` or `PROBES=0` in the shell overrides `PROBES=1` in `.env`.

To run multi-turn simulations or soak tests without stopping on the first failure, set `PROBE_NO_HALT=1` (or `[runtime].halt_on_fail = false`) and run `probe report --check` at the end to inspect failures and exit `1` if any probe failed.

To temporarily suppress probes inside mocked unit tests (including code using worker threads) when `PROBES=1` is set globally, use `with probe.disabled():` (or `from probe import disabled`).

4. Inspect results in call-entry order (works while a server is running too, via per-run JSONL logs):

```sh
probe report --all
probe report --check
probe log tutor.opening_turn rag.*
probe runs
```

## Writing Effective Checks & Fixtures

- **Keep `probe.type.decision` checks short and atomic:** A single-pass decision model evaluates each check in parallel without multi-step reasoning. Prefer several short single-fact statements (`max_checks = 12` default) naming `output` or specific watched variables directly over one long compound sentence joining multiple `and`/`without`/`unless` clauses.
- **Use `kind=probe.type.text` for comparison or reasoning:** When a check requires comparing `output` against multi-turn `inputs` or retrieved context first, use `kind=probe.type.text` so `[models.text]` extracts factual evidence before `[models.decision]` scores the rule.
- **Add regression fixtures in `.probes/fixtures/<probe_id>.json`:** Place optional JSON files with either a single `"pass"` / `"fail"` snapshot object or lists of historical regression snapshots (`"pass": [{...}, ...]`, `"fail": [{...}, ...]`). `probe check` backtests every listed pass and fail case whenever checks or fixtures change.

## CLI

| Command | Purpose |
|---|---|
| `probe check [paths...]` | Audit checks (Gate 1–2), run pass/fail calibration (Gate 3), and update `.probes/lock.json`. |
| `probe list [paths...]` | List discovered `@probe` and `probe.here` sites with lock status. |
| `probe report [--run ID] [--all] [--check]` | Show call-ordered summary of failures/errors (or all events with `--all`); exit `1` on failures with `--check`. |
| `probe log <probe_id>... [--payload]` | Interleave chronological events across one or more probe IDs or globs. |
| `probe runs` | List recorded runs and pass/fail/error counts. |
| `probe clean [--keep N]` | Prune older run logs from `.probes/runs` (default keeps last 20). |

## Configuration (`~/.config/probe/config.toml`)

Project overrides can also be placed under `[tool.probe]` in `pyproject.toml`. (If `~/.config/probe/config.toml` already configures remote endpoints, overriding them to different remote hosts from `pyproject.toml` requires `PROBE_TRUST_PROJECT_ENDPOINTS=1`.)

```toml
domain_context = ""              # Optional domain conventions (passed to Gate 2/3 & runtime checks)

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

# Local text LLM via Ollama (paired with your configured [models.decision]):
# [models.text]
# provider = "openai_compat"
# url = "http://127.0.0.1:11434/v1"
# model = "qwen3.5:9b"

[decision]
pass_threshold = 0.50

[text]
detail = "standard"
max_tokens = 1024
explain_decision_failures = true
system_prompt = ""

[rate_limit]
max_calls = 20
window_seconds = 10.0
max_calls_per_run = -1

[check]
max_checks = 12
max_check_chars = 200
require_check = true
max_workers = 4

[redact]
env_files = [".env"]
extra_keys = []

[runtime]
runs_dir = ".probes/runs"
lock_path = ".probes/lock.json"
strict_api = true
halt_on_fail = true
```

## License

Functional Source License, Version 1.1, with an Apache 2.0 future license (`FSL-1.1-Apache-2.0`). See [`LICENSE`](./LICENSE).
