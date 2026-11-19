# `manners`

[![CI](https://github.com/celtiberi/manners/actions/workflows/ci.yml/badge.svg)](https://github.com/celtiberi/manners/actions/workflows/ci.yml)

Plain-English runtime behavioral checks (`@manners` and `manners.here`) and call-ordered debug logs for Python. Inactive when `MANNERS` is unset; when `MANNERS=1` is active during dev or testing, `manners` evaluates inputs and outputs against plain-English rules and halts on the exact call that fails. See [`SPEC.md`](./SPEC.md) and [`docs/case-study-spanish-tutor.md`](./docs/case-study-spanish-tutor.md).

Python `assert`s catch structural bugs (`x is not None`, valid JSON), not behavioral bugs where a valid response has the wrong meaning—such as a tutor giving away the answer it just asked for, an assistant claiming a refund ran when no tool executed, or a RAG reply contradicting its retrieved chunks:

```python
from manners import manners

@manners(
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

Validate checks once, then run with checks enabled (`manners check` rejects checks that belong as plain Python `assert`s):

```sh
manners check
MANNERS=1 pytest evals/test_tutor_turns.py
```

When a rule fails at runtime, `manners` stops execution on that call and prints which rule broke and why:

```text
[MANNERS HALT] call #003 tutor.opening_turn (tutor/turn.py:61)
  ✓ [0.97] Tutor opens with a short Spanish greeting or phrase
  ✓ [0.91] Tutor gives brief teaching or instructions to help the student
  ✓ [0.89] Tutor asks the student to say something in Spanish
  ✗ [0.06] Tutor does NOT give away the exact Spanish sentence the student is expected to say
  analysis: Tutor tells the student: 'Try saying: Quiero un café por favor', giving away the target sentence verbatim.
```

For mid-function checks comparing intermediate values (like a draft against retrieved context), use `manners.here(...)` with `kind=manners.type.text`:

```python
manners.here(
    [
        "Draft answer stays strictly grounded in retrieved chunks",
        "Draft answer directly addresses the user question",
        "Draft does NOT answer confidently when retrieved_chunks do not cover user_question",
    ],
    id="rag.draft_answer",
    kind=manners.type.text,
    user_question=question,
    retrieved_chunks=chunks,
    draft=draft_text,
)
```

## How It Works

- **Zero overhead in production, inline halts in dev/tests:** When `MANNERS` is unset (or `0`), `@manners` and `manners.here` are no-ops. When `MANNERS=1`, `manners` captures function arguments or watched variables (redacting secrets first), writes chronological events to `.manners/runs/<run_id>.jsonl`, and raises `MannersHaltError` on the exact failing call.
- **Split across decision and text models:** Asking a chat LLM to reply `"pass"` or `"fail"` tends to hedge and drift, so `manners` splits grading across two configurable roles:
  - **Decision model (`manners.type.decision`, default):** Evaluates discrete yes/no checks directly and returns a calibrated probability per check.
  - **Text LLM + Decision model (`manners.type.text`):** A text LLM first analyzes the snapshot against source context, then passes its findings to the decision model for the final `pass` or `fail` verdict.
- **Pre-run `manners check` gate:** Before checks run, `manners check` audits checks, calibrates them against pass/fail cases, and locks approved hashes in `.manners/lock.json`.

## Install

```sh
# Install CLI globally (or run with `uvx manners check`)
uv tool install manners

# Add to a project
uv add manners
# Optional local ONNX NLI decision runtime (no cloud decision API key required):
# uv add "manners[onnx]"
# or via pip:
# pip install manners
```

Fallback (from a GitHub Release wheel or git main — useful in containers/CI):

```sh
uv add https://github.com/celtiberi/manners/releases/download/v0.7.0/manners-0.7.0-py3-none-any.whl
# uv add git+https://github.com/celtiberi/manners.git
```

## Quick Start

1. **Configure models and API keys** in shell env or `.env` (see [Configuration](#configuration-configmannersconfigtoml) to override endpoints in `~/.config/manners/config.toml` or `[tool.manners]` in `pyproject.toml`):
   - **`[models.decision]` (scores yes/no checks):** Defaults to **TypeSafe Jev** (`TYPESAFE_API_KEY` from [typesafe.ai](https://typesafe.ai)), or install `"manners[onnx]"` with `provider = "onnx_nli"` for local NLI weights (such as `DeBERTa-v3-base-mnli-fever-anli`).
   - **`[models.text]` (runs `manners check`, `manners.type.text` analysis, and failure explanations):** Defaults to **DeepSeek** (`DEEPSEEK_API_KEY`), or point `url` / `model` / `api_key_env` at **OpenAI** (`OPENAI_API_KEY`), **Gemini** (`GEMINI_API_KEY`), or local **Ollama** (`qwen3.5:9b`, no API key).

   ```sh
   TYPESAFE_API_KEY=...
   DEEPSEEK_API_KEY=...
   ```

2. **Validate your checks** (populates `.manners/lock.json`):

   ```sh
   manners check
   ```

3. **Run your server or tests with `MANNERS=1`** (in the shell or `.env`):

   ```sh
   MANNERS=1 uvicorn app:app --reload
   # or in CI / non-interactive test runs:
   MANNERS=1 MANNERS_NO_PDB=1 pytest
   ```

   - **Skip lockfile during iteration:** Set `MANNERS_SKIP_LOCK=1`. Setting `MANNERS=""` or `MANNERS=0` in the shell overrides `MANNERS=1` in `.env`.
   - **Non-halting soak tests:** Set `MANNERS_NO_HALT=1` (or `[runtime].halt_on_fail = false`) to log all failures across multi-turn runs via a bounded background pool (`[runtime].max_workers = 4`) and exit `1` via `manners report --check` at the end (`manners.flush()` or process/pytest teardown drains pending evaluations).
   - **Controlling `pytest` runs when `MANNERS=1` is in `.env`:**
     - Add `--no-manners` to `[tool.pytest.ini_options] addopts` so mocked unit tests skip checks across fixture `setup`, test `call`, and `teardown` without `conftest.py` boilerplate.
     - Mark manners-backed integration tests with `@pytest.mark.manners` to exempt them from `--no-manners` (inheriting `MANNERS` from `.env`/env so keyless CI stays a no-op; pass `--manners` on the CLI or `@pytest.mark.manners(force=True)` to enable unconditionally). Mark exclusions with `@pytest.mark.no_manners`.
     - The `pytest` plugin drains background/daemon-thread evaluations across `setup`/`call`/`teardown` so uncaught `MannersHaltError`s fail the test cleanly (`manners.flush()` drains threads that have already entered `@manners`/`manners.here`; `join()` any slow daemon thread that has not reached `manners.here` before the test returns).
     - In plain Python code or fixtures, use stackable `with manners.disabled():` and `with manners.enabled():` (`from manners import disabled, enabled, flush`).

4. **Inspect results in call-entry order** (works live against per-run JSONL logs while a server is running):

   ```sh
   manners report --all
   manners report --check
   manners log tutor.opening_turn rag.*
   manners runs
   ```

```text
Run 2026-11-18T15:10:04Z (cmd: pytest evals/test_tutor_turns.py) — HALTED on FAIL at call #003
Checks: 2 active | Calls: 3 (2 pass, 1 fail, 0 error)

Chronological Timeline (ordered by call_seq):
  #001  +0.088s  [PASS]  rag.draft_answer (rag/synthesize.py:94) [kind=text]
        checks:
          ✓ [0.96] Draft answer stays strictly grounded in the retrieved chunks without inventing figures or dates
          ✓ [0.93] Draft answer directly addresses the user's question
        analysis: Response cites only figures from chunk #1 (filing 2026-11-16) and directly answers the question.

  #002  +0.412s  [PASS]  tutor.opening_turn (tutor/turn.py:61) [kind=decision]
        checks:  4/4 passed (min p=0.89)

  #003  +0.785s  [FAIL]  tutor.opening_turn (tutor/turn.py:61) [kind=decision]  <-- HALT TRIGGER
        checks:
          ✓ [0.97] Tutor opens with a short Spanish greeting or phrase
          ✓ [0.91] Tutor gives brief teaching or instructions to help the student
          ✓ [0.89] Tutor asks the student to say something in Spanish
          ✗ [0.06] Tutor does NOT give away the exact Spanish sentence the student is expected to say
        analysis: Tutor tells the student: 'Try saying: Quiero un café por favor', giving away the target sentence verbatim.
```

## Writing Effective Checks & Fixtures

- **Keep `manners.type.decision` checks short and atomic:** A single-pass decision model evaluates each check in parallel without multi-step reasoning. Prefer several short single-fact statements (`max_checks = 12` default) naming `output` or specific watched variables directly over one long compound sentence joining multiple `and`/`without`/`unless` clauses.
- **Exclude noisy service handles with `ignore=`:** On `@manners(..., ignore=("data_dir", "llm_client"))`, ignored parameters are skipped completely before snapshot serialization, excluded from Gate 3 fixtures, and checked via AST/runtime.
- **Use `kind=manners.type.text` for comparison or reasoning:** When a check requires comparing `output` against multi-turn `inputs` or retrieved context first, use `kind=manners.type.text` so `[models.text]` extracts factual evidence before `[models.decision]` scores the rule.
- **Add regression fixtures in `.manners/fixtures/<check_id>.json`:** Place optional JSON files with either a single `"pass"` / `"fail"` snapshot object or lists of historical regression snapshots (`"pass": [{...}, ...]`, `"fail": [{...}, ...]`). `manners check` prints Gate 3 pass/fail calibration margins (`pass min p`, `fail max p`) and backtests every listed case whenever checks or fixtures change. Run `manners fixture <check_id>` to promote recorded pass/fail snapshots out of `.manners/runs/*.jsonl` (or `manners fixture <check_id> --init` to scaffold missing entries from the check's AST signature/watched keys).
- **Monorepo `.env` & subproject root (`MANNERS_PROJECT_ROOT` / `--root`):** `.env` lookups walk from `project_root` up to the enclosing `.git` root (nearest `.env` takes precedence), keeping API keys at the repo root while storing `.manners/` and `pyproject.toml` config in each subproject. Set `MANNERS_PROJECT_ROOT=services/portfolio` (in env or root `.env`) or pass `--root <dir>` on CLI commands when running from a monorepo root.

## CLI

| Command | Purpose |
|---|---|
| `manners check [paths...] [--root DIR]` | Audit checks (Gate 1–2), run pass/fail calibration (Gate 3), show calibration margins, and update `.manners/lock.json`. |
| `manners list [paths...] [--root DIR]` | List discovered `@manners` and `manners.here` sites with lock status. |
| `manners fixture <id> [--run ID] [--init] [--root DIR]` | Promote recorded pass/fail run snapshots into `.manners/fixtures/<id>.json` or scaffold from AST. |
| `manners report [--run ID] [--all] [--check] [--root DIR]` | Show call-ordered summary of failures/errors (or all events with `--all`); exit `1` on failures with `--check`. |
| `manners log <check_id>... [--payload] [--root DIR]` | Interleave chronological events across one or more check IDs or globs. |
| `manners runs [--root DIR]` | List recorded runs and pass/fail/error counts. |
| `manners clean [--keep N] [--root DIR]` | Prune older run logs from `.manners/runs` (default keeps last 20). |

## Configuration (`~/.config/manners/config.toml`)

Project overrides can also be placed under `[tool.manners]` in `pyproject.toml`. (If `~/.config/manners/config.toml` already configures remote endpoints, overriding them to different remote hosts from `pyproject.toml` requires `MANNERS_TRUST_PROJECT_ENDPOINTS=1`.)

```toml
domain_context = ""              # Optional domain conventions (passed to Gate 2/3 & runtime checks)

[models.decision]
provider = "typesafe_jev"
url = "https://api.typesafe.ai/v1/systemone"
model = "jev-latest"
api_key_env = "TYPESAFE_API_KEY"

# Alternative: local ONNX NLI decision runtime (install "manners[onnx] @ ..."):
# [models.decision]
# provider = "onnx_nli"
# onnx_path = "~/.cache/manners/deberta-v3-mnli/model.onnx"
# tokenizer_path = "~/.cache/manners/deberta-v3-mnli/tokenizer.json"

[models.text]
provider = "openai_compat"
url = "https://api.deepseek.com/v1"
model = "deepseek-chat"
api_key_env = "DEEPSEEK_API_KEY"

# Alternatives for [models.text] (OpenAI, Gemini, or local Ollama):
# [models.text]
# provider = "openai_compat"
# url = "https://api.openai.com/v1"
# model = "gpt-4.1-mini"
# api_key_env = "OPENAI_API_KEY"
#
# [models.text]
# provider = "openai_compat"
# url = "https://generativelanguage.googleapis.com/v1beta/openai"
# model = "gemini-3.8-flash"
# api_key_env = "GEMINI_API_KEY"
#
# [models.text]
# provider = "openai_compat"
# url = "http://127.0.0.1:11434/v1"
# model = "qwen3.5:9b"

[decision]
pass_threshold = 0.50

[text]
detail = "standard"
# max_tokens = 2048  # Optional cap; omitted by default (uses model default)
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
runs_dir = ".manners/runs"
lock_path = ".manners/lock.json"
strict_api = true
halt_on_fail = true
max_workers = 4
```

## License

Functional Source License, Version 1.1, with an Apache 2.0 future license (`FSL-1.1-Apache-2.0`). See [`LICENSE`](./LICENSE).
