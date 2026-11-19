# `probe`

[![CI](https://github.com/celtiberi/probe/actions/workflows/ci.yml/badge.svg)](https://github.com/celtiberi/probe/actions/workflows/ci.yml)

Plain-English runtime behavioral checks (`@probe` and `probe.here`) and call-ordered debug logs for Python. See [`SPEC.md`](./SPEC.md) for the full design and [`docs/case-study-spanish-tutor.md`](./docs/case-study-spanish-tutor.md) for a real-world multi-stage tutor integration report.

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

> **Note:** Install from the GitHub Release wheel or git URL below—do **not** run bare `pip install probe` or `uv add probe`, which resolves to an unrelated legacy package on PyPI.

From a GitHub Release wheel (recommended in containers/CI — no `git` binary required):

```sh
uv tool install https://github.com/celtiberi/probe/releases/download/v0.6.0/probe-0.6.0-py3-none-any.whl
```

In a project virtual environment (wheel or git):

```sh
uv add https://github.com/celtiberi/probe/releases/download/v0.6.0/probe-0.6.0-py3-none-any.whl
# Optional local ONNX NLI decision runtime (no cloud decision API key required):
# uv add "probe[onnx] @ https://github.com/celtiberi/probe/releases/download/v0.6.0/probe-0.6.0-py3-none-any.whl"
# or via pip:
# pip install https://github.com/celtiberi/probe/releases/download/v0.6.0/probe-0.6.0-py3-none-any.whl
# or from git main:
# uv add git+https://github.com/celtiberi/probe.git
```

## Quick Start

1. Configure your models and API keys (`probe` splits grading into two roles, configured once in `~/.config/probe/config.toml` or `[tool.probe]` in `pyproject.toml`, reading keys from shell env or `.env`):
   - **`[models.decision]` (scores yes/no checks):** Defaults to **TypeSafe Jev** (`TYPESAFE_API_KEY` from [typesafe.ai](https://typesafe.ai)). Without a TypeSafe key, install `"probe[onnx] @ ..."` and point `provider = "onnx_nli"` at local NLI weights (such as `DeBERTa-v3-base-mnli-fever-anli` — see [Configuration](#configuration-configprobeconfigtoml)).
   - **`[models.text]` (runs `probe check` calibration, `probe.type.text` analysis, and failure explanations):** Defaults to **DeepSeek** (`DEEPSEEK_API_KEY`), or point `url` / `model` / `api_key_env` at **OpenAI** (`OPENAI_API_KEY`), **Gemini** (`GEMINI_API_KEY`), or local **Ollama** (`qwen3.5:9b`, no API key).

Minimal `.env` when using defaults (no `config.toml` needed; see [Configuration](#configuration-configprobeconfigtoml) to swap in OpenAI, Gemini, Ollama, or local ONNX):

```sh
TYPESAFE_API_KEY=...
DEEPSEEK_API_KEY=...
```

2. Attach checks to a function or intermediate step:

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

3. Run the pre-run check (populates `.probes/lock.json`):

```sh
probe check
```

4. Run your server or tests with probes enabled (either in the shell or by setting `PROBES=1` in `.env`):

```sh
PROBES=1 uvicorn app:app --reload
# or in CI / non-interactive test runs:
PROBES=1 PROBE_NO_PDB=1 pytest
```

To skip `.probes/lock.json` enforcement during quick local iteration, set `PROBE_SKIP_LOCK=1`. Setting `PROBES=""` or `PROBES=0` in the shell overrides `PROBES=1` in `.env`.

To run multi-turn simulations or soak tests without stopping on the first failure, set `PROBE_NO_HALT=1` (or `[runtime].halt_on_fail = false`) and run `probe report --check` at the end to inspect failures and exit `1` if any probe failed. In non-halting soak mode, model grading dispatches onto a bounded background pool (`[runtime].max_workers = 4`) after capturing `call_seq` and snapshots synchronously; call `probe.flush()` (or let process exit / pytest teardown drain automatically) to flush pending evaluations.

To control probes inside `pytest` when `PROBES=1` is set in `.env`:
- Add `--no-probes` to `[tool.pytest.ini_options] addopts` so mocked unit tests skip probes across fixture `setup`, test `call`, and `teardown` without extra `conftest.py` boilerplate.
- Mark probe-backed integration tests with `@pytest.mark.probes` to exempt them from `--no-probes` (so `PROBES=1` in `.env` runs them locally while keyless CI with `PROBES` unset keeps them a no-op; pass `--probes` on the CLI or `@pytest.mark.probes(force=True)` to enable unconditionally). Mark exclusions with `@pytest.mark.no_probes`.
- The built-in `pytest` plugin drains background/daemon-thread probe evaluations across `setup`/`call`/`teardown` so uncaught `ProbeHaltError`s fail the test cleanly (`probe.flush()` drains threads that have already entered `@probe`/`probe.here`; if a test spawns a daemon thread doing slow upstream work before reaching `probe.here`, `join()` it before returning).
- In plain Python code or fixtures, use stackable `with probe.disabled():` and `with probe.enabled():` (`from probe import disabled, enabled, flush`).

5. Inspect results in call-entry order (works while a server is running too, via per-run JSONL logs):

```sh
probe report --all
probe report --check
probe log tutor.opening_turn rag.*
probe runs
```

### Example Output

```text
Run 2026-11-18T15:10:04Z (cmd: pytest evals/test_tutor_turns.py) — HALTED on FAIL at call #003
Probes: 2 active | Calls: 3 (2 pass, 1 fail, 0 error)

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

- **Keep `probe.type.decision` checks short and atomic:** A single-pass decision model evaluates each check in parallel without multi-step reasoning. Prefer several short single-fact statements (`max_checks = 12` default) naming `output` or specific watched variables directly over one long compound sentence joining multiple `and`/`without`/`unless` clauses.
- **Exclude noisy service handles with `ignore=`:** On `@probe(..., ignore=("data_dir", "llm_client"))`, ignored parameters are skipped completely before snapshot serialization, excluded from Gate 3 fixtures, and checked via AST/runtime.
- **Use `kind=probe.type.text` for comparison or reasoning:** When a check requires comparing `output` against multi-turn `inputs` or retrieved context first, use `kind=probe.type.text` so `[models.text]` extracts factual evidence before `[models.decision]` scores the rule.
- **Add regression fixtures in `.probes/fixtures/<probe_id>.json`:** Place optional JSON files with either a single `"pass"` / `"fail"` snapshot object or lists of historical regression snapshots (`"pass": [{...}, ...]`, `"fail": [{...}, ...]`). `probe check` prints Gate 3 pass/fail calibration margins (`pass min p`, `fail max p`) and backtests every listed case whenever checks or fixtures change. Run `probe fixture <probe_id>` to promote recorded pass/fail snapshots out of `.probes/runs/*.jsonl` (or `probe fixture <probe_id> --init` to scaffold missing entries from the probe's AST signature/watched keys).
- **Monorepo `.env` & subproject root (`PROBE_PROJECT_ROOT` / `--root`):** `.env` lookups walk from `project_root` up to the enclosing `.git` root (nearest `.env` takes precedence), keeping API keys at the repo root while storing `.probes/` and `pyproject.toml` config in each subproject. Set `PROBE_PROJECT_ROOT=services/portfolio` (in env or root `.env`) or pass `--root <dir>` on CLI commands when running from a monorepo root.

## CLI

| Command | Purpose |
|---|---|
| `probe check [paths...] [--root DIR]` | Audit checks (Gate 1–2), run pass/fail calibration (Gate 3), show calibration margins, and update `.probes/lock.json`. |
| `probe list [paths...] [--root DIR]` | List discovered `@probe` and `probe.here` sites with lock status. |
| `probe fixture <id> [--run ID] [--init] [--root DIR]` | Promote recorded pass/fail run snapshots into `.probes/fixtures/<id>.json` or scaffold from AST. |
| `probe report [--run ID] [--all] [--check] [--root DIR]` | Show call-ordered summary of failures/errors (or all events with `--all`); exit `1` on failures with `--check`. |
| `probe log <probe_id>... [--payload] [--root DIR]` | Interleave chronological events across one or more probe IDs or globs. |
| `probe runs [--root DIR]` | List recorded runs and pass/fail/error counts. |
| `probe clean [--keep N] [--root DIR]` | Prune older run logs from `.probes/runs` (default keeps last 20). |

## Configuration (`~/.config/probe/config.toml`)

Project overrides can also be placed under `[tool.probe]` in `pyproject.toml`. (If `~/.config/probe/config.toml` already configures remote endpoints, overriding them to different remote hosts from `pyproject.toml` requires `PROBE_TRUST_PROJECT_ENDPOINTS=1`.)

```toml
domain_context = ""              # Optional domain conventions (passed to Gate 2/3 & runtime checks)

[models.decision]
provider = "typesafe_jev"
url = "https://api.typesafe.ai/v1/systemone"
model = "jev-latest"
api_key_env = "TYPESAFE_API_KEY"

# Alternative: local ONNX NLI decision runtime (install "probe[onnx] @ ..."):
# [models.decision]
# provider = "onnx_nli"
# onnx_path = "~/.cache/probe/deberta-v3-mnli/model.onnx"
# tokenizer_path = "~/.cache/probe/deberta-v3-mnli/tokenizer.json"

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
max_workers = 4
```

## License

Functional Source License, Version 1.1, with an Apache 2.0 future license (`FSL-1.1-Apache-2.0`). See [`LICENSE`](./LICENSE).
