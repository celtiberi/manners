# `probe` — Specification

> **Status:** Draft v0.6
> **Scope:** Python instrumentation, pre-run check, inline model-backed runtime checks, and call-ordered reports.

---

## 1. Overview

`probe` puts AI into the debugging loop on running Python code. You attach plain-English checks to functions (`@probe(...)`) or mid-function lines (`probe.here(...)`).

- When `PROBES` is unset, empty, or `0` (the default), `@probe` passes calls straight through to the original function and `probe.here` does nothing.
- When `PROBES=1` (via shell env or `.env`), `probe` captures runtime inputs, outputs, and watched variables, evaluates checks inline before continuing, writes results in call-entry order, and halts execution on the active frame if a check fails or a probe fires in a tight loop.
- Two model roles are configured in `~/.config/probe/config.toml`:
  - **`decision` (Jev-type model):** Takes a snapshot and discrete checks and returns calibrated probabilities `p ∈ [0.0, 1.0]` per check, mapping cleanly to `pass` or `fail`.
  - **`text` (chat/reasoning LLM):** Handles analysis on `probe.type.text` probes (before passing its findings to `[models.decision]` for the final `pass` / `fail` verdict), explains failures when configured, and runs `probe check` pre-run validation.

Switching between cloud endpoints (e.g., TypeSafe Jev + DeepSeek) and local models (e.g., a local decision service + local Qwen via Ollama) is just a config change.

---

## 2. Python API

### 2.1 Probe Types (`probe.type`)

```python
probe.type.decision   # Default: checks evaluated directly by [models.decision] -> pass | fail
probe.type.text       # Open-ended: analyzed first by [models.text], then graded by [models.decision] -> pass | fail
```

Both `@probe` and `probe.here` take `checks: str | Sequence[str]` as their first argument. A single string normalizes to `[checks]`; passing a list keeps each check discrete.

### 2.2 `@probe` Decorator

Supports both sync `def` and `async def` functions and preserves `functools.wraps` metadata (`__wrapped__`, `__signature__`, `__annotations__`).

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
    kind=probe.type.decision,
)
def generate_opening_turn(lesson: Lesson, history: list[Turn]) -> str:
    ...
```

**Signature:**

```python
def probe(
    checks: str | Sequence[str],
    *,
    id: str | None = None,
    kind: ProbeType = probe.type.decision,
    ignore: str | Sequence[str] = (),
) -> Callable[[F], F]: ...
```

- `checks`: One expectation string or a list of discrete expectations checked against captured arguments and return value.
- `id`: Probe identifier (e.g., `"tutor.opening_turn"`). Defaults to `<module>.<qualname>`.
- `kind`: `probe.type.decision` *(default)* or `probe.type.text`.
- `ignore`: Optional parameter name or sequence of parameter names omitted before runtime input serialization, `lock.json` `param_names`, and Gate 3 fixtures.

Waitpoints and decorators can also be suppressed or explicitly opted back in using stackable context managers (`from probe import disabled, enabled, flush`):

```python
with probe.disabled():       # Suppress probes across worker threads by default
    with probe.enabled():    # Re-enable inside nested blocks/workers
        ...
probe.flush()                # Drain background soak/thread evaluations and check halts
```
*(Overrides `PROBES=1` / unset `PROBES` across worker threads via `scope="process"` by default; pass `scope="context"` to scope to the active `ContextVar` without mutating environment variables.)*

A built-in `pytest11` plugin (`probe.pytest_plugin`) adds `--no-probes`, `--probes`, `@pytest.mark.probes` (exempts tests from `--no-probes` so env/`.env` `PROBES` controls execution, or pass `force=True` / `--probes` to force-enable), and `@pytest.mark.no_probes`. It wraps `pytest_runtest_setup`, `pytest_runtest_call`, and `pytest_runtest_teardown`, handles `ProbeHaltError` inside `pytest_fixture_setup` cleanly so shared fixture finalizers still run, and calls `probe.flush()` at the end of each phase.

### 2.3 `probe.here` Watchpoint

Mid-function check for intermediate values such as LLM drafts, prompts, or retrieved context.

```python
probe.here(
    [
        "Draft answer stays faithful to the meaning and conditions of retrieved_chunks",
        "Draft answer directly addresses the user's question",
        "Draft does NOT answer confidently when retrieved_chunks do not cover user_question",
    ],
    id="rag.draft_answer",
    kind=probe.type.text,
    user_question=question,
    retrieved_chunks=chunks,
    draft=draft_text,
)
```

**Signature:**

```python
def here(
    checks: str | Sequence[str],
    *,
    id: str | None = None,
    kind: ProbeType = probe.type.decision,
    **watched_vars: Any,
) -> None: ...
```

*(First positional argument is `checks: str | Sequence[str]`.)* Auto-derived `id` when omitted: `<module>.<qualname>:L<line>`.

---

## 3. Runtime Behavior

### 3.1 Call-Entry Ordering & Snapshot Capture

When probes are nested (e.g., an outer `@probe` calls an inner `@probe` or `probe.here`), inner checks finish before the outer function returns. `probe` sequences events by call entry:

1. At entry into `@probe` or `probe.here(...)`, the caller claims an atomic `call_seq` (`1, 2, 3, ...`), relative `elapsed_ms`, UTC `started_at`, and `file:line`.
2. Arguments (or `watched_vars`) are bounded-serialized on entry before function body execution can mutate them (caps: max depth `4`, max items `50`, max string `4 KB`).
3. On `@probe` exit (and immediately on `probe.here`), the full snapshot is redaction-scrubbed and evaluated inline before execution continues.

### 3.2 Secret Redaction (Runtime Secret Collection + Key Masking)

Before a snapshot is written to `.probes/runs/<run_id>.jsonl` or sent to any model:

1. **Startup secret collection (exact value scrubbing):** When `PROBES=1`, `probe` collects known secret values once at startup (`len >= 8`, skipping plain booleans, integers, ports, and local file paths) from:
   - Secret-named keys (`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASS*`, `*AUTH*`, `*DSN*`, `*DATABASE_URL*`, `*PRIVATE*`, `*CRED*`, `*BEARER*`, `*SIGNING*`, `*ENCRYPT*`) and connection/webhook URLs in `.env*` files across project root/working directory plus `[redact].env_files`.
   - Matching secret-named keys and userinfo passwords parsed out of URLs/DSNs in `os.environ`.

   Collected values (sorted longest-first) are replaced with `"[REDACTED]"` anywhere they appear inside serialized snapshots.
2. **Request-field masking:** Values under dynamic user-credential argument or dict keys (`password`, `passwd`, `passphrase`, `authorization`, `cookie`, plus `[redact].extra_keys`) are masked directly as `"[REDACTED]"` to cover per-request secrets that do not live in `.env` or `os.environ`.

### 3.3 Rate Guard (`PROBE_MISUSE`)

To catch probes accidentally placed inside row/item loops without cutting off long-lived local dev servers, `[rate_limit]` applies a sliding-window check per `probe_id`:

- Default: max `20` calls per `10.0` seconds sliding window, with `max_calls_per_run = -1` (unlimited; set a non-negative integer in config to enforce a per-run ceiling).
- If breached: trip that probe's circuit breaker before making another model call, record `verdict = "error"` (`error_code = "PROBE_MISUSE"`) with caller stack trace, and halt execution (§3.5).

### 3.4 Verdict Pipeline (`pass` / `fail`)

Every probe ends at `[models.decision]` and produces either `pass` or `fail` (plus `error` only for `PROBE_MISUSE` or backend errors):

1. **`probe.type.decision` (1-step):**
   ```text
   Redacted Snapshot + Checks ──▶ [models.decision] ──▶ p_i per check ──▶ pass | fail
   ```
   - A check passes when `p_i > pass_threshold` (default `0.50`). The event passes only if every check passes.
   - If any check fails and `[text].explain_decision_failures = true`, `[models.text]` writes an explanation using the configured `[text]` detail/token settings.

2. **`probe.type.text` (2-step):**
   ```text
   Redacted Snapshot + Checks ──▶ [models.text] (analysis) ──▶ [models.decision] ──▶ pass | fail
   ```
   - `[models.text]` writes free-form analysis of the snapshot against `checks` using the `[text]` config settings (`detail`, `max_tokens`, optional `system_prompt`).
   - `[models.decision]` reads `(checks, analysis, snapshot)` and returns `p_i` per check -> `pass` or `fail`.

| Verdict | Stored In Report | Runtime Effect |
|---|---|---|
| `pass` | Counted in summary; shown in `--all` and `probe log` | Continues normally |
| `fail` | Shown in `probe report` with per-check `p` and text analysis | **Halts execution (§3.5)** unless `PROBE_NO_HALT=1` / `halt_on_fail = false` (soak mode) |
| `error` | Shown in `probe report` | `PROBE_MISUSE`, `PROBE_TRUNCATED`, and (with default `strict_api = true`) `BACKEND_ERROR` **halt execution**; with `PROBE_NO_HALT=1` / `halt_on_fail = false` (or `strict_api = false` for non-truncation `BACKEND_ERROR`), prints to `stderr`, records `error` (`PROBE_MISUSE` still skips repeat calls for that probe), and marks the run `failed` |

### 3.5 Execution Halt (and `PROBE_NO_HALT=1` Soak Mode)

By default, when a probe produces `fail`, `PROBE_MISUSE`, `PROBE_TRUNCATED`, or (with `strict_api = true`) `BACKEND_ERROR`:

1. Flush the full event record (`call_seq`, `probe_id`, `probe_type`, `file_line`, redacted payload, per-check scores, `text_analysis`, `stack_excerpt`) to `.probes/runs/<run_id>.jsonl`, mark the run `halted`, and print `[PROBE HALT]` to `stderr`.
2. Set a global `tripped` flag so any further probe call raises `ProbeHaltError` *(a `BaseException` subclass, like `KeyboardInterrupt` and `bdb.BdbQuit`, so application `except Exception:` retry/fallback blocks cannot swallow probe halts while `finally:` blocks still run)*.
3. Stop directly on the active caller stack frame:
   - In an interactive TTY (`sys.stdin.isatty()` and `sys.stdout.isatty()`, unless `PROBE_NO_PDB=1`), invoke `pdb.Pdb().set_trace(caller_frame)` so locals can be inspected right before the function returns.
   - Otherwise raise `ProbeHaltError`.

When `PROBE_NO_HALT=1` or `[runtime].halt_on_fail = false` is set (soak / multi-turn eval mode), `call_seq`, timestamps, and redacted snapshots are captured synchronously while model grading dispatches onto a bounded background worker pool (`[runtime].max_workers`, default `4`, drained on `probe.flush()` and process `atexit`), `fail` and error events print `[PROBE FAIL]` / `[PROBE ERROR]` to `stderr`, `PROBE_MISUSE` trips the per-probe circuit breaker without halting the rest of the run, the run finishes with status `"failed"`, and `probe report --check` exits `1`.

When a probe running inside a fire-and-forget background `threading.Thread` fails in halting mode, `probe` captures uncaught thread `ProbeHaltError` exits via `threading.excepthook` and re-raises on the caller thread during `probe.flush()` (and automatically across `pytest` `setup`/`call`/`teardown` phases). Note that `probe.flush()` waits on evaluations that have already entered `@probe` or `probe.here`; if a test spawns a daemon thread doing slow work before calling `probe`, `join()` the thread before returning.

---

## 4. Pre-Run Check (`probe check`)

Probes must pass `probe check` before running (and whenever checks, `kind`, or target function source change). Approved hashes are saved in `.probes/lock.json`.

1. **Static AST scan:** Extracts `@probe(...)` and `probe.here(...)` calls, signatures, and enclosing context (`probe.here` includes the enclosing function's docstring and up to 40 pre-call lines in `source_excerpt`) without importing project modules.
2. **Gate 1 — Budgets & structure:** Checks for duplicate `probe_id`s, `len(checks) <= max_checks` (default `12`), and `len(check) <= max_check_chars` (default `200`).
3. **Gate 2 — Assert vs. prose audit (`[models.text]` + `[models.decision]`):** `[models.text]` describes factually what each check inspects at runtime, and `[models.decision]` scores short positive Gate 2 defect questions (`expected: false`; every defect question must score `< 0.50` to pass)—rejecting checks that belong as regular Python `assert`s or regexes (math, null checks, regex matches, type/length checks), vague unfalsifiable filler, or checks requiring uncaptured external state.
4. **Gate 3 — Dry-run calibration:** Skipped when `(probe_id, site, kind, watched_keys, param_names)` and combined `(checks, source, threshold, domain_context, manual fixture)` hashes are unchanged in `.probes/lock.json` (unless `--force`). For `@probe`, `source_hash` hashes `(signature, docstring[, ignore=...])` so editing internal statements/logging does not invalidate `lock.json`; for `probe.here`, `source_hash` hashes `(enclosing_signature, watched_keys)`. Uncached probes run Gates 2–3 in bounded parallel (`[check].max_workers`, default `4`) while returning deterministic order. Decision snapshots separate `Observed Output (under test)` / `Watched Variables (under test)` ahead of `Call Inputs (context only)`. Uses fixtures from `.probes/fixtures/<probe_id>.json` if present (either a single `"pass"`/`"fail"` snapshot or lists of historical snapshots `"pass": [...]`, `"fail": [...]`, optionally promoted via `probe fixture <probe_id>`), or asks `[models.text]` for one passing and one failing sample matching the signature (retrying synthetic synthesis up to 1x on transient shape/calibration failures), then verifies every pass sample scores `pass` on all checks, every fail sample fails at least one check, and reports the resulting `(pass min p, fail max p)` margins in `probe check` output and `.probes/lock.json`.

---

## 5. CLI

Per-run JSONL files (`.probes/runs/<run_id>.jsonl`) are append-only plain text; CLI commands sort and summarize events by `call_seq ASC`, and accept `--root <dir>` (either globally or per subcommand) to override project root discovery.

| Command | Purpose |
|---|---|
| `probe check [paths...]` | Run Gates 1–3 on new or changed probes, surface calibration margins, and update `.probes/lock.json` (`--force`, `--json`, `--root`). |
| `probe list [paths...]` | List probes (`id`, `file:line`, `kind`, lock status, `checks`, `--root`). |
| `probe fixture <id>` | Promote recorded pass/fail snapshots from `.probes/runs/*.jsonl` into `.probes/fixtures/<id>.json` (`--run ID`, `--init`, `--json`, `--root`). |
| `probe report` | Print call-ordered summary of the latest run (`--run <id>`, `--all`, `--format text\|md\|json`, `--check`, `--root`). |
| `probe log <probe_id>...` | Interleave detailed chronological entries (`call_seq ASC`) for one or more probe IDs / glob patterns (`--root`). |
| `probe runs` | List recent runs with timestamps, command, verdict counts, and status (`--root`). |
| `probe clean [--keep N]` | Prune older run logs from `.probes/runs` (default keeps last `20`, `--root`). |

### Example `probe report --all` Output

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

---

## 6. Configuration & Storage

### 6.1 Configuration (`~/.config/probe/config.toml` + `[tool.probe]`)

Shared user config lives in `~/.config/probe/config.toml`; projects can override individual keys under `[tool.probe]` in `pyproject.toml`. When a user config already defines remote endpoints, external `pyproject.toml` overrides to different remote origins require `PROBE_TRUST_PROJECT_ENDPOINTS=1` (prevents untrusted repos from redirecting API keys).

**Example — Cloud Profile (`typesafe_jev` + `deepseek`):**

```toml
domain_context = ""              # Optional short domain conventions shared across Gate 2/3 and runtime

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
```

**Example — Local Profile (local decision service + local `qwen3.5:9b` via Ollama):**

```toml
[models.decision]
provider = "http_decision"
url = "http://127.0.0.1:8080/v1/evaluate"
model = "local-decision"

[models.text]
provider = "openai_compat"
url = "http://127.0.0.1:11434/v1"
model = "qwen3.5:9b"
```

*(You can also configure `provider = "onnx_nli"` with local ONNX weights — default `max_length = 512`, `onnx_output = "logits"` or `"probabilities"`, optional explicit `entailment_index` / `contradiction_index`, truncating premise `only_first` — or mix cloud `[models.decision]` `typesafe_jev` with local Ollama `[models.text]` `qwen3.5:9b`.)*

**Shared Settings:**

```toml
domain_context = ""              # Optional domain conventions (keeps check strings concise)

[decision]
pass_threshold = 0.50            # Check passes iff p > pass_threshold

[text]
detail = "standard"              # "concise" | "standard" | "detailed"
max_tokens = 1024
explain_decision_failures = true
system_prompt = ""               # Optional custom instructions for [models.text]

[rate_limit]
max_calls = 20
window_seconds = 10.0
max_calls_per_run = -1           # -1 = unlimited; set >= 0 for a per-run hard cap

[check]
max_checks = 12
max_check_chars = 200
require_check = true
max_workers = 4

[redact]
env_files = [".env"]             # Dotenv files scanned for secret values (in addition to cwd .env*)
extra_keys = []                  # Additional argument/dict key names to mask

[runtime]
runs_dir = ".probes/runs"
lock_path = ".probes/lock.json"
strict_api = true                # Halt on BACKEND_ERROR; set false for stderr warning + non-halting log
halt_on_fail = true              # Set false (or PROBE_NO_HALT=1) for non-halting soak/eval runs
max_workers = 4                  # Bounded background grading pool size in non-halting soak mode
```

**Environment flags (shell env or `.env`, searching nearest-first from `project_root` up to the enclosing `.git` root):**
- `PROBES`: `0` *(default, disabled)* | `1` *(enabled; shell env overrides `.env`)*
- `PROBE_PROJECT_ROOT`: Optional project root directory override (useful in monorepos when running from the repository root so `pyproject.toml`, `.probes/lock.json`, and `.probes/runs` stay inside a subpackage)
- `PROBE_CONFIG`: Optional path override for `config.toml`
- `PROBE_SKIP_LOCK`: `1` to bypass `.probes/lock.json` during rapid local edits
- `PROBE_NO_PDB`: `1` to always raise `ProbeHaltError` instead of opening `pdb` in interactive terminals
- `PROBE_NO_HALT`: `1` to record all fail/error events across a soak run without halting mid-run (pair with `probe report --check`)
- `PROBE_TRUST_PROJECT_ENDPOINTS`: `1` to allow project `pyproject.toml` to redirect remote model endpoint origins configured in user `config.toml`

### 6.2 Per-Run JSONL Log Format (`.probes/runs/<run_id>.jsonl`)

Each run writes one plain-text JSON Lines file named `<run_id>.jsonl` (`ULID`), sorted chronologically by filename and containing three record types (`"type"`):

- **`run_start`**: `id` (ULID), `started_at`, `cmd`, `pid`, `status` (`"running"`), `decision_model`, `text_model`
- **`event`**: `run_id`, `call_seq`, `elapsed_ms`, `started_at`, `probe_id`, `site` (`decorator` | `here`), `probe_type` (`decision` | `text`), `file_line`, `checks`, `payload` (redacted snapshot object), `check_scores` (array of `{check, probability, passed}`), `verdict` (`pass` | `fail` | `error`), `error_code` (`PROBE_MISUSE` | `PROBE_TRUNCATED` | `BACKEND_ERROR` | ...), `text_analysis`, `stack_excerpt`, `judged_at`, `latency_ms`
- **`run_finish`**: `id`, `status` (`completed` | `failed` | `halted`), `finished_at`, `halt_call_seq`

Readers sort events per run by `call_seq ASC` and inspect runs directly via `probe report` / `probe log` (or `jq` / `tail -f`).

### 6.3 Package Layout (`probe/`)

```text
probe/
├── LICENSE
├── SPEC.md
├── pyproject.toml
├── src/
│   └── probe/
│       ├── __init__.py          # Public API: @probe, probe.here, probe.type.{decision,text}
│       ├── py.typed             # PEP 561 typing marker
│       ├── config.py            # ~/.config/probe/config.toml + pyproject.toml + env/.env resolution
│       ├── redact.py            # .env / os.environ secret collection + request-key masking
│       ├── prompts.json         # Internal prompts with role/stage/purpose metadata
│       ├── prompts.py           # Prompt catalog loader + model message builders
│       ├── backends.py          # Decision providers + Text LLM client
│       ├── runtime.py           # call_seq tracker, rate guard, inline grading, halt switch
│       ├── store.py             # Per-run JSONL store
│       ├── check.py             # AST discovery + Gates 1-3 + .probes/lock.json
│       ├── pytest_plugin.py     # Built-in pytest11 plugin (--probes/--no-probes, markers, flush)
│       └── cli.py               # Typer CLI: check, list, fixture, report, log <ids...>, runs, clean
└── tests/
```

---

## 7. Scope Limits (v0)

- Intended for pre-deploy runs, eval sweeps, local/staging debugging, and canary verification—not high-QPS production traffic.
- Does not replace deterministic `assert`s or Pydantic validators (Gate 2 rejects checks that belong in deterministic code).
- Never mutates return values or control flow other than halting on `fail` or `PROBE_MISUSE`.
