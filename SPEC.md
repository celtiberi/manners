# `probe` — Specification

> **Status:** Draft v0.6
> **Scope:** Python instrumentation, pre-run check, inline model-backed runtime checks, and call-ordered reports.

---

## 1. Overview

`probe` puts AI into the debugging loop on running Python code. You attach plain-English checks to functions (`@probe(...)`) or mid-function lines (`probe.here(...)`).

- When `PROBES` is unset or `0` (the default), `@probe` returns the original function unwrapped and `probe.here` does nothing.
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
) -> Callable[[F], F]: ...
```

- `checks`: One expectation string or a list of discrete expectations checked against arguments and return value (or exception).
- `id`: Probe identifier (e.g., `"tutor.opening_turn"`). Defaults to `<module>.<qualname>`.
- `kind`: `probe.type.decision` *(default)* or `probe.type.text`.

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
2. Arguments (or `watched_vars`) are bounded-serialized on entry before function body execution can mutate them (caps: max depth `4`, max items `25`, max string `4 KB`).
3. On `@probe` exit (and immediately on `probe.here`), the full snapshot is redaction-scrubbed and evaluated inline before execution continues.

### 3.2 Secret Redaction (Runtime Secret Collection + Key Masking)

Before a snapshot is written to `.probes/probes.db` or sent to any model:

1. **Startup secret collection (exact value scrubbing):** When `PROBES=1`, `probe` collects known secret values once at startup (`len >= 8`, skipping plain booleans, integers, ports, and local file paths) from:
   - `.env*` files in the working directory plus any paths listed in `[redact].env_files`.
   - `os.environ` keys matching secret-like names (`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASS*`, `*AUTH*`, `*DSN*`, `*DATABASE_URL*`, `*PRIVATE*`, `*CREDENTIAL*`), plus userinfo passwords parsed out of connection URLs/DSNs.

   Collected values (sorted longest-first) are replaced with `"[REDACTED]"` anywhere they appear inside serialized snapshots.
2. **Request-field masking:** Values under dynamic user-credential argument or dict keys (`password`, `passwd`, `passphrase`, `authorization`, `cookie`, plus `[redact].extra_keys`) are masked directly as `"[REDACTED]"` to cover per-request secrets that do not live in `.env` or `os.environ`.

### 3.3 Rate Guard (`PROBE_MISUSE`)

To catch probes accidentally placed inside row/item loops without cutting off long-lived local dev servers, `[rate_limit]` applies a sliding-window check per `probe_id`:

- Default: max `5` calls per `10.0` seconds sliding window, with `max_calls_per_run = -1` (unlimited; set a non-negative integer in config to enforce a per-run ceiling).
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
| `fail` | Shown in `probe report` with per-check `p` and text analysis | **Halts execution (§3.5)** |
| `error` | Shown in `probe report` | `PROBE_MISUSE` **halts execution**; backend HTTP errors log `error` unless `strict_api = true` |

### 3.5 Execution Halt

When a probe produces `fail` or `PROBE_MISUSE`:

1. Flush the full event record (`call_seq`, `probe_id`, `probe_type`, `file_line`, redacted payload, per-check scores, `text_analysis`, `stack_excerpt`) to `.probes/probes.db`, mark the run `halted`, and print `[PROBE HALT]` to `stderr`.
2. Set a global `tripped` flag so any further probe call raises `ProbeHaltError`.
3. Stop directly on the active caller stack frame:
   - In an interactive TTY (`sys.stdin.isatty()` and `sys.stdout.isatty()`, unless `PROBE_NO_PDB=1`), invoke `pdb.Pdb().set_trace(caller_frame)` so locals can be inspected right before the function returns.
   - Otherwise raise `ProbeHaltError`.

---

## 4. Pre-Run Check (`probe check`)

Probes must pass `probe check` before running (and whenever checks, `kind`, or target function source change). Approved hashes are saved in `.probes/lock.json`.

1. **Static AST scan:** Extracts `@probe(...)` and `probe.here(...)` calls, signatures, and bodies without importing project modules.
2. **Gate 1 — Budgets & structure:** Checks for duplicate `probe_id`s, `len(checks) <= max_checks` (default `5`), and `len(check) <= max_check_chars` (default `200`).
3. **Gate 2 — Assert vs. prose audit (`[models.text]` + `[models.decision]`):** `[models.text]` describes factually what each check inspects at runtime, and `[models.decision]` scores the Gate 2 `noul` propositions—rejecting checks that belong as regular Python `assert`s or regexes (math, null checks, regex matches, type/length checks), vague unfalsifiable filler, or checks requiring uncaptured external state.
4. **Gate 3 — Dry-run calibration:** Skipped if `(probe_id, check_hash)` is unchanged in `.probes/lock.json` (unless `--force`). Uses fixtures from `.probes/fixtures/<probe_id>.json` if present, or asks `[models.text]` for one passing and one failing sample matching the signature, then verifies the configured evaluator scores the valid sample `pass` and the broken sample `fail`.

---

## 5. CLI

SQLite is internal storage. All inspection goes through the `probe` CLI, with events sorted by `call_seq ASC`.

| Command | Purpose |
|---|---|
| `probe check [paths...]` | Run Gates 1–3 on new or changed probes and update `.probes/lock.json` (`--force`, `--json`). |
| `probe list [paths...]` | List probes (`id`, `file:line`, `kind`, lock status, `checks`). |
| `probe report` | Print call-ordered summary of the latest run (`--run <id>`, `--all`, `--format text\|md\|json`). |
| `probe log <probe_id>...` | Interleave detailed chronological entries (`call_seq ASC`) for one or more probe IDs / glob patterns (e.g., `probe log tutor.opening_turn rag.*`). |
| `probe runs` | List recent runs with timestamps, command, verdict counts, and status. |
| `probe clean [--keep N]` | Prune old runs from `.probes/probes.db` (default keeps last `20`). |

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

Shared user config lives in `~/.config/probe/config.toml`; projects can override individual keys under `[tool.probe]` in `pyproject.toml`.

**Example — Cloud Profile (`typesafe_jev` + `deepseek`):**

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

*(You can also configure `provider = "onnx_nli"` with local ONNX weights or mix cloud `[models.decision]` `typesafe_jev` with local Ollama `[models.text]` `qwen3.5:9b`.)*

**Shared Settings:**

```toml
[decision]
pass_threshold = 0.50            # Check passes iff p > pass_threshold

[text]
detail = "standard"              # "concise" | "standard" | "detailed"
max_tokens = 400
explain_decision_failures = true
system_prompt = ""               # Optional custom instructions for [models.text]

[rate_limit]
max_calls = 5
window_seconds = 10.0
max_calls_per_run = -1           # -1 = unlimited; set >= 0 for a per-run hard cap

[check]
max_checks = 5
max_check_chars = 200
require_check = true

[redact]
env_files = [".env"]             # Dotenv files scanned for secret values (in addition to cwd .env*)
extra_keys = []                  # Additional argument/dict key names to mask

[runtime]
db_path = ".probes/probes.db"
lock_path = ".probes/lock.json"
```

**Environment flags (shell env or `.env`):**
- `PROBES`: `0` *(default, disabled)* | `1` *(enabled; shell env overrides `.env`)*
- `PROBE_CONFIG`: Optional path override for `config.toml`
- `PROBE_SKIP_LOCK`: `1` to bypass `.probes/lock.json` during rapid local edits
- `PROBE_NO_PDB`: `1` to always raise `ProbeHaltError` instead of opening `pdb` in interactive terminals

### 6.2 Internal SQLite Schema (`.probes/probes.db`, WAL mode)

- **`runs`**: `id` (ULID), `started_at`, `finished_at`, `cmd`, `pid`, `status` (`running` | `completed` | `halted`), `halt_call_seq`, `decision_model`, `text_model`
- **`events`**: `run_id`, `call_seq` (`PRIMARY KEY (run_id, call_seq)`), `elapsed_ms`, `started_at`, `probe_id`, `site` (`decorator` | `here`), `probe_type` (`decision` | `text`), `file_line`, `checks_json`, `payload_json` (redacted snapshot), `check_scores_json` (array of `{check, probability, passed}`), `verdict` (`pass` | `fail` | `error`), `error_code` (`PROBE_MISUSE` | ...), `text_analysis`, `stack_excerpt`, `judged_at`, `latency_ms`

Indexes on `(run_id, call_seq)` and `(run_id, probe_id, call_seq)` serve `probe report` and multi-ID `probe log`.

### 6.3 Package Layout (`probe/`)

```text
probe/
├── LICENSE
├── SPEC.md
├── pyproject.toml
├── src/
│   └── probe/
│       ├── __init__.py          # Public API: @probe, probe.here, probe.type.{decision,text}
│       ├── config.py            # ~/.config/probe/config.toml + pyproject.toml + env/.env resolution
│       ├── redact.py            # .env / os.environ secret collection + request-key masking
│       ├── prompts.json         # Internal prompts with role/stage/purpose metadata
│       ├── prompts.py           # Prompt catalog loader + model message builders
│       ├── backends.py          # Decision providers + Text LLM client
│       ├── runtime.py           # call_seq tracker, rate guard, inline grading, halt switch
│       ├── store.py             # SQLite WAL store
│       ├── check.py             # AST discovery + Gates 1-3 + .probes/lock.json
│       └── cli.py               # Typer CLI: check, list, report, log <ids...>, runs, clean
└── tests/
```

---

## 7. Scope Limits (v0)

- Intended for pre-deploy runs, eval sweeps, local/staging debugging, and canary verification—not high-QPS production traffic.
- Does not replace deterministic `assert`s or Pydantic validators (Gate 2 rejects checks that belong in deterministic code).
- Never mutates return values or control flow other than halting on `fail` or `PROBE_MISUSE`.
