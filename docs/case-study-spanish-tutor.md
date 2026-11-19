# Case Study: Using `manners` in an AI Spanish Tutor

This document summarizes an integration of `manners` (`v0.1.0` through `v0.4.0`) across a 222-file AI Spanish tutor codebase (`2,493` unit tests) with a multi-stage LLM turn pipeline and live AI-student soak simulations.

`manners` targets the gap normal tests miss: calls that return valid strings or JSON, `200 OK`, and green `pytest` checks, yet are behaviorally wrong in context.

---

## Integration Layout

In this tutor, a single student turn is assembled across multiple stages rather than one prompt:

- parallel acknowledgment and lesson generation (`tutor/pedagogy/teach_calls.py`),
- partner dialogue generation and stitching (`tutor/pedagogy/mix_play.py`),
- scripted diagnostic placement (`tutor/spans/placement_script.py`),
- question detours and final reply assembly (`tutor/turn_pipeline/finish.py`),
- B1/B2 conversational turns (`tutor/turn_pipeline/model_call.py`),
- asynchronous background working-memory summarization (`tutor/session/history_summary.py`), and
- end-of-cycle longitudinal skill grading (`tutor/turn_pipeline/cycle_grade.py`).

**9 checks** were placed across those boundaries, split between `manners`'s two modes:

1. **`@manners` (`manners.type.decision`) on the every-turn exit gate (`audit_settled_tutor_reply` in `tutor/turn_pipeline/finish.py`):** 3 short, single-fact checks verifying that every learner-visible reply uses a natural tutor voice, leaks no internal prompt/planning headers, and leaves the learner with a clear prompt, question, or listening task rather than a dead end.
2. **`manners.here(..., kind=manners.type.text)` on 8 internal relational seams:** Used wherever pass/fail depends on comparing runtime values against one another (for example, comparing `clamped_delta` skill promotions against the Spanish the learner actually produced in `cycle_turns`, or comparing a compressed `summary` against the raw `segment`).

In production (`MANNERS` unset), all 9 sites are zero-overhead no-ops. In mocked unit tests, an autouse `with manners.disabled():` fixture in `tests/conftest.py` keeps canned stubs (`"Nice."`, fake LLM payloads) from firing model calls, while `tests/test_manners.py` and live DeepSeek AI-student runs execute with `MANNERS=1`.

---

## Bugs Found (and Why Normal Tests Missed Them)

### 1. Silent Architectural Violation in a Background Daemon Thread (`tutor.history_summary`)

- **The bug:** `tutor/session/history_summary.py` runs a fast background model (`gemini-3.5-flash-lite`) on a daemon thread *after* the HTTP reply ships to compress older turns into working memory. A core rule of the application is that the character sheet is the sole source of learner-ability state—rolling summaries must record neutral turn facts and verbatim learner errors, never proficiency judgments. On an advanced test persona (`sofia_fluent`), the summarizer wrote: *"The learner demonstrated advanced conversational fluency... native-level elaboration"* and *"all learner utterances were grammatically correct and idiomatic."* On a beginner persona (`casey_steady`), it added *"successfully produced introductory Spanish."*
- **Why normal tests missed it:** Summarization runs asynchronously on a daemon thread and never appears in the chat UI. Nothing crashed, no schema check failed, and an external chat-transcript evaluation would not see the internal summary—it only polluted subsequent system prompts.
- **How `manners` caught it:** `tutor.history_summary` fired inside `build_summary` on the background thread, scored check 0 at `p=0.04`, halted the run, and printed the offending summary alongside the evidence breakdown. Tightening `_PROMPT` in `tutor/session/history_summary.py` resolved both personas (`0.98`–`0.99` on re-run), and the failing case was added to `.manners/fixtures/tutor.history_summary.json`.

### 2. Modality Hallucination & Dropped Check-Ins in Parallel Calls (`tutor.ack_turn`)

- **The bug:** During a live `qa_detour` session, the AI student typed `"¿Qué tal?"` in text chat. The parallel acknowledgment call (`tutor/pedagogy/teach_calls.py`) praised how clear the learner's *spoken pronunciation* sounded and skipped answering `"¿Qué tal?"` before the stitched lesson continued.
- **Why normal tests missed it:** The output was valid Spanish UTF-8 with no leaked tags, valid `<listen>` markup, and clean envelope structure.
- **How `manners` caught it:** `tutor.ack_turn` flagged both the modality mismatch (praising spoken voice clarity on typed text) and the ignored conversational check-in on `call #015`. Fixing the modality label (`typed (text)` vs. `spoke (audio)`) in `tutor/pedagogy/teach_calls.py` and tightening `prompts/ack_turn.md` eliminated both issues on subsequent runs.

### 3. Envelope Stitching & Spoiled Audio Quizzes During `MANNERS=1` Soak Rollout

Running live AI-student simulations with `MANNERS=1` surfaced two stitching defects in `tutor/pedagogy/mix_play.py` and `tutor/pedagogy/listen_review.py`:

1. When a student made a Spanish error on a `mix` or `listen` turn, inlining `<recast>...</recast>` without clearing `turn_env.recast` caused downstream box-merging to drop the lesson body and prompt—leaving a bare dead-end correction (`"Y un café, por favor."`) with no next task (the exact condition watched by `tutor.settled_reply`).
2. On hidden audio comprehension checks (`<listen hide="1">¿Qué tal?</listen>`), `extract_opening_ack` let `"¡Hola, qué tal!"` leak into the visible greeting directly above the hidden audio tag, spoiling the listening check.

---

## What Worked Well in Practice

1. **In-process watchpoints (`manners.here`) vs. transcript-only evaluation:**
   External eval harnesses only inspect final assistant messages. When a multi-stage pipeline produces a defective turn, isolating the stage responsible requires digging through logs, and background states like working-memory summaries or cycle graders never appear in the final output at all. Placing `manners.here` inside `_unpack_ack_turn`, `build_summary`, and `_cycle_grade_call` checked behavior at the exact function seam with local variables attached.
2. **Pre-run validation via `manners check` (Gates 1–3):**
   Gates 1–2 reject checks that belong in regular Python `assert`s or regexes (`is not None`, JSON syntax, exact counts). Gate 3 calibrates passing and failing snapshots up front and locks hashes in `.manners/lock.json`. Vague checks or rules conflicting with app conventions surfaced during `manners check` in seconds instead of failing intermittently mid-soak.
3. **`domain_context` + multi-case `.manners/fixtures/`:**
   Tutor-specific conventions—such as `<listen hide="1">` markup, `Carlos:` partner dialogue prefixes, silent-*h* pronunciation notes, and band ordering (`unknown/blocked < emerging < fragile < known`)—would confuse a generic judge without context. Centralizing them once in `[tool.manners] domain_context` inside `pyproject.toml` kept individual check strings concise, while `v0.4.0`'s multi-case `"pass": [...]` / `"fail": [...]` fixture lists (`.manners/fixtures/tutor.teach_turn.json`, `.manners/fixtures/tutor.history_summary.json`) locked past failures into Gate 3 regression checks.
4. **`v0.3.0`–`v0.4.0` runtime ergonomics:**
   - Subclassing `MannersHaltError` from `BaseException` eliminated manual `except MannersHaltError: raise` pass-throughs around internal `try ... except Exception:` fallback wrappers.
   - Making `manners.disabled()` process-wide and thread-safe by default ensured daemon threads (`history-summary`) and `ThreadPoolExecutor` workers (`teach_calls`) honored test suppression without manual `contextvars` propagation.
   - Pairing `MANNERS_NO_HALT=1` with `manners report --check` supported both workflows cleanly: strict fail-fast halting during targeted debugging, and non-halting full-run event collection across multi-turn student soak runs.

---

## Observed Boundaries & Trade-Offs

- **High-arity multi-variable seams need manual fixtures:** On 2–4 variable checks, synthetic Gate 3 fixture generation calibrated cleanly out of the box. On `tutor.teach_turn` (7 interdependent watched variables including `also_introduce` and `warmup_recall_only`), synthetic generation struggled with valid state combinations until a 2-pass / 2-fail fixture file was added under `.manners/fixtures/`. Writing that file took ~2 minutes and made calibration deterministic.
- **Test/eval/soak tool, not a live production guardrail:** Each `manners.type.decision` check performs a model API round-trip, and `manners.type.text` adds an evidence-extraction call (~1.5–2.5s per check evaluation). Leaving `MANNERS` unset in production keeps normal requests at zero overhead, while enabling `MANNERS=1` covers smoke tests, CI checks, and multi-turn AI-student soak runs.
