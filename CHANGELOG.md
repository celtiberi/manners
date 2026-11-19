# Changelog

All notable changes to `probe` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.1] - 2026-11-18

### Changed
- Raised Gate 3 synthetic fixture generation default `max_tokens` floor from `1200` to `2400` and added automatic 1x retry with `max(4096, [text].max_tokens * 2)` and concise-output feedback when attempt 1 hits `finish_reason='length'` (`ModelTruncatedError`).
- Raised default `[runtime].max_items` snapshot serialization cap from `25` to `50` so wider structured return models serialize intact without `__truncated_items__`.
- Clarified Gate 1's non-callable `probe` module error hint to note that `from probe import probe` already exposes `probe.type`, `probe.here`, and `probe.disabled`.

## [0.6.0] - 2026-11-18

### Added
- Monorepo `.env` and project-root discovery: `.env` flag, API-key, and secret-redaction lookups now walk nearest-first from `project_root` up to the enclosing `.git` root without escaping above the repository boundary. Added `PROBE_PROJECT_ROOT` (env/`.env`) and `--root <dir>` across CLI commands (`check`, `list`, `fixture`, `report`, `log`, `runs`, `clean`) plus `[tool.probe]` loading from the resolved project root.
- `@probe(..., ignore=...)` support to exclude specified parameter names (such as `data_dir`, `llm_client`, or `self`) before input serialization, lockfile `param_names` checks, Gate 1–3 audits/fixtures, and `probe fixture --init` skeletons.
- Gate 3 calibration probability margins (`pass min p`, `fail max p`) surfaced in `probe check` pass/cache output and stored in `.probes/lock.json`.

## [0.5.1] - 2026-11-18

### Fixed
- Wrapped all three pytest item phases (`pytest_runtest_setup`, `pytest_runtest_call`, and `pytest_runtest_teardown`) in `probe.pytest_plugin` so `--no-probes`, `--probes`, and `@pytest.mark.{probes,no_probes}` protect fixture setup and teardown as well as test bodies.
- Migrated `pytest_plugin.py` hooks from legacy `@pytest.hookimpl(hookwrapper=True)` to modern `@pytest.hookimpl(wrapper=True)` and added `pytest_fixture_setup` handling for `ProbeHaltError`, eliminating `PluggyTeardownRaisedWarning` and subsequent-test `_finalizers` assertion crashes on shared failing fixtures.
- Changed `@pytest.mark.probes` to exempt marked tests from `--no-probes` (inheriting `PROBES` from `os.environ` / `.env` so keyless offline CI stays green), reserving force-enable for `--probes` or `@pytest.mark.probes(force=True)`, and honored explicit empty `os.environ` API-key overrides (`KEY=""`) without falling through to `.env`.
- Clarified natural-language precondition scoping in `check_gate2_analysis` / `check_gate2_nouls` and added a 1x Gate 2 retry (`temperature=0.2`) in `run_probe_check` to absorb transient text/decision disagreements.
- Added AST type-hint / local-value inference to `probe fixture --init` (`bool`, `int`, `float`, `list`, `dict`, and `-> None`) and documented daemon-thread pre-`probe.here` timing expectations.

## [0.5.0] - 2026-11-18

### Added
- `probe fixture <probe_id>` CLI command (`--run`, `--init`, `--json`) and `upsert_probe_fixture` helper to promote deduplicated `"pass"` and `"fail"` snapshots out of `.probes/runs/*.jsonl` into `.probes/fixtures/<probe_id>.json` (or scaffold starter snapshots from AST metadata).
- Built-in `pytest11` plugin (`probe.pytest_plugin`) adding `--no-probes`, `--probes`, `@pytest.mark.probes`, `@pytest.mark.no_probes`, clean background-thread `ProbeHaltError` capture without spurious `PytestUnhandledThreadExceptionWarning`, and automatic post-test `probe.flush()`.
- Stackable `probe.enabled(scope="process" | "context")` counterpart to `probe.disabled()`, plus `probe.flush(check=True)` public API (`from probe import disabled, enabled, flush`).
- `[runtime].max_workers` (default `4`) configuration option and non-blocking bounded background grading pool when `PROBE_NO_HALT=1` / `[runtime].halt_on_fail = false` soak mode is active.

### Changed
- Expanded AST `source_excerpt` for `probe.here` calls inside functions to include the enclosing function's docstring and up to 40 pre-call source lines (keeping `source_hash` invariant to internal edits).
- Unified `probe.disabled()` and `probe.enabled()` around stackable context and process override tables (`_OVERRIDE_CTX` and `_PROCESS_OVERRIDES`).

### Fixed
- Eliminated dropped `ProbeHaltError` failures from fire-and-forget daemon `threading.Thread`s by recording uncaught thread halts via `threading.excepthook` and surfacing them on `probe.flush()` / `pytest` teardown without double-raising on `ThreadPoolExecutor` futures.

## [0.4.0] - 2026-11-18

### Added
- Support for multi-case regression fixture lists in `.probes/fixtures/<probe_id>.json` (`"pass": [...]` and `"fail": [...]` alongside single snapshot objects): `probe check` validates and scores every passing and failing snapshot during Gate 3, folds all cases into the lock hash, and reports specific failing case indices.
- Declarative `"expected"` boolean polarity on `models.decision` entries in `prompts.json` (`get_noul_expected`), enabling positive defect questions where passing requires `probability < 0.50`.

### Changed
- Raised default `[check].max_checks` from `5` to `12` so `probe.type.decision` suites can decompose compound requirements into short, atomic single-fact checks.
- Rewrote `check_gate2_nouls` in `prompts.json` as short, positive, single-fact defect questions (`"expected": false`) and aligned `check_gate2_analysis` to describe observed defects factually without echoing unrelated buzzwords.
- Updated `format_typesafe_state` (`src/probe/prompts.py`) and `_format_onnx_premise` (`src/probe/backends.py`) to foreground and label `Observed Output (under test)` / `Watched Variables (under test)` ahead of `Call Inputs (context only)`.

## [0.3.0] - 2026-11-18

### Added
- Non-halting soak/eval audit mode via `PROBE_NO_HALT=1` or `[runtime].halt_on_fail = false`: records all `fail` and `error` events across multi-turn runs (printing `[PROBE FAIL]` / `[PROBE ERROR]` to `stderr`, tripping per-probe `PROBE_MISUSE` circuit breakers cleanly without spamming duplicate events, and setting run status to `"failed"`).
- `probe report --check` CLI flag exiting with code `1` when a run recorded any failure/error event (pair with `PROBE_NO_HALT=1`).
- Bounded parallel execution of uncached probes in `run_probe_check` via `[check].max_workers` (default `4`), preserving deterministic result ordering.
- `probe.disabled(scope="process" | "context")` option on `probe.disabled()`.

### Changed
- **Breaking:** Subclassed `ProbeHaltError` from `BaseException` instead of `RuntimeError` so application `except Exception:` retry and fallback wrappers unwind cleanly while `finally:` cleanup still executes.
- Changed `probe.disabled()` default scope from `ContextVar`-only to a thread-safe process-wide reference count (`scope="process"`) so worker pools such as `ThreadPoolExecutor` honor disabled probes automatically.
- Changed `source_hash` in `probe check` / `.probes/lock.json` to hash `(signature, docstring)` for `@probe` and `(enclosing_signature, watched_keys)` for `probe.here` instead of full function bodies/context lines, preventing lock invalidation on internal refactors or logging edits.

## [0.2.1] - 2026-11-18

### Changed
- Raised default `[text].max_tokens` from `400` to `1024` to avoid avoidable response truncation on multi-check or thinking-model turns.
- Changed default `[runtime].strict_api` from `false` to `true` so `PROBES=1` runs halt on backend/API errors by default rather than letting test suites pass silently with ungraded probes.

### Fixed
- Added `ModelTruncatedError` (`error_code = "PROBE_TRUNCATED"`) capturing partial model text on `finish_reason == "length"`; runtime `probe.type.text` truncation now always halts execution even when `strict_api = false`.
- Preserved partial explanation text plus an explicit truncation or unavailability note when `explain_failure` fails or hits `max_tokens` instead of dropping `analysis_text` to `None`.
- Printed a `[PROBE ERROR]` warning to `stderr` whenever `strict_api = false` and a non-truncation `BACKEND_ERROR` is recorded, eliminating silent error swallowing.

## [0.2.0] - 2026-11-18

### Added
- `domain_context` top-level configuration field (in `config.toml` and `[tool.probe]`) to share project-specific domain conventions across `probe check` (Gates 2–3) and runtime evaluation without stuffing carve-outs into 200-character checks.
- `probe.disabled()` (and `from probe import disabled`) `ContextVar`-scoped context manager to suppress probes cleanly inside mocked unit tests when `PROBES=1` is enabled globally.
- Automatic 1x synthetic fixture retry during `probe check` Gate 3 when attempt 1 fails snapshot validation or calibration, feeding back specific validation/score errors at `temperature = 0.3` (manual `.probes/fixtures/` files are never retried).
- ONNX NLI configuration validation and `onnx_output` (`"logits"` | `"probabilities"`) with explicit `entailment_index` / `contradiction_index` precedence.
- GitHub Release wheel and sdist distribution assets for container and CI installs without requiring `git`.

### Changed
- Increased `[rate_limit].max_calls` default from `5` to `20` per `10.0` seconds to accommodate multi-turn test and agent workflows cleanly.
- Updated `probe check`, `probe list`, and `probe.verify_lock` for PEP 420 namespace packages, stacked decorators (`inspect.unwrap`), comprehension/generator-expression scopes, lexical import shadowing, cross-file probe moves during partial checks, directory/deleted-file lock pruning, `pass_threshold` + `domain_context` lock hash invalidation, and runtime `param_names`/wrong-file verification.
- Updated ONNX NLI 3-class scoring to use `P(entailment)` directly, preventing high-neutral rows from passing.
- Updated provider schema translation for `deepseek-reasoner`, Gemini (`additionalProperties` stripping), OpenAI `strict` mode checks, HTTP 400 `json_schema` -> `json_object` 1x fallback, and localized Ollama `reasoning_effort = "none"` detection (`:11434` / `ollama`).
- Hardened CLI commands (`report`, `log`, `runs`, `clean`) with consistent clean error exits on invalid configs or I/O errors, single-pass `get_run_with_events` in `report`, and `[kind=...]` + `**(HALT TRIGGER)**` annotations in `log --format md`.

### Fixed
- Fixed per-run JSONL store file permissions (`0o700` dir, `0o600` files), inner `runs/.gitignore` (`*\n!.gitignore\n`), torn trailing newline repair, canonical run ID resolution, preserving `"halted"` over later `"completed"` markers, and finite numeric validation for scores.
- Fixed async wrapper post-`await` `ensure_not_tripped()` checks, TTY `pdb` entry-time `ProbeHaltError` handling, `pytest` `PYTEST_CURRENT_TEST` guard, closed runtime `try/finally` resource cleanup, and bounded exception scrubbing.
- Fixed redaction secret-shrinkage boundary truncation, multibyte UTF-8 slice leaks, `quote`/`quote_plus` URL variants, passwordless database URL over-redaction, bounded iterator/generator/custom sequence handling, and uninitialized dataclass `field(init=False)` safety.
- Fixed `extract_json_object` outer code fence matching with embedded triple backticks, truncation/refusal errors in OpenAI-compat responses, and closed HTTP client error wrapping.

### Security
- Prevented untrusted `pyproject.toml` files from silently redirecting configured remote model endpoints or stealing API keys unless `PROBE_TRUST_PROJECT_ENDPOINTS=1` is explicitly set.
- Escaped closing `</` sequences inside all XML-delimited model prompt blocks and kept `Signature:` inside `<source_excerpt>`.
- Hardened config path containment (`resolve_api_key` `is_relative_to`), strict unknown config key rejection, config value bounds validation, and hash-disambiguated `.probes/fixtures/` loading with `probe_id` verification.

## [0.1.0] - 2026-11-17

### Added
- Initial release of `probe`: `@probe`, `probe.here`, `probe.type.{decision,text}`, pre-run `probe check` (Gates 1–3 with `.probes/lock.json`), secret redaction, rate-limit circuit breaker (`PROBE_MISUSE`), per-run ULID JSONL logs, and Typer CLI (`check`, `list`, `report`, `log`, `runs`, `clean`).
