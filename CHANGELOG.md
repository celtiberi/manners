# Changelog

All notable changes to `probe` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
