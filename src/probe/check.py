"""Static AST discovery, Gates 1-3 pre-run checks, and .probes/lock.json management."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from probe.backends import BackendError, CheckScore, DecisionBackend, TextBackend
from probe.config import ProbeConfig, load_config, utc_now_iso
from probe.prompts import get_nouls
from probe.runtime import ProbeType, compute_checks_hash

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".probes",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)


@dataclass(slots=True, frozen=True)
class DiscoveredProbe:
    """Static AST representation of a @probe decorator or probe.here(...) call."""

    probe_id: str
    site: str  # "decorator" | "here"
    probe_type: ProbeType
    when: str | None
    file_path: str
    lineno: int
    file_line: str
    checks: tuple[str, ...]
    literal_error: str | None
    signature: str
    source_excerpt: str
    checks_hash: str
    source_hash: str
    combined_hash: str
    watched_keys: tuple[str, ...] = ()


@dataclass(slots=True)
class ProbeCheckResult:
    """Outcome of running Gates 1-3 for one discovered probe."""

    probe_id: str
    file_line: str
    site: str
    probe_type: str
    status: str  # "passed" | "cached" | "failed"
    gate: str  # "gate1" | "gate2" | "gate3" | "cache" | "done"
    reason: str = ""
    checks: tuple[str, ...] = ()
    pass_scores: list[dict[str, Any]] = field(default_factory=list)
    fail_scores: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_python_files(paths: Sequence[Path], root: Path) -> list[Path]:
    targets = list(paths) if paths else [root]
    files: list[Path] = []
    seen: set[Path] = set()

    for target in targets:
        p = target if target.is_absolute() else (root / target)
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix == ".py":
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    files.append(rp)
            continue

        for dirpath, dirnames, filenames in p.walk():
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for fname in sorted(filenames):
                if fname.endswith(".py"):
                    fp = (dirpath / fname).resolve()
                    if fp not in seen:
                        seen.add(fp)
                        files.append(fp)

    return sorted(files)


def _module_name_for_file(file_path: Path, root: Path) -> str:
    """Resolve Python dotted module name matching runtime __module__ when possible."""
    resolved_file = file_path.resolve()
    resolved_root = root.resolve()

    search_roots = []
    if (resolved_root / "src").is_dir():
        search_roots.append((resolved_root / "src").resolve())
    search_roots.append(resolved_root)

    rel: Path | None = None
    for base in search_roots:
        try:
            rel = resolved_file.relative_to(base)
            break
        except ValueError:
            continue

    if rel is None:
        return resolved_file.stem

    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else resolved_file.stem


def _collect_probe_imports(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return local symbol aliases bound to `probe` and `probe.here`."""
    probe_aliases: set[str] = {"probe"}
    here_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "probe":
                    probe_aliases.add(alias.asname or "probe")
        elif isinstance(node, ast.ImportFrom) and node.module == "probe":
            for alias in node.names:
                if alias.name == "probe":
                    probe_aliases.add(alias.asname or "probe")
                elif alias.name == "here":
                    here_aliases.add(alias.asname or "here")

    return probe_aliases, here_aliases


def _is_probe_decorator_call(node: ast.expr, probe_aliases: set[str]) -> ast.Call | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id in probe_aliases:
        return node
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "probe"
        and isinstance(func.value, ast.Name)
        and func.value.id in probe_aliases
    ):
        return node
    return None


def _is_here_call(node: ast.expr, probe_aliases: set[str], here_aliases: set[str]) -> ast.Call | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id in here_aliases:
        return node
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "here"
        and isinstance(func.value, ast.Name)
        and func.value.id in probe_aliases
    ):
        return node
    return None


def _extract_kwarg(call: ast.Call, name: str, pos_index: int | None = None) -> ast.expr | None:
    if pos_index is not None and len(call.args) > pos_index:
        return call.args[pos_index]
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _extract_checks_literal(expr: ast.expr | None) -> tuple[tuple[str, ...], str | None]:
    if expr is None:
        return (), "Missing required `checks` argument"
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        if not expr.value.strip():
            return (), "Check string must not be empty"
        return (expr.value,), None
    if isinstance(expr, (ast.List, ast.Tuple)):
        items: list[str] = []
        for elt in expr.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                return (), "All elements in `checks` must be literal strings for static inspection"
            if not elt.value.strip():
                return (), "Check strings must not be empty"
            items.append(elt.value)
        if not items:
            return (), "`checks` list must not be empty"
        return tuple(items), None
    return (), "`checks` must be a string literal or list/tuple of string literals"


def _extract_str_constant(expr: ast.expr | None) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return None


def _extract_probe_type(expr: ast.expr | None) -> tuple[ProbeType, str | None]:
    if expr is None:
        return ProbeType.decision, None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        try:
            return ProbeType(expr.value.strip().lower()), None
        except ValueError:
            return ProbeType.decision, f"Unknown probe kind: {expr.value!r}"
    if isinstance(expr, ast.Attribute):
        attr = expr.attr.strip().lower()
        if attr in {"decision", "text"}:
            return ProbeType(attr), None
    return ProbeType.decision, f"Unsupported dynamic probe kind expression: {ast.unparse(expr)}"


def _extract_when_cond(expr: ast.expr | None) -> tuple[str | None, str | None]:
    if expr is None:
        return None, None
    if isinstance(expr, ast.Constant) and expr.value is None:
        return None, None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        stripped = expr.value.strip()
        if not stripped:
            return None, "`when` condition string must not be empty"
        return stripped, None
    return None, f"`when` must be a string literal or None: {ast.unparse(expr)}"


def _format_signature(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(fn_node, ast.AsyncFunctionDef) else "def"
    args_src = ast.unparse(fn_node.args)
    ret_src = f" -> {ast.unparse(fn_node.returns)}" if fn_node.returns is not None else ""
    return f"{prefix} {fn_node.name}({args_src}){ret_src}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _ModuleProbeVisitor(ast.NodeVisitor):
    """AST visitor collecting @probe decorators and probe.here(...) calls in one module."""

    def __init__(
        self,
        *,
        source: str,
        module_name: str,
        display_path: str,
        probe_aliases: set[str],
        here_aliases: set[str],
    ) -> None:
        self.source = source
        self.lines = source.splitlines()
        self.module_name = module_name
        self.display_path = display_path
        self.probe_aliases = probe_aliases
        self.here_aliases = here_aliases
        self.scope: list[str] = []
        self.fn_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.discovered: list[DiscoveredProbe] = []

    def _qualname(self, extra: str | None = None) -> str:
        parts = list(self.scope)
        if extra:
            parts.append(extra)
        return ".".join(parts)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = self._qualname(node.name)
        for dec in node.decorator_list:
            call = _is_probe_decorator_call(dec, self.probe_aliases)
            if call is None:
                continue
            checks, check_err = _extract_checks_literal(_extract_kwarg(call, "checks", 0))
            explicit_id = _extract_str_constant(_extract_kwarg(call, "id"))
            probe_id = explicit_id or f"{self.module_name}.{qualname}"
            kind, kind_err = _extract_probe_type(_extract_kwarg(call, "kind"))
            when_cond, when_err = _extract_when_cond(_extract_kwarg(call, "when"))
            literal_error = check_err or kind_err or when_err
            sig = _format_signature(node)
            body_excerpt = ast.get_source_segment(self.source, node) or sig
            if len(body_excerpt) > 1500:
                body_excerpt = body_excerpt[:1500]
            c_hash = compute_checks_hash(kind, checks)
            s_hash = _sha256_text(f"{sig}\n{body_excerpt}")
            combined = _sha256_text(f"{c_hash}:{s_hash}")
            self.discovered.append(
                DiscoveredProbe(
                    probe_id=probe_id,
                    site="decorator",
                    probe_type=kind,
                    when=when_cond,
                    file_path=self.display_path,
                    lineno=node.lineno,
                    file_line=f"{self.display_path}:{node.lineno}",
                    checks=checks,
                    literal_error=literal_error,
                    signature=sig,
                    source_excerpt=body_excerpt,
                    checks_hash=c_hash,
                    source_hash=s_hash,
                    combined_hash=combined,
                )
            )

        self.scope.append(node.name)
        self.fn_stack.append(node)
        self.generic_visit(node)
        self.fn_stack.pop()
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        call = _is_here_call(node, self.probe_aliases, self.here_aliases)
        if call is not None:
            checks, check_err = _extract_checks_literal(_extract_kwarg(call, "checks", 0))
            explicit_id = _extract_str_constant(_extract_kwarg(call, "id"))
            qual = self._qualname()
            default_id = (
                f"{self.module_name}.{qual}:L{call.lineno}" if qual else f"{self.module_name}:L{call.lineno}"
            )
            probe_id = explicit_id or default_id
            kind, kind_err = _extract_probe_type(_extract_kwarg(call, "kind"))
            when_cond, when_err = _extract_when_cond(_extract_kwarg(call, "when"))
            literal_error = check_err or kind_err or when_err
            enclosing = self.fn_stack[-1] if self.fn_stack else None
            sig = _format_signature(enclosing) if enclosing else "<module>"
            start_idx = max(0, call.lineno - 6)
            end_idx = min(len(self.lines), call.end_lineno + 3 if call.end_lineno else call.lineno + 3)
            excerpt = "\n".join(self.lines[start_idx:end_idx])
            c_hash = compute_checks_hash(kind, checks)
            s_hash = _sha256_text(f"{sig}\n{excerpt}")
            combined = _sha256_text(f"{c_hash}:{s_hash}")
            watched_keys = tuple(
                kw.arg for kw in call.keywords if kw.arg and kw.arg not in {"checks", "id", "kind", "when"}
            )
            self.discovered.append(
                DiscoveredProbe(
                    probe_id=probe_id,
                    site="here",
                    probe_type=kind,
                    when=when_cond,
                    file_path=self.display_path,
                    lineno=call.lineno,
                    file_line=f"{self.display_path}:{call.lineno}",
                    checks=checks,
                    literal_error=literal_error,
                    signature=sig,
                    source_excerpt=excerpt,
                    checks_hash=c_hash,
                    source_hash=s_hash,
                    combined_hash=combined,
                    watched_keys=watched_keys,
                )
            )
        self.generic_visit(node)


def discover_probes(
    paths: Sequence[Path] = (),
    *,
    root: Path | None = None,
) -> list[DiscoveredProbe]:
    """Statically scan Python files via AST for @probe(...) and probe.here(...) sites."""
    base_root = (root or Path.cwd()).resolve()
    py_files = _iter_python_files(paths, base_root)
    discovered: list[DiscoveredProbe] = []

    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, SyntaxError):
            continue

        module_name = _module_name_for_file(py_file, base_root)
        probe_aliases, here_aliases = _collect_probe_imports(tree)
        try:
            display_path = str(py_file.relative_to(base_root))
        except ValueError:
            display_path = str(py_file)

        visitor = _ModuleProbeVisitor(
            source=source,
            module_name=module_name,
            display_path=display_path,
            probe_aliases=probe_aliases,
            here_aliases=here_aliases,
        )
        visitor.visit(tree)
        discovered.extend(visitor.discovered)

    return discovered


def _load_lockfile(lock_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        return {"version": 1, "probes": {}}
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("probes"), dict):
            return raw
    except Exception:
        pass
    return {"version": 1, "probes": {}}


def _save_lockfile(lock_path: Path, data: dict[str, Any]) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_manual_fixture(fixtures_dir: Path, probe_id: str) -> tuple[str, str] | None:
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", probe_id)
    for name in (f"{probe_id}.json", f"{safe_id}.json"):
        fp = fixtures_dir / name
        if fp.is_file():
            data = json.loads(fp.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise BackendError(f"Fixture {fp} must contain a JSON object with 'pass' and 'fail' keys")
            passing = data.get("pass", data.get("passing_snapshot"))
            failing = data.get("fail", data.get("failing_snapshot"))
            if passing is None or failing is None:
                raise BackendError(f"Fixture {fp} must define both 'pass' and 'fail' snapshots")
            pass_json = passing if isinstance(passing, str) else json.dumps(passing, ensure_ascii=False)
            fail_json = failing if isinstance(failing, str) else json.dumps(failing, ensure_ascii=False)
            return pass_json, fail_json
    return None


def _evaluate_sample(
    *,
    item: DiscoveredProbe,
    snapshot_json: str,
    decision_backend: DecisionBackend,
    text_backend: TextBackend,
) -> list[CheckScore]:
    analysis = ""
    if item.probe_type == ProbeType.text:
        analysis = text_backend.analyze_snapshot(
            probe_id=item.probe_id,
            checks=item.checks,
            snapshot_json=snapshot_json,
        )
    return decision_backend.evaluate_checks(
        probe_id=item.probe_id,
        checks=item.checks,
        snapshot_json=snapshot_json,
        analysis=analysis,
    )


def run_probe_check(
    paths: Sequence[Path] = (),
    *,
    force: bool = False,
    config: ProbeConfig | None = None,
    transport: httpx.BaseTransport | None = None,
) -> tuple[bool, list[ProbeCheckResult]]:
    """Run Gates 1-3 across discovered probes and update .probes/lock.json."""
    cfg = config or load_config()
    root = cfg.project_root
    discovered = discover_probes(paths, root=root)

    lock_path = cfg.resolve_lock_path()
    lock_data = _load_lockfile(lock_path)
    locked_probes: dict[str, Any] = dict(lock_data.get("probes", {}))
    fixtures_dir = lock_path.parent / "fixtures"

    id_counts = Counter(p.probe_id for p in discovered)
    results: list[ProbeCheckResult] = []
    decision_backend = DecisionBackend(cfg.models_decision, cfg.decision, transport=transport)
    text_backend = TextBackend(cfg.models_text, cfg.text, transport=transport)

    try:
        for item in discovered:
            # Gate 1: Budgets, static literals, duplicate IDs.
            gate1_err: str | None = None
            if id_counts[item.probe_id] > 1:
                gate1_err = f"Duplicate probe_id '{item.probe_id}' ({id_counts[item.probe_id]} sites)"
            elif item.literal_error:
                gate1_err = item.literal_error
            elif len(item.checks) > cfg.check.max_checks:
                gate1_err = f"Too many checks ({len(item.checks)} > max_checks={cfg.check.max_checks})"
            else:
                overlong = [c for c in item.checks if len(c) > cfg.check.max_check_chars]
                if overlong:
                    gate1_err = (
                        f"Check exceeds max_check_chars={cfg.check.max_check_chars} "
                        f"({len(overlong[0])} chars)"
                    )

            if gate1_err:
                locked_probes.pop(item.probe_id, None)
                results.append(
                    ProbeCheckResult(
                        probe_id=item.probe_id,
                        file_line=item.file_line,
                        site=item.site,
                        probe_type=item.probe_type.value,
                        status="failed",
                        gate="gate1",
                        reason=gate1_err,
                        checks=item.checks,
                    )
                )
                continue

            existing = locked_probes.get(item.probe_id)
            if (
                not force
                and isinstance(existing, dict)
                and existing.get("combined_hash") == item.combined_hash
            ):
                results.append(
                    ProbeCheckResult(
                        probe_id=item.probe_id,
                        file_line=item.file_line,
                        site=item.site,
                        probe_type=item.probe_type.value,
                        status="cached",
                        gate="cache",
                        reason="Unchanged since last approved lock entry",
                        checks=item.checks,
                    )
                )
                continue

            # Gate 2: Factual analysis via [models.text] + noul approval via [models.decision].
            try:
                gate2_analysis = text_backend.analyze_checks_gate2(
                    probe_id=item.probe_id,
                    site=item.site,
                    checks=item.checks,
                    signature=item.signature,
                    source_excerpt=item.source_excerpt,
                    watched_keys=item.watched_keys,
                )
                gate2_state = json.dumps(
                    {
                        "site": item.site,
                        "signature": item.signature,
                        "watched_keys": list(item.watched_keys),
                        "checks": list(item.checks),
                    },
                    ensure_ascii=False,
                )
                gate2_scores = decision_backend.evaluate_checks(
                    probe_id=f"{item.probe_id}#gate2",
                    checks=get_nouls("check_gate2_nouls"),
                    snapshot_json=gate2_state,
                    analysis=gate2_analysis,
                )
            except BackendError as exc:
                locked_probes.pop(item.probe_id, None)
                results.append(
                    ProbeCheckResult(
                        probe_id=item.probe_id,
                        file_line=item.file_line,
                        site=item.site,
                        probe_type=item.probe_type.value,
                        status="failed",
                        gate="gate2",
                        reason=str(exc),
                        checks=item.checks,
                    )
                )
                continue

            approved = bool(gate2_scores) and all(s.passed for s in gate2_scores)
            if not approved:
                failed_nouls = [s.check for s in gate2_scores if not s.passed]
                audit_reason = (
                    f"{gate2_analysis} (failed: {'; '.join(failed_nouls)})".strip()
                    if gate2_analysis
                    else "; ".join(failed_nouls)
                )
                locked_probes.pop(item.probe_id, None)
                results.append(
                    ProbeCheckResult(
                        probe_id=item.probe_id,
                        file_line=item.file_line,
                        site=item.site,
                        probe_type=item.probe_type.value,
                        status="failed",
                        gate="gate2",
                        reason=audit_reason or "Rejected by Gate 2 check audit",
                        checks=item.checks,
                    )
                )
                continue
            audit_reason = "Passed Gate 2 check audit"

            # Gate 3: Dry-run calibration against fixture or synthesized samples.
            try:
                fixture = _load_manual_fixture(fixtures_dir, item.probe_id)
                if fixture is not None:
                    pass_snap, fail_snap = fixture
                    fixture_source = "fixture"
                else:
                    pass_snap, fail_snap = text_backend.generate_calibration_fixtures_gate3(
                        probe_id=item.probe_id,
                        site=item.site,
                        checks=item.checks,
                        signature=item.signature,
                        source_excerpt=item.source_excerpt,
                        watched_keys=item.watched_keys,
                    )
                    fixture_source = "synthetic"

                pass_scores = _evaluate_sample(
                    item=item,
                    snapshot_json=pass_snap,
                    decision_backend=decision_backend,
                    text_backend=text_backend,
                )
                fail_scores = _evaluate_sample(
                    item=item,
                    snapshot_json=fail_snap,
                    decision_backend=decision_backend,
                    text_backend=text_backend,
                )
            except BackendError as exc:
                locked_probes.pop(item.probe_id, None)
                results.append(
                    ProbeCheckResult(
                        probe_id=item.probe_id,
                        file_line=item.file_line,
                        site=item.site,
                        probe_type=item.probe_type.value,
                        status="failed",
                        gate="gate3",
                        reason=str(exc),
                        checks=item.checks,
                    )
                )
                continue

            pass_ok = bool(pass_scores) and all(s.passed for s in pass_scores)
            fail_ok = bool(fail_scores) and any(not s.passed for s in fail_scores)

            pass_dicts = [
                {"check": s.check, "probability": s.probability, "passed": s.passed}
                for s in pass_scores
            ]
            fail_dicts = [
                {"check": s.check, "probability": s.probability, "passed": s.passed}
                for s in fail_scores
            ]

            if not (pass_ok and fail_ok):
                locked_probes.pop(item.probe_id, None)
                parts = []
                if not pass_ok:
                    parts.append("passing fixture did not pass all checks")
                if not fail_ok:
                    parts.append("failing fixture did not fail any check")
                results.append(
                    ProbeCheckResult(
                        probe_id=item.probe_id,
                        file_line=item.file_line,
                        site=item.site,
                        probe_type=item.probe_type.value,
                        status="failed",
                        gate="gate3",
                        reason=f"Calibration failed ({'; '.join(parts)})",
                        checks=item.checks,
                        pass_scores=pass_dicts,
                        fail_scores=fail_dicts,
                    )
                )
                continue

            locked_probes[item.probe_id] = {
                "probe_id": item.probe_id,
                "site": item.site,
                "kind": item.probe_type.value,
                "file_line": item.file_line,
                "checks": list(item.checks),
                "checks_hash": item.checks_hash,
                "source_hash": item.source_hash,
                "combined_hash": item.combined_hash,
                "source": fixture_source,
                "approved_at": utc_now_iso(),
            }
            results.append(
                ProbeCheckResult(
                    probe_id=item.probe_id,
                    file_line=item.file_line,
                    site=item.site,
                    probe_type=item.probe_type.value,
                    status="passed",
                    gate="done",
                    reason=audit_reason or "Passed Gates 1-3",
                    checks=item.checks,
                    pass_scores=pass_dicts,
                    fail_scores=fail_dicts,
                )
            )
    finally:
        decision_backend.close()
        text_backend.close()

    # Remove stale entries for checked files whose probes no longer exist.
    discovered_ids = {p.probe_id for p in discovered}
    if not paths:
        locked_probes = {pid: meta for pid, meta in locked_probes.items() if pid in discovered_ids}
    else:
        checked_prefixes: set[str] = set()
        for py_file in _iter_python_files(paths, root):
            try:
                checked_prefixes.add(str(py_file.relative_to(root.resolve())))
            except ValueError:
                checked_prefixes.add(str(py_file))
        locked_probes = {
            pid: meta
            for pid, meta in locked_probes.items()
            if pid in discovered_ids
            or not any(str(meta.get("file_line", "")).startswith(f"{fp}:") for fp in checked_prefixes)
        }

    lock_data = {"version": 1, "probes": locked_probes}
    _save_lockfile(lock_path, lock_data)

    all_ok = all(r.status in {"passed", "cached"} for r in results)
    return all_ok, results
