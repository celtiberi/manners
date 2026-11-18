"""Static AST discovery, Gates 1-3 pre-run checks, and .probes/lock.json management."""

from __future__ import annotations

import ast
import hashlib
import json
import os
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
from probe.runtime import ProbeType, compute_checks_hash, resolve_module_name

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

        for dirpath_str, dirnames, filenames in os.walk(p):
            dirpath = Path(dirpath_str)
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
    return resolve_module_name(file_path, root)


def _collect_probe_imports(tree: ast.Module) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return local symbol aliases bound to `probe` module, `probe` API, `here`, and `ProbeType`."""
    module_aliases: set[str] = set()
    api_aliases: set[str] = set()
    here_aliases: set[str] = set()
    type_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "probe":
                    module_aliases.add(alias.asname or "probe")
                elif alias.name.startswith("probe.") and alias.asname is None:
                    module_aliases.add("probe")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == "probe":
                for alias in node.names:
                    if alias.name == "probe":
                        api_aliases.add(alias.asname or "probe")
                    elif alias.name == "here":
                        here_aliases.add(alias.asname or "here")
                    elif alias.name in {"ProbeType", "type"}:
                        type_aliases.add(alias.asname or alias.name)
                    elif alias.name == "*":
                        api_aliases.add("probe")
                        here_aliases.add("here")
                        type_aliases.add("ProbeType")
            elif node.module == "probe.runtime":
                for alias in node.names:
                    if alias.name == "ProbeType":
                        type_aliases.add(alias.asname or "ProbeType")

    return module_aliases, api_aliases, here_aliases, type_aliases


def _classify_probe_decorator(
    node: ast.expr,
    module_aliases: set[str],
    api_aliases: set[str],
) -> tuple[ast.Call | None, str | None, bool]:
    """Identify @probe decorators and catch common bare/module-call misuse at Gate 1."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in api_aliases:
                return node, None, True
            if func.id in module_aliases:
                return (
                    node,
                    "Module `probe` is not callable; use `@probe.probe(...)` or `from probe import probe`",
                    True,
                )
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "probe"
            and isinstance(func.value, ast.Name)
        ):
            if func.value.id in module_aliases:
                return node, None, True
            if func.value.id in api_aliases:
                return (
                    node,
                    "`probe` imported via `from probe import probe` has no `.probe` attribute; use `@probe(...)`",
                    True,
                )
        return None, None, False

    if isinstance(node, ast.Name) and node.id in (api_aliases | module_aliases):
        return None, "Bare `@probe` requires arguments; use `@probe(\"...\")` or `@probe.probe(\"...\")`", True
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "probe"
        and isinstance(node.value, ast.Name)
        and node.value.id in (api_aliases | module_aliases)
    ):
        return None, "Bare `@probe.probe` requires arguments; use `@probe.probe(\"...\")`", True
    return None, None, False


def _is_here_call(
    node: ast.expr,
    module_aliases: set[str],
    api_aliases: set[str],
    here_aliases: set[str],
) -> ast.Call | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id in here_aliases:
        return node
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "here"
        and isinstance(func.value, ast.Name)
        and func.value.id in (api_aliases | module_aliases)
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


def _extract_probe_type(
    expr: ast.expr | None,
    *,
    module_aliases: set[str],
    api_aliases: set[str],
    type_aliases: set[str],
) -> tuple[ProbeType, str | None]:
    if expr is None:
        return ProbeType.decision, None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        try:
            return ProbeType(expr.value.strip().lower()), None
        except ValueError:
            return ProbeType.decision, f"Unknown probe kind: {expr.value!r}"
    if isinstance(expr, ast.Attribute) and expr.attr in {"decision", "text"}:
        val = expr.value
        valid = (
            (isinstance(val, ast.Name) and val.id in type_aliases)
            or (
                isinstance(val, ast.Attribute)
                and val.attr == "type"
                and isinstance(val.value, ast.Name)
                and val.value.id in (api_aliases | module_aliases)
            )
            or (
                isinstance(val, ast.Attribute)
                and val.attr == "ProbeType"
                and isinstance(val.value, ast.Name)
                and val.value.id in module_aliases
            )
        )
        if valid:
            return ProbeType(expr.attr), None
    return ProbeType.decision, f"Unsupported probe kind expression: {ast.unparse(expr)}"


def _validate_call_structure_and_id(
    call: ast.Call,
    *,
    site: str,
) -> tuple[str | None, str | None]:
    """Validate positional/keyword arguments and optional literal `id` on @probe or probe.here."""
    if any(isinstance(a, ast.Starred) for a in call.args):
        return None, f"Dynamic *args is not allowed in {site}; use static literals"
    if len(call.args) > 1:
        return None, f"{site} accepts at most 1 positional argument (checks)"

    seen_kws: set[str] = set()
    for kw in call.keywords:
        if kw.arg is None:
            return None, f"Dynamic **kwargs is not allowed in {site}"
        if kw.arg in seen_kws:
            return None, f"Duplicate keyword argument '{kw.arg}' in {site}"
        seen_kws.add(kw.arg)
        if site == "@probe" and kw.arg not in {"checks", "id", "kind"}:
            return None, f"Unknown @probe keyword argument: {kw.arg!r}"

    if call.args and "checks" in seen_kws:
        return None, f"Duplicate 'checks' passed positionally and by keyword in {site}"

    id_expr = _extract_kwarg(call, "id")
    if id_expr is None:
        return None, None
    if not (isinstance(id_expr, ast.Constant) and isinstance(id_expr.value, str)):
        return None, "`id` must be a string literal"
    raw_id = id_expr.value
    if not raw_id.strip():
        return None, "`id` must not be empty"
    if "/" in raw_id or "\\" in raw_id or ".." in raw_id:
        return None, "`id` must not contain '/', '\\\\', or '..'"
    return raw_id, None


def _is_generator_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if function body yields directly (ignoring nested defs/classes/lambdas)."""
    stack: list[ast.AST] = list(node.body)
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(cur))
    return False


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
        module_aliases: set[str],
        api_aliases: set[str],
        here_aliases: set[str],
        type_aliases: set[str],
    ) -> None:
        self.source = source
        self.lines = source.splitlines()
        self.module_name = module_name
        self.display_path = display_path
        self.module_aliases = module_aliases
        self.api_aliases = api_aliases
        self.here_aliases = here_aliases
        self.type_aliases = type_aliases
        self.scope: list[tuple[str, bool]] = []
        self.fn_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.discovered: list[DiscoveredProbe] = []

    def _qualname(self, extra: str | None = None) -> str:
        items = list(self.scope)
        if extra:
            items.append((extra, False))
        parts: list[str] = []
        for idx, (name, is_fn) in enumerate(items):
            parts.append(name)
            if is_fn and idx < len(items) - 1:
                parts.append("<locals>")
        return ".".join(parts)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = self._qualname(node.name)
        is_gen = _is_generator_function(node)
        gen_err = "Generator functions cannot be decorated with @probe" if is_gen else None
        sig = _format_signature(node)
        full_source = ast.get_source_segment(self.source, node) or ast.unparse(node)
        body_excerpt = full_source[:1500]
        s_hash = _sha256_text(f"{sig}\n{full_source}")

        for dec in node.decorator_list:
            call, call_err, matched = _classify_probe_decorator(dec, self.module_aliases, self.api_aliases)
            if not matched:
                continue
            if call is None:
                c_hash = compute_checks_hash(ProbeType.decision, ())
                self.discovered.append(
                    DiscoveredProbe(
                        probe_id=f"{self.module_name}.{qualname}",
                        site="decorator",
                        probe_type=ProbeType.decision,
                        file_path=self.display_path,
                        lineno=node.lineno,
                        file_line=f"{self.display_path}:{node.lineno}",
                        checks=(),
                        literal_error=call_err,
                        signature=sig,
                        source_excerpt=body_excerpt,
                        checks_hash=c_hash,
                        source_hash=s_hash,
                        combined_hash=_sha256_text(f"{c_hash}:{s_hash}"),
                    )
                )
                continue

            checks, check_err = _extract_checks_literal(_extract_kwarg(call, "checks", 0))
            explicit_id, struct_err = _validate_call_structure_and_id(call, site="@probe")
            probe_id = explicit_id or f"{self.module_name}.{qualname}"
            kind, kind_err = _extract_probe_type(
                _extract_kwarg(call, "kind"),
                module_aliases=self.module_aliases,
                api_aliases=self.api_aliases,
                type_aliases=self.type_aliases,
            )
            literal_error = call_err or struct_err or check_err or kind_err or gen_err
            c_hash = compute_checks_hash(kind, checks)
            combined = _sha256_text(f"{c_hash}:{s_hash}")
            self.discovered.append(
                DiscoveredProbe(
                    probe_id=probe_id,
                    site="decorator",
                    probe_type=kind,
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

        for dec in node.decorator_list:
            self.visit(dec)
        for default in node.args.defaults:
            self.visit(default)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                self.visit(kw_default)
        for arg in (*node.args.posonlyargs, *node.args.args, node.args.vararg, *node.args.kwonlyargs, node.args.kwarg):
            if arg is not None and arg.annotation is not None:
                self.visit(arg.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        for tp in getattr(node, "type_params", ()):
            self.visit(tp)

        self.scope.append((node.name, True))
        self.fn_stack.append(node)
        for stmt in node.body:
            self.visit(stmt)
        self.fn_stack.pop()
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        for kw in node.keywords:
            self.visit(kw)
        for tp in getattr(node, "type_params", ()):
            self.visit(tp)
        self.scope.append((node.name, False))
        for stmt in node.body:
            self.visit(stmt)
        self.scope.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults:
            self.visit(default)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                self.visit(kw_default)
        self.scope.append(("<lambda>", True))
        self.visit(node.body)
        self.scope.pop()

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        if node.generators:
            self.visit(node.generators[0].iter)
        self.scope.append(("<genexpr>", True))
        for idx, comp in enumerate(node.generators):
            if idx > 0:
                self.visit(comp.iter)
            self.visit(comp.target)
            for if_expr in comp.ifs:
                self.visit(if_expr)
        self.visit(node.elt)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        call = _is_here_call(node, self.module_aliases, self.api_aliases, self.here_aliases)
        if call is not None:
            checks, check_err = _extract_checks_literal(_extract_kwarg(call, "checks", 0))
            explicit_id, struct_err = _validate_call_structure_and_id(call, site="probe.here")
            qual = self._qualname()
            default_id = (
                f"{self.module_name}.{qual}:L{call.lineno}" if qual else f"{self.module_name}:L{call.lineno}"
            )
            probe_id = explicit_id or default_id
            kind, kind_err = _extract_probe_type(
                _extract_kwarg(call, "kind"),
                module_aliases=self.module_aliases,
                api_aliases=self.api_aliases,
                type_aliases=self.type_aliases,
            )
            watched_keys = tuple(
                kw.arg for kw in call.keywords if kw.arg and kw.arg not in {"checks", "id", "kind"}
            )
            watched_err = None if watched_keys else "probe.here requires at least one watched keyword variable"
            literal_error = struct_err or check_err or kind_err or watched_err
            enclosing = self.fn_stack[-1] if self.fn_stack else None
            sig = _format_signature(enclosing) if enclosing else "<module>"
            start_idx = max(0, call.lineno - 6)
            end_idx = min(len(self.lines), call.end_lineno + 3 if call.end_lineno else call.lineno + 3)
            excerpt = "\n".join(self.lines[start_idx:end_idx])
            c_hash = compute_checks_hash(kind, checks)
            s_hash = _sha256_text(f"{sig}\n{excerpt}")
            combined = _sha256_text(f"{c_hash}:{s_hash}")
            self.discovered.append(
                DiscoveredProbe(
                    probe_id=probe_id,
                    site="here",
                    probe_type=kind,
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


def _rel_display_path(py_file: Path, base_root: Path) -> str:
    try:
        return str(py_file.relative_to(base_root))
    except ValueError:
        return str(py_file)


def _scan_files(
    paths: Sequence[Path] = (),
    *,
    root: Path | None = None,
) -> tuple[list[Path], list[DiscoveredProbe], dict[Path, str]]:
    """Scan Python files via AST, returning (scanned_files, discovered_probes, parse_errors)."""
    base_root = (root or Path.cwd()).resolve()
    py_files = _iter_python_files(paths, base_root)
    discovered: list[DiscoveredProbe] = []
    parse_errors: dict[Path, str] = {}

    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            parse_errors[py_file] = f"{type(exc).__name__}: {exc}"
            continue

        module_aliases, api_aliases, here_aliases, type_aliases = _collect_probe_imports(tree)
        if not (module_aliases or api_aliases or here_aliases):
            continue

        module_name = _module_name_for_file(py_file, base_root)
        display_path = _rel_display_path(py_file, base_root)

        visitor = _ModuleProbeVisitor(
            source=source,
            module_name=module_name,
            display_path=display_path,
            module_aliases=module_aliases,
            api_aliases=api_aliases,
            here_aliases=here_aliases,
            type_aliases=type_aliases,
        )
        visitor.visit(tree)
        discovered.extend(visitor.discovered)

    return py_files, discovered, parse_errors


def discover_probes(
    paths: Sequence[Path] = (),
    *,
    root: Path | None = None,
) -> list[DiscoveredProbe]:
    """Statically scan Python files via AST for @probe(...) and probe.here(...) sites."""
    _, discovered, _ = _scan_files(paths, root=root)
    return discovered


def _load_lockfile(lock_path: Path) -> dict[str, Any]:
    if not lock_path.exists():
        return {"version": 1, "probes": {}}
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendError(f"Corrupt lockfile {lock_path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("probes"), dict):
        raise BackendError(f"Invalid lockfile format in {lock_path}: expected object with 'probes' dict")
    return raw


def _save_lockfile(lock_path: Path, data: dict[str, Any]) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_manual_fixture(fixtures_dir: Path, probe_id: str) -> tuple[str, str] | None:
    if "/" in probe_id or "\\" in probe_id or ".." in probe_id:
        raise BackendError(f"Invalid fixture probe_id {probe_id!r}: path separators and '..' are not allowed")
    fixtures_root = fixtures_dir.resolve()
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", probe_id)
    for name in dict.fromkeys((f"{probe_id}.json", f"{safe_id}.json")):
        fp = fixtures_dir / name
        if not fp.exists():
            continue
        if not fp.resolve().is_relative_to(fixtures_root):
            raise BackendError(f"Fixture path {fp} escapes {fixtures_dir}")
        if not fp.is_file():
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(f"Failed to read fixture {fp}: {exc}") from exc
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


def _matches_file_prefixes(file_line: str, prefixes: set[str]) -> bool:
    return any(file_line == fp or file_line.startswith(f"{fp}:") for fp in prefixes)


def _validate_gate3_snapshot(snapshot_json: str, *, item: DiscoveredProbe, label: str) -> str:
    """Validate Gate 3 pass/fail snapshot JSON matches the site schema."""
    try:
        obj = json.loads(snapshot_json)
    except ValueError as exc:
        raise BackendError(f"Gate 3 {label} for {item.probe_id} is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise BackendError(f"Gate 3 {label} for {item.probe_id} must be a JSON object")

    if item.site == "decorator":
        if not isinstance(obj.get("inputs"), dict):
            raise BackendError(
                f"Gate 3 {label} for decorator probe {item.probe_id} must include dict 'inputs'"
            )
        if "output" not in obj and "exception" not in obj:
            raise BackendError(
                f"Gate 3 {label} for decorator probe {item.probe_id} must include 'output' or 'exception'"
            )
    elif item.site == "here":
        watched = obj.get("watched")
        if not isinstance(watched, dict):
            raise BackendError(
                f"Gate 3 {label} for here probe {item.probe_id} must include dict 'watched'"
            )
        if item.watched_keys:
            missing = [k for k in item.watched_keys if k not in watched]
            if missing:
                raise BackendError(
                    f"Gate 3 {label} for {item.probe_id} missing watched key(s): {', '.join(missing)}"
                )
    return snapshot_json


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
    base_root = root.resolve()
    py_files, discovered, parse_errors = _scan_files(paths, root=root)

    lock_path = cfg.resolve_lock_path()
    lock_data = _load_lockfile(lock_path)
    locked_probes: dict[str, Any] = dict(lock_data.get("probes", {}))
    fixtures_dir = lock_path.parent / "fixtures"

    checked_prefixes = {_rel_display_path(pf, base_root) for pf in py_files}
    error_file_prefixes = {_rel_display_path(pf, base_root) for pf in parse_errors}

    unscanned_locked_ids: set[str] = set()
    if paths:
        for pid, meta in locked_probes.items():
            if not isinstance(meta, dict):
                continue
            meta_fl = str(meta.get("file_line", ""))
            if not meta_fl or _matches_file_prefixes(meta_fl, checked_prefixes):
                continue
            meta_file = meta_fl.rsplit(":", 1)[0] if ":" in meta_fl else meta_fl
            candidate = Path(meta_file) if Path(meta_file).is_absolute() else (base_root / meta_file)
            if candidate.exists():
                unscanned_locked_ids.add(pid)

    id_counts = Counter(p.probe_id for p in discovered)
    results: list[ProbeCheckResult] = [
        ProbeCheckResult(
            probe_id="<parse_error>",
            file_line=_rel_display_path(pf, base_root),
            site="file",
            probe_type="decision",
            status="failed",
            gate="gate1",
            reason=err,
        )
        for pf, err in sorted(parse_errors.items())
    ]
    decision_backend = DecisionBackend(cfg.models_decision, cfg.decision, transport=transport)
    text_backend = TextBackend(cfg.models_text, cfg.text, transport=transport)

    try:
        for item in discovered:
            # Gate 1: Budgets, static literals, duplicate IDs.
            gate1_err: str | None = None
            if item.probe_id in unscanned_locked_ids:
                gate1_err = f"Duplicate probe_id '{item.probe_id}' already locked in another file"
            elif id_counts[item.probe_id] > 1:
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
                if item.probe_id not in unscanned_locked_ids:
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

            try:
                fixture = _load_manual_fixture(fixtures_dir, item.probe_id)
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

            if fixture is not None:
                fix_digest = _sha256_text(f"{fixture[0]}\n{fixture[1]}")
                effective_combined_hash = _sha256_text(f"{item.combined_hash}:fixture:{fix_digest}")
            else:
                effective_combined_hash = item.combined_hash

            existing = locked_probes.get(item.probe_id)
            watched_ok = item.site != "here" or (
                isinstance(existing, dict) and existing.get("watched_keys") == list(item.watched_keys)
            )
            if (
                not force
                and isinstance(existing, dict)
                and existing.get("combined_hash") == effective_combined_hash
                and existing.get("site") == item.site
                and existing.get("kind") == item.probe_type.value
                and watched_ok
            ):
                existing["file_line"] = item.file_line
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

            approved = bool(gate2_scores) and all(s.probability > 0.50 for s in gate2_scores)
            if not approved:
                failed_nouls = [s.check for s in gate2_scores if s.probability <= 0.50]
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

                pass_snap = _validate_gate3_snapshot(pass_snap, item=item, label="passing_snapshot")
                fail_snap = _validate_gate3_snapshot(fail_snap, item=item, label="failing_snapshot")

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
                "watched_keys": list(item.watched_keys),
                "checks_hash": item.checks_hash,
                "source_hash": item.source_hash,
                "combined_hash": effective_combined_hash,
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

    # Remove stale entries for checked files whose probes no longer exist (preserving unparseable files).
    discovered_ids = {p.probe_id for p in discovered}
    if not paths:
        locked_probes = {
            pid: meta
            for pid, meta in locked_probes.items()
            if pid in discovered_ids
            or (
                isinstance(meta, dict)
                and _matches_file_prefixes(str(meta.get("file_line", "")), error_file_prefixes)
            )
        }
    else:
        locked_probes = {
            pid: meta
            for pid, meta in locked_probes.items()
            if pid in discovered_ids
            or (
                isinstance(meta, dict)
                and (
                    _matches_file_prefixes(str(meta.get("file_line", "")), error_file_prefixes)
                    or not _matches_file_prefixes(str(meta.get("file_line", "")), checked_prefixes)
                )
            )
        }

    lock_data = {"version": 1, "probes": locked_probes}
    _save_lockfile(lock_path, lock_data)

    all_ok = all(r.status in {"passed", "cached"} for r in results)
    return all_ok, results
