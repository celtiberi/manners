"""Static AST discovery, Gates 1-3 pre-run checks, and .probes/lock.json management."""

from __future__ import annotations

import ast
import concurrent.futures
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

from probe.backends import BackendError, CheckScore, DecisionBackend, ModelTruncatedError, TextBackend
from probe.config import ProbeConfig, load_config, utc_now_iso
from probe.prompts import get_noul_expected, get_nouls
from probe.redact import SecretRedactor
from probe.runtime import ProbeType, compute_checks_hash, resolve_module_name
from probe.store import ProbeStore

_VALID_PROBE_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")

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
    param_names: tuple[str, ...] = ()
    required_params: tuple[str, ...] = ()
    ignored_params: tuple[str, ...] = ()
    type_tags: tuple[tuple[str, str], ...] = ()
    return_type_tag: str = "str"


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


def _rel_display_path(py_file: Path, base_root: Path) -> str:
    resolved_file = py_file.resolve()
    resolved_root = base_root.resolve()
    try:
        return resolved_file.relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved_file.as_posix()


def _resolve_scan_targets(
    paths: Sequence[Path],
    root: Path,
) -> tuple[list[Path], set[str], set[str], bool]:
    """Return (py_files, checked_dir_prefixes, explicit_deleted_prefixes, full_root_scan)."""
    base_root = root.resolve()
    targets = list(paths) if paths else [base_root]
    full_root_scan = not paths
    files: list[Path] = []
    seen: set[Path] = set()
    checked_dir_prefixes: set[str] = set()
    explicit_deleted_prefixes: set[str] = set()

    for target in targets:
        p = target if target.is_absolute() else (base_root / target)
        resolved = p.resolve()
        if not p.exists():
            explicit_deleted_prefixes.add(_rel_display_path(resolved, base_root))
            continue
        if p.is_file():
            if p.suffix == ".py":
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(resolved)
            continue

        if resolved == base_root:
            full_root_scan = True
        else:
            checked_dir_prefixes.add(_rel_display_path(resolved, base_root))

        for dirpath_str, dirnames, filenames in os.walk(p):
            dirpath = Path(dirpath_str)
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for fname in sorted(filenames):
                if fname.endswith(".py"):
                    fp = (dirpath / fname).resolve()
                    if fp not in seen:
                        seen.add(fp)
                        files.append(fp)

    return sorted(files), checked_dir_prefixes, explicit_deleted_prefixes, full_root_scan


def _iter_python_files(paths: Sequence[Path], root: Path) -> list[Path]:
    files, _, _, _ = _resolve_scan_targets(paths, root)
    return files


def _module_name_for_file(file_path: Path, root: Path) -> str:
    """Resolve Python dotted module name matching runtime __module__ when possible."""
    return resolve_module_name(file_path, root)


def _file_may_use_probe(tree: ast.Module) -> bool:
    """Fast pre-check before running the scoped AST visitor on a parsed module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "probe" or alias.name.startswith("probe."):
                    return True
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in {"probe", "probe.runtime"}:
            return True
    return False


def _classify_probe_decorator(
    node: ast.expr,
    lookup_role: Any,
) -> tuple[ast.Call | None, str | None, bool]:
    """Identify @probe decorators using lexically scoped symbol roles."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            role = lookup_role(func.id)
            if role == "api":
                return node, None, True
            if role == "module":
                return (
                    node,
                    (
                        "Module `probe` is not callable; use `from probe import probe` "
                        "(which also exposes `probe.type`, `probe.here`, and `probe.disabled`) "
                        "or `@probe.probe(...)`"
                    ),
                    True,
                )
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "probe"
            and isinstance(func.value, ast.Name)
        ):
            role = lookup_role(func.value.id)
            if role == "module":
                return node, None, True
            if role == "api":
                return (
                    node,
                    "`probe` imported via `from probe import probe` has no `.probe` attribute; use `@probe(...)`",
                    True,
                )
        return None, None, False

    if isinstance(node, ast.Name) and lookup_role(node.id) in {"api", "module"}:
        return None, "Bare `@probe` requires arguments; use `@probe(\"...\")` or `@probe.probe(\"...\")`", True
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "probe"
        and isinstance(node.value, ast.Name)
        and lookup_role(node.value.id) in {"api", "module"}
    ):
        return None, "Bare `@probe.probe` requires arguments; use `@probe.probe(\"...\")`", True
    return None, None, False


def _is_here_call(node: ast.expr, lookup_role: Any) -> ast.Call | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and lookup_role(func.id) == "here":
        return node
    if isinstance(func, ast.Attribute) and func.attr == "here":
        if isinstance(func.value, ast.Name) and lookup_role(func.value.id) in {"api", "module"}:
            return node
        if (
            isinstance(func.value, ast.Attribute)
            and func.value.attr == "probe"
            and isinstance(func.value.value, ast.Name)
            and lookup_role(func.value.value.id) == "module"
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


def _extract_ignore_literal(expr: ast.expr | None) -> tuple[tuple[str, ...], str | None]:
    if expr is None or (isinstance(expr, ast.Constant) and expr.value is None):
        return (), None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        if not expr.value.strip():
            return (), "`ignore` parameter name must not be empty"
        return (expr.value,), None
    if isinstance(expr, (ast.List, ast.Tuple)):
        items: list[str] = []
        for elt in expr.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                return (), "All elements in `ignore` must be literal strings for static inspection"
            if not elt.value.strip():
                return (), "`ignore` parameter names must not be empty"
            items.append(elt.value)
        return tuple(dict.fromkeys(items)), None
    return (), "`ignore` must be a string literal or list/tuple of string literals"


def _extract_probe_type(
    expr: ast.expr | None,
    *,
    lookup_role: Any,
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
            (isinstance(val, ast.Name) and lookup_role(val.id) == "type")
            or (
                isinstance(val, ast.Attribute)
                and val.attr == "type"
                and isinstance(val.value, ast.Name)
                and lookup_role(val.value.id) in {"api", "module"}
            )
            or (
                isinstance(val, ast.Attribute)
                and val.attr == "type"
                and isinstance(val.value, ast.Attribute)
                and val.value.attr == "probe"
                and isinstance(val.value.value, ast.Name)
                and lookup_role(val.value.value.id) == "module"
            )
            or (
                isinstance(val, ast.Attribute)
                and val.attr == "ProbeType"
                and isinstance(val.value, ast.Name)
                and lookup_role(val.value.id) == "module"
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
        if site == "@probe" and kw.arg not in {"checks", "id", "kind", "ignore"}:
            return None, f"Unknown @probe keyword argument: {kw.arg!r}"

    if call.args and "checks" in seen_kws:
        return None, f"Duplicate 'checks' passed positionally and by keyword in {site}"

    id_expr = _extract_kwarg(call, "id")
    if id_expr is None or (isinstance(id_expr, ast.Constant) and id_expr.value is None):
        return None, None
    if not (isinstance(id_expr, ast.Constant) and isinstance(id_expr.value, str)):
        return None, "`id` must be a string literal"
    raw_id = id_expr.value
    if not raw_id.strip():
        return None, "`id` must not be empty"
    if not _VALID_PROBE_ID_RE.fullmatch(raw_id) or ".." in raw_id:
        return (
            None,
            "`id` must be 1-128 chars matching [A-Za-z0-9._:-] without '/' , '\\\\', whitespace, or '..'",
        )
    return raw_id, None


def _is_generator_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if function yields directly (checking nested defs' defaults/decorators too)."""
    stack: list[ast.AST] = list(node.body)
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(cur.decorator_list)
            stack.extend(cur.args.defaults)
            stack.extend(d for d in cur.args.kw_defaults if d is not None)
            if cur.returns is not None:
                stack.append(cur.returns)
            continue
        if isinstance(cur, ast.Lambda):
            stack.extend(cur.args.defaults)
            stack.extend(d for d in cur.args.kw_defaults if d is not None)
            continue
        if isinstance(cur, ast.ClassDef):
            stack.extend(cur.decorator_list)
            stack.extend(cur.bases)
            stack.extend(kw.value for kw in cur.keywords)
            continue
        stack.extend(ast.iter_child_nodes(cur))
    return False


_SIMPLE_TYPE_NAMES: dict[str, str] = {
    "bool": "bool",
    "int": "int",
    "float": "float",
    "str": "str",
    "none": "none",
    "nonetype": "none",
    "list": "list",
    "tuple": "list",
    "set": "list",
    "frozenset": "list",
    "sequence": "list",
    "mutablesequence": "list",
    "iterable": "list",
    "dict": "dict",
    "mapping": "dict",
    "mutablemapping": "dict",
}


def _annotation_head_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _classify_annotation_ast(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "none"
        if isinstance(node.value, str):
            try:
                parsed = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return None
            return _classify_annotation_ast(parsed)
        return None
    head = _annotation_head_name(node)
    if head is not None:
        return _SIMPLE_TYPE_NAMES.get(head.lower())
    if isinstance(node, ast.Subscript):
        base = _annotation_head_name(node.value)
        if base is None:
            return None
        base_low = base.lower()
        if base_low == "optional":
            return _classify_annotation_ast(node.slice)
        if base_low == "annotated":
            sl = node.slice
            first = sl.elts[0] if isinstance(sl, ast.Tuple) and sl.elts else sl
            return _classify_annotation_ast(first)
        if base_low == "literal":
            sl = node.slice
            first = sl.elts[0] if isinstance(sl, ast.Tuple) and sl.elts else sl
            return _classify_value_ast(first)
        if base_low == "union":
            sl = node.slice
            elts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
            tags = [t for t in (_classify_annotation_ast(e) for e in elts) if t is not None]
            non_none = [t for t in tags if t != "none"]
            return non_none[0] if non_none else (tags[0] if tags else None)
        return _SIMPLE_TYPE_NAMES.get(base_low)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _classify_annotation_ast(node.left)
        right = _classify_annotation_ast(node.right)
        for cand in (left, right):
            if cand is not None and cand != "none":
                return cand
        return left or right
    return None


def _classify_value_ast(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        val = node.value
        if val is None:
            return "none"
        if isinstance(val, bool):
            return "bool"
        if isinstance(val, int):
            return "int"
        if isinstance(val, float):
            return "float"
        if isinstance(val, str):
            return "str"
        return None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp)):
        return "list"
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return _SIMPLE_TYPE_NAMES.get(node.func.id.lower())
    return None


def _extract_fn_type_tags(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """Infer lightweight JSON-shape tags ('str'|'int'|'float'|'bool'|'list'|'dict'|'none') for locals/params."""
    tags: dict[str, str] = {}
    args = fn_node.args
    pos_all = [*args.posonlyargs, *args.args]
    pos_defaults: list[ast.expr | None] = [None] * (len(pos_all) - len(args.defaults)) + list(args.defaults)
    for arg, dflt in zip(pos_all, pos_defaults, strict=True):
        t = _classify_annotation_ast(arg.annotation) or _classify_value_ast(dflt)
        if t is not None:
            tags[arg.arg] = t
    for arg, dflt in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        t = _classify_annotation_ast(arg.annotation) or _classify_value_ast(dflt)
        if t is not None:
            tags[arg.arg] = t

    stack: list[ast.AST] = list(fn_node.body)
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(cur, ast.AnnAssign) and isinstance(cur.target, ast.Name):
            t = _classify_annotation_ast(cur.annotation) or _classify_value_ast(cur.value)
            if t is not None:
                tags[cur.target.id] = t
        elif isinstance(cur, ast.Assign):
            t = _classify_value_ast(cur.value)
            if t is not None:
                for tgt in cur.targets:
                    if isinstance(tgt, ast.Name):
                        tags[tgt.id] = t
        stack.extend(ast.iter_child_nodes(cur))
    return tags


def _extract_param_names(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    is_class_body: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (all_param_names, required_param_names) matching runtime inspect.signature order."""
    args = fn_node.args
    pos_all = [*args.posonlyargs, *args.args]
    all_names = [a.arg for a in pos_all]
    if args.vararg is not None:
        all_names.append(args.vararg.arg)
    all_names.extend(a.arg for a in args.kwonlyargs)
    if args.kwarg is not None:
        all_names.append(args.kwarg.arg)

    n_req_pos = max(0, len(pos_all) - len(args.defaults))
    req_pos = [a.arg for a in pos_all[:n_req_pos]]
    is_staticmethod = any(
        isinstance(dec, ast.Name) and dec.id == "staticmethod" for dec in fn_node.decorator_list
    )
    if is_class_body and not is_staticmethod and req_pos and pos_all and req_pos[0] == pos_all[0].arg:
        req_pos = req_pos[1:]

    req_kw = [
        kw_arg.arg
        for kw_arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
        if default is None
    ]
    return tuple(all_names), tuple([*req_pos, *req_kw])


def _format_signature(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(fn_node, ast.AsyncFunctionDef) else "def"
    args_src = ast.unparse(fn_node.args)
    ret_src = f" -> {ast.unparse(fn_node.returns)}" if fn_node.returns is not None else ""
    return f"{prefix} {fn_node.name}({args_src}){ret_src}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _ModuleProbeVisitor(ast.NodeVisitor):
    """Lexically scoped AST visitor collecting @probe decorators and probe.here(...) calls."""

    def __init__(
        self,
        *,
        source: str,
        module_name: str,
        display_path: str,
    ) -> None:
        self.source = source
        self.lines = source.splitlines()
        self.module_name = module_name
        self.display_path = display_path
        self.scope: list[tuple[str, bool]] = []
        self.fn_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self._env_stack: list[dict[str, str]] = [{}]
        self.discovered: list[DiscoveredProbe] = []

    def _bind(self, name: str, role: str) -> None:
        if name:
            self._env_stack[-1][name] = role

    def _lookup_role(self, name: str) -> str | None:
        for env in reversed(self._env_stack):
            if name in env:
                role = env[name]
                return None if role == "shadowed" else role
        return None

    def _shadow_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._bind(target.id, "shadowed")
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._shadow_target(elt)
        elif isinstance(target, ast.Starred):
            self._shadow_target(target.value)

    def _shadow_fn_args(self, args: ast.arguments) -> None:
        for a in (*args.posonlyargs, *args.args, args.vararg, *args.kwonlyargs, args.kwarg):
            if a is not None:
                self._bind(a.arg, "shadowed")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "probe":
                self._bind(alias.asname or "probe", "module")
            elif alias.name.startswith("probe.") and alias.asname is None:
                self._bind("probe", "module")
            else:
                self._bind(alias.asname or alias.name.split(".", 1)[0], "shadowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module == "probe":
            for alias in node.names:
                if alias.name == "probe":
                    self._bind(alias.asname or "probe", "api")
                elif alias.name == "here":
                    self._bind(alias.asname or "here", "here")
                elif alias.name in {"ProbeType", "type"}:
                    self._bind(alias.asname or alias.name, "type")
                elif alias.name == "*":
                    self._bind("probe", "api")
                    self._bind("here", "here")
                    self._bind("ProbeType", "type")
                else:
                    self._bind(alias.asname or alias.name, "shadowed")
        elif node.level == 0 and node.module == "probe.runtime":
            for alias in node.names:
                if alias.name == "ProbeType":
                    self._bind(alias.asname or "ProbeType", "type")
                elif alias.name != "*":
                    self._bind(alias.asname or alias.name, "shadowed")
        else:
            for alias in node.names:
                if alias.name != "*":
                    self._bind(alias.asname or alias.name, "shadowed")

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
            self._shadow_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.annotation)
        if node.value is not None:
            self._shadow_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._shadow_target(node.target)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._shadow_target(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._shadow_target(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
                self._shadow_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
                self._shadow_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._shadow_target(target)

    def _visit_comprehension(self, generators: list[ast.comprehension], tail: list[ast.expr]) -> None:
        if generators:
            self.visit(generators[0].iter)
        self._env_stack.append({})
        try:
            for idx, comp in enumerate(generators):
                if idx > 0:
                    self.visit(comp.iter)
                self._shadow_target(comp.target)
                for if_expr in comp.ifs:
                    self.visit(if_expr)
            for expr in tail:
                self.visit(expr)
        finally:
            self._env_stack.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, [node.key, node.value])

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
        doc = ast.get_docstring(node, clean=True) or ""
        s_hash = _sha256_text(f"{sig}\n{doc}")
        is_class_body = bool(self.scope and not self.scope[-1][1])
        param_names, required_params = _extract_param_names(node, is_class_body=is_class_body)
        fn_type_tags = _extract_fn_type_tags(node)
        param_type_tags = tuple((p, fn_type_tags.get(p, "str")) for p in param_names)
        ret_type_tag = _classify_annotation_ast(node.returns) or "str"

        for dec in node.decorator_list:
            call, call_err, matched = _classify_probe_decorator(dec, self._lookup_role)
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
                        param_names=param_names,
                        required_params=required_params,
                        type_tags=param_type_tags,
                        return_type_tag=ret_type_tag,
                    )
                )
                continue

            checks, check_err = _extract_checks_literal(_extract_kwarg(call, "checks", 0))
            explicit_id, struct_err = _validate_call_structure_and_id(call, site="@probe")
            probe_id = explicit_id or f"{self.module_name}.{qualname}"
            kind, kind_err = _extract_probe_type(
                _extract_kwarg(call, "kind"),
                lookup_role=self._lookup_role,
            )
            ignored_params, ignore_err = _extract_ignore_literal(_extract_kwarg(call, "ignore"))
            if ignore_err is None and ignored_params:
                unknown_ign = [p for p in ignored_params if p not in param_names]
                if unknown_ign:
                    ignore_err = (
                        f"@probe ignore references unknown parameter(s): "
                        f"{', '.join(repr(p) for p in unknown_ign)}"
                    )
            ign_set = frozenset(ignored_params)
            eff_param_names = tuple(p for p in param_names if p not in ign_set) if ign_set else param_names
            eff_req_params = tuple(p for p in required_params if p not in ign_set) if ign_set else required_params
            eff_type_tags = (
                tuple((p, fn_type_tags.get(p, "str")) for p in eff_param_names) if ign_set else param_type_tags
            )
            eff_s_hash = (
                _sha256_text(f"{sig}\n{doc}\nignore={','.join(ignored_params)}")
                if ignored_params
                else s_hash
            )
            literal_error = call_err or struct_err or check_err or kind_err or ignore_err or gen_err
            c_hash = compute_checks_hash(kind, checks)
            combined = _sha256_text(f"{c_hash}:{eff_s_hash}")
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
                    source_hash=eff_s_hash,
                    combined_hash=combined,
                    param_names=eff_param_names,
                    required_params=eff_req_params,
                    ignored_params=ignored_params,
                    type_tags=eff_type_tags,
                    return_type_tag=ret_type_tag,
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

        self._bind(node.name, "shadowed")
        self._env_stack.append({})
        self._shadow_fn_args(node.args)
        self.scope.append((node.name, True))
        self.fn_stack.append(node)
        try:
            for stmt in node.body:
                self.visit(stmt)
        finally:
            self.fn_stack.pop()
            self.scope.pop()
            self._env_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = self._qualname(node.name)
        sig = f"class {node.name}"
        full_source = ast.get_source_segment(self.source, node) or sig
        for dec in node.decorator_list:
            call, _, matched = _classify_probe_decorator(dec, self._lookup_role)
            if not matched:
                continue
            checks: tuple[str, ...] = ()
            explicit_id: str | None = None
            kind = ProbeType.decision
            if call is not None:
                checks, _ = _extract_checks_literal(_extract_kwarg(call, "checks", 0))
                explicit_id, _ = _validate_call_structure_and_id(call, site="@probe")
                kind, _ = _extract_probe_type(_extract_kwarg(call, "kind"), lookup_role=self._lookup_role)
            c_hash = compute_checks_hash(kind, checks)
            s_hash = _sha256_text(f"{sig}\n{full_source}")
            self.discovered.append(
                DiscoveredProbe(
                    probe_id=explicit_id or f"{self.module_name}.{qualname}",
                    site="decorator",
                    probe_type=kind,
                    file_path=self.display_path,
                    lineno=node.lineno,
                    file_line=f"{self.display_path}:{node.lineno}",
                    checks=checks,
                    literal_error="@probe can only decorate functions or methods, not classes",
                    signature=sig,
                    source_excerpt=full_source[:1500],
                    checks_hash=c_hash,
                    source_hash=s_hash,
                    combined_hash=_sha256_text(f"{c_hash}:{s_hash}"),
                )
            )

        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        for kw in node.keywords:
            self.visit(kw)
        for tp in getattr(node, "type_params", ()):
            self.visit(tp)

        self._bind(node.name, "shadowed")
        self._env_stack.append({})
        self.scope.append((node.name, False))
        try:
            for stmt in node.body:
                self.visit(stmt)
        finally:
            self.scope.pop()
            self._env_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults:
            self.visit(default)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                self.visit(kw_default)
        self._env_stack.append({})
        self._shadow_fn_args(node.args)
        self.scope.append(("<lambda>", True))
        try:
            self.visit(node.body)
        finally:
            self.scope.pop()
            self._env_stack.pop()

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        if node.generators:
            self.visit(node.generators[0].iter)
        self._env_stack.append({})
        self.scope.append(("<genexpr>", False))
        try:
            for idx, comp in enumerate(node.generators):
                if idx > 0:
                    self.visit(comp.iter)
                self._shadow_target(comp.target)
                for if_expr in comp.ifs:
                    self.visit(if_expr)
            self.visit(node.elt)
        finally:
            self.scope.pop()
            self._env_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        call = _is_here_call(node, self._lookup_role)
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
                lookup_role=self._lookup_role,
            )
            watched_keys = tuple(
                kw.arg for kw in call.keywords if kw.arg and kw.arg not in {"checks", "id", "kind"}
            )
            watched_err = None if watched_keys else "probe.here requires at least one watched keyword variable"
            literal_error = struct_err or check_err or kind_err or watched_err
            enclosing = self.fn_stack[-1] if self.fn_stack else None
            sig = _format_signature(enclosing) if enclosing else "<module>"
            enclosing_tags = _extract_fn_type_tags(enclosing) if enclosing is not None else {}
            watched_type_pairs: list[tuple[str, str]] = []
            for kw in call.keywords:
                if not kw.arg or kw.arg in {"checks", "id", "kind"}:
                    continue
                t = _classify_value_ast(kw.value)
                if t is None and isinstance(kw.value, ast.Name):
                    t = enclosing_tags.get(kw.value.id)
                watched_type_pairs.append((kw.arg, t or "str"))
            end_idx = min(len(self.lines), (call.end_lineno or call.lineno) + 3)
            if enclosing is not None:
                start_idx = max(enclosing.lineno - 1, call.lineno - 41)
                excerpt = "\n".join(self.lines[start_idx:end_idx])
                enclosing_doc = ast.get_docstring(enclosing, clean=True) or ""
                if enclosing_doc and enclosing_doc not in excerpt:
                    excerpt = f'"""{enclosing_doc}"""\n{excerpt}'
            else:
                start_idx = max(0, call.lineno - 16)
                excerpt = "\n".join(self.lines[start_idx:end_idx])
            excerpt = excerpt[:1500]
            c_hash = compute_checks_hash(kind, checks)
            s_hash = _sha256_text(f"{sig}\n{','.join(watched_keys)}")
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
                    type_tags=tuple(watched_type_pairs),
                )
            )
        self.generic_visit(node)


def _parse_and_discover(
    py_files: Sequence[Path],
    base_root: Path,
) -> tuple[list[DiscoveredProbe], dict[Path, str]]:
    """Parse Python files and discover @probe and probe.here sites."""
    discovered: list[DiscoveredProbe] = []
    parse_errors: dict[Path, str] = {}

    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            parse_errors[py_file] = f"{type(exc).__name__}: {exc}"
            continue

        if not _file_may_use_probe(tree):
            continue

        module_name = _module_name_for_file(py_file, base_root)
        display_path = _rel_display_path(py_file, base_root)

        visitor = _ModuleProbeVisitor(
            source=source,
            module_name=module_name,
            display_path=display_path,
        )
        visitor.visit(tree)
        discovered.extend(visitor.discovered)

    return discovered, parse_errors


def _scan_files(
    paths: Sequence[Path] = (),
    *,
    root: Path | None = None,
) -> tuple[list[Path], list[DiscoveredProbe], dict[Path, str]]:
    """Scan Python files via AST, returning (scanned_files, discovered_probes, parse_errors)."""
    base_root = (root or Path.cwd()).resolve()
    py_files, _, _, _ = _resolve_scan_targets(paths, base_root)
    discovered, parse_errors = _parse_and_discover(py_files, base_root)
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


def _normalize_fixture_cases(raw_val: Any, *, fp: Path, field_name: str) -> tuple[str, ...]:
    """Normalize a manual fixture section into a non-empty tuple of JSON snapshot strings."""
    if isinstance(raw_val, list):
        if not raw_val:
            raise BackendError(f"Fixture {fp} '{field_name}' list must contain at least one snapshot")
        out: list[str] = []
        for idx, elem in enumerate(raw_val):
            if isinstance(elem, dict):
                out.append(json.dumps(elem, ensure_ascii=False))
            elif isinstance(elem, str):
                out.append(elem)
            else:
                raise BackendError(
                    f"Fixture {fp} '{field_name}[{idx}]' must be a JSON snapshot object or JSON string"
                )
        return tuple(out)
    if isinstance(raw_val, dict):
        return (json.dumps(raw_val, ensure_ascii=False),)
    if isinstance(raw_val, str):
        return (raw_val,)
    raise BackendError(
        f"Fixture {fp} '{field_name}' must be a JSON snapshot object, non-empty list of snapshots, or JSON string"
    )


def _fixture_candidates(fixtures_dir: Path, probe_id: str) -> tuple[Path, list[Path], str, str]:
    """Validate probe_id and return (resolved_root, candidate_paths, safe_id, hashed_id)."""
    if (
        not probe_id
        or "/" in probe_id
        or "\\" in probe_id
        or ".." in probe_id
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in probe_id)
    ):
        raise BackendError(
            f"Invalid fixture probe_id {probe_id!r}: path separators, control chars, and '..' are not allowed"
        )
    fixtures_root = fixtures_dir.resolve()
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", probe_id)
    hashed_id = f"{safe_id}-{_sha256_text(probe_id)[:6]}"
    names = list(dict.fromkeys((f"{hashed_id}.json", f"{probe_id}.json", f"{safe_id}.json")))
    paths: list[Path] = []
    for name in names:
        fp = fixtures_dir / name
        if not fp.resolve().is_relative_to(fixtures_root):
            raise BackendError(f"Fixture path {fp} escapes {fixtures_dir}")
        paths.append(fp)
    return fixtures_root, paths, safe_id, hashed_id


def resolve_fixture_path(fixtures_dir: Path, probe_id: str) -> Path:
    """Return existing manual fixture path for probe_id or the canonical path for creating one."""
    fixtures_root, candidates, safe_id, hashed_id = _fixture_candidates(fixtures_dir, probe_id)
    for fp in candidates:
        if fp.exists() and fp.is_file():
            return fp
    target_name = f"{probe_id}.json" if safe_id == probe_id else f"{hashed_id}.json"
    target = fixtures_dir / target_name
    if not target.resolve().is_relative_to(fixtures_root):
        raise BackendError(f"Fixture path {target} escapes {fixtures_dir}")
    return target


def _placeholder_for_tag(type_tag: str, tag: str, name: str) -> Any:
    if type_tag == "none":
        return None
    if type_tag == "bool":
        return tag == "PASS"
    if type_tag == "int":
        return 1 if tag == "PASS" else 0
    if type_tag == "float":
        return 1.0 if tag == "PASS" else 0.0
    if type_tag == "list":
        return [f"<{tag}:{name}>"]
    if type_tag == "dict":
        return {f"<{tag}:{name}>": "..."}
    return f"<{tag}:{name}>"


def build_skeleton_snapshot(item: DiscoveredProbe, tag: str) -> dict[str, Any]:
    """Build a starter fixture snapshot matching item's AST signature, type hints, or watched keys."""
    tag_map = dict(item.type_tags)
    if item.site == "here":
        return {
            "watched": {
                k: _placeholder_for_tag(tag_map.get(k, "str"), tag, k)
                for k in item.watched_keys
            }
        }
    return {
        "inputs": {
            p: _placeholder_for_tag(tag_map.get(p, "str"), tag, p)
            for p in item.param_names
        },
        "output": _placeholder_for_tag(item.return_type_tag, tag, "output"),
    }


def _parse_existing_fixture_section(raw_val: Any, *, fp: Path, field_name: str) -> list[dict[str, Any]]:
    if raw_val is None:
        return []
    raw_items = raw_val if isinstance(raw_val, list) else [raw_val]
    out: list[dict[str, Any]] = []
    for idx, elem in enumerate(raw_items):
        if isinstance(elem, str):
            try:
                elem = json.loads(elem)
            except json.JSONDecodeError as exc:
                raise BackendError(f"Invalid JSON in {fp} '{field_name}[{idx}]': {exc}") from exc
        if not isinstance(elem, dict):
            raise BackendError(f"Fixture {fp} '{field_name}[{idx}]' must be a JSON object")
        out.append(elem)
    return out


def upsert_probe_fixture(
    *,
    probe_id: str,
    cfg: ProbeConfig,
    run_id: str | None = None,
    init_skeleton: bool = False,
    max_per_side: int = 5,
) -> tuple[Path, int, int]:
    """Promote recorded pass/fail snapshots (and optionally AST skeletons) into .probes/fixtures/<id>.json."""
    fixtures_dir = cfg.resolve_lock_path().parent / "fixtures"
    fp = resolve_fixture_path(fixtures_dir, probe_id)

    pass_list: list[dict[str, Any]] = []
    fail_list: list[dict[str, Any]] = []
    if fp.is_file():
        try:
            existing = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(f"Failed to read existing fixture {fp}: {exc}") from exc
        if not isinstance(existing, dict):
            raise BackendError(f"Fixture {fp} must contain a JSON object")
        if "probe_id" in existing and existing["probe_id"] != probe_id:
            raise BackendError(f"Fixture {fp} probe_id {existing['probe_id']!r} does not match {probe_id!r}")
        pass_list = _parse_existing_fixture_section(
            existing.get("pass", existing.get("passing_snapshot")),
            fp=fp,
            field_name="pass",
        )
        fail_list = _parse_existing_fixture_section(
            existing.get("fail", existing.get("failing_snapshot")),
            fp=fp,
            field_name="fail",
        )

    def _snap_key(obj: dict[str, Any]) -> str:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)

    seen_pass = {_snap_key(x) for x in pass_list}
    seen_fail = {_snap_key(x) for x in fail_list}

    runs_dir = cfg.resolve_runs_dir()
    if runs_dir.is_dir():
        store = ProbeStore(runs_dir, create=False)
        try:
            if run_id is not None:
                run = store.get_run(run_id)
                if run is None:
                    raise BackendError(f"Run {run_id!r} not found in {runs_dir}")
                target_runs = [run]
            else:
                target_runs = store.list_runs(limit=50)

            added_pass = 0
            added_fail = 0
            for r in target_runs:
                if added_pass >= max_per_side and added_fail >= max_per_side:
                    break
                for ev in store.list_events(r.id):
                    if ev.probe_id != probe_id or ev.verdict not in {"pass", "fail"}:
                        continue
                    try:
                        snap = json.loads(ev.payload_json)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(snap, dict) or "exception" in snap:
                        continue
                    key = _snap_key(snap)
                    if ev.verdict == "pass" and added_pass < max_per_side and key not in seen_pass:
                        pass_list.append(snap)
                        seen_pass.add(key)
                        added_pass += 1
                    elif ev.verdict == "fail" and added_fail < max_per_side and key not in seen_fail:
                        fail_list.append(snap)
                        seen_fail.add(key)
                        added_fail += 1
        finally:
            store.close()
    elif run_id is not None:
        raise BackendError(f"Run {run_id!r} not found: {runs_dir} does not exist")

    if init_skeleton and (not pass_list or not fail_list):
        discovered = discover_probes(root=cfg.project_root)
        item = next((p for p in discovered if p.probe_id == probe_id), None)
        if item is None:
            raise BackendError(f"Probe {probe_id!r} not found in Python files under {cfg.project_root}")
        if not pass_list:
            pass_list.append(build_skeleton_snapshot(item, "PASS"))
        if not fail_list:
            fail_list.append(build_skeleton_snapshot(item, "FAIL"))

    if not pass_list or not fail_list:
        raise BackendError(
            f"Fixture for {probe_id!r} requires at least 1 pass and 1 fail snapshot "
            f"(found {len(pass_list)} pass, {len(fail_list)} fail); record both verdicts or pass --init"
        )

    fp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe_id": probe_id,
        "pass": pass_list,
        "fail": fail_list,
    }
    fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return fp, len(pass_list), len(fail_list)


def _load_manual_fixture(fixtures_dir: Path, probe_id: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    _, candidates, _, _ = _fixture_candidates(fixtures_dir, probe_id)
    for fp in candidates:
        if not fp.exists() or not fp.is_file():
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(f"Failed to read fixture {fp}: {exc}") from exc
        if not isinstance(data, dict):
            raise BackendError(f"Fixture {fp} must contain a JSON object with 'pass' and 'fail' keys")
        if "probe_id" in data and data["probe_id"] != probe_id:
            raise BackendError(
                f"Fixture {fp} probe_id {data['probe_id']!r} does not match {probe_id!r}"
            )
        passing = data.get("pass", data.get("passing_snapshot"))
        failing = data.get("fail", data.get("failing_snapshot"))
        if passing is None or failing is None:
            raise BackendError(f"Fixture {fp} must define both 'pass' and 'fail' snapshots")
        return (
            _normalize_fixture_cases(passing, fp=fp, field_name="pass"),
            _normalize_fixture_cases(failing, fp=fp, field_name="fail"),
        )
    return None


def compute_effective_combined_hash(
    item: DiscoveredProbe,
    *,
    fixtures_dir: Path,
    pass_threshold: float,
    domain_context: str = "",
) -> tuple[str, tuple[tuple[str, ...], tuple[str, ...]] | None, str | None]:
    """Compute effective lock hash folding combined_hash, pass_threshold, domain_context, and any manual fixture."""
    try:
        fixture = _load_manual_fixture(fixtures_dir, item.probe_id)
    except BackendError as exc:
        return "", None, str(exc)

    threshold_key = f"{float(pass_threshold)!r}"
    base = f"{item.combined_hash}:threshold={threshold_key}"
    ctx = domain_context.strip()
    if ctx:
        base = f"{base}:ctx={_sha256_text(ctx)}"
    if fixture is not None:
        fix_payload = json.dumps(
            {"pass": list(fixture[0]), "fail": list(fixture[1])},
            ensure_ascii=False,
        )
        fix_digest = _sha256_text(fix_payload)
        effective_hash = _sha256_text(f"{base}:fixture:{fix_digest}")
    else:
        effective_hash = _sha256_text(base)
    return effective_hash, fixture, None


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


def _matches_dir_prefixes(file_line: str, dir_prefixes: set[str]) -> bool:
    return any(file_line.startswith(f"{dp}/") for dp in dir_prefixes)


def _matches_file_or_dir_prefixes(file_line: str, prefixes: set[str]) -> bool:
    return any(
        file_line == fp or file_line.startswith(f"{fp}:") or file_line.startswith(f"{fp}/")
        for fp in prefixes
    )


def _file_from_file_line(file_line: str) -> str:
    if ":" in file_line:
        head, tail = file_line.rsplit(":", 1)
        if tail.isdigit():
            return head
    return file_line


def _validate_gate3_snapshot(snapshot_json: str, *, item: DiscoveredProbe, label: str) -> str:
    """Validate Gate 3 pass/fail snapshot JSON matches the runtime site schema."""
    try:
        obj = json.loads(snapshot_json)
    except ValueError as exc:
        raise BackendError(f"Gate 3 {label} for {item.probe_id} is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise BackendError(f"Gate 3 {label} for {item.probe_id} must be a JSON object")

    if item.site == "decorator":
        if "exception" in obj:
            raise BackendError(
                f"Gate 3 {label} for decorator probe {item.probe_id} must not use 'exception'; provide 'output'"
            )
        if "watched" in obj:
            raise BackendError(
                f"Gate 3 {label} for decorator probe {item.probe_id} must not include 'watched'"
            )
        inputs = obj.get("inputs")
        if not isinstance(inputs, dict):
            raise BackendError(
                f"Gate 3 {label} for decorator probe {item.probe_id} must include dict 'inputs'"
            )
        missing_params = [p for p in item.required_params if p not in inputs]
        if missing_params:
            raise BackendError(
                f"Gate 3 {label} for decorator probe {item.probe_id} missing required input(s): "
                f"{', '.join(missing_params)}"
            )
        if item.ignored_params:
            unexpected_ignored = [p for p in item.ignored_params if p in inputs]
            if unexpected_ignored:
                raise BackendError(
                    f"Gate 3 {label} for decorator probe {item.probe_id} must omit ignored input(s): "
                    f"{', '.join(unexpected_ignored)}"
                )
        if "output" not in obj:
            raise BackendError(
                f"Gate 3 {label} for decorator probe {item.probe_id} must include 'output'"
            )
    elif item.site == "here":
        for forbidden in ("inputs", "output", "exception"):
            if forbidden in obj:
                raise BackendError(
                    f"Gate 3 {label} for here probe {item.probe_id} must not include {forbidden!r}"
                )
        watched = obj.get("watched")
        if not isinstance(watched, dict):
            raise BackendError(
                f"Gate 3 {label} for here probe {item.probe_id} must include dict 'watched'"
            )
        if item.watched_keys:
            missing = [k for k in item.watched_keys if k not in watched]
            extra = sorted(k for k in watched if k not in item.watched_keys)
            if missing:
                raise BackendError(
                    f"Gate 3 {label} for {item.probe_id} missing watched key(s): {', '.join(missing)}"
                )
            if extra:
                raise BackendError(
                    f"Gate 3 {label} for {item.probe_id} has unexpected watched key(s): {', '.join(extra)}"
                )
    return snapshot_json


def _format_case_scores(case_scores: list[list[CheckScore]]) -> list[dict[str, Any]]:
    """Serialize per-case CheckScore lists for ProbeCheckResult."""
    multi = len(case_scores) > 1
    out: list[dict[str, Any]] = []
    for idx, scores in enumerate(case_scores):
        for s in scores:
            entry: dict[str, Any] = {"check": s.check, "probability": s.probability, "passed": s.passed}
            if multi:
                entry["case"] = idx
            out.append(entry)
    return out


def _format_gate3_pass_reason(
    fixture_source: str,
    pass_case_scores: list[list[CheckScore]],
    fail_case_scores: list[list[CheckScore]],
) -> tuple[str, float, float]:
    """Summarize Gate 2 + Gate 3 calibration margins and return (reason, pass_min_p, fail_max_p)."""
    pass_min_p = min(s.probability for scores in pass_case_scores for s in scores)
    fail_max_p = max(min(s.probability for s in scores) for scores in fail_case_scores)
    if len(pass_case_scores) == 1 and len(fail_case_scores) == 1:
        reason = (
            f"Passed Gate 2 audit & Gate 3 {fixture_source} calibration "
            f"(pass min p={pass_min_p:.2f}, fail max p={fail_max_p:.2f})"
        )
    else:
        reason = (
            f"Passed Gate 2 audit & Gate 3 {fixture_source} calibration "
            f"({len(pass_case_scores)} pass, {len(fail_case_scores)} fail; "
            f"pass min p={pass_min_p:.2f}, fail max p={fail_max_p:.2f})"
        )
    return reason, round(pass_min_p, 4), round(fail_max_p, 4)


def _evaluate_uncached_probe(
    item: DiscoveredProbe,
    *,
    effective_combined_hash: str,
    fixture: tuple[tuple[str, ...], tuple[str, ...]] | None,
    cfg: ProbeConfig,
    decision_backend: DecisionBackend,
    text_backend: TextBackend,
    redactor: SecretRedactor,
) -> tuple[ProbeCheckResult, dict[str, Any] | None]:
    """Run Gates 2-3 for one uncached probe and return (result, lock_entry_or_none)."""
    # Gate 2: Factual analysis via [models.text] + noul approval via [models.decision] (1x retry on disagreement).
    gate2_expected = get_noul_expected("check_gate2_nouls")

    def _gate2_noul_ok(score: CheckScore) -> bool:
        return score.probability > 0.50 if gate2_expected else score.probability < 0.50

    gate2_analysis = ""
    gate2_scores: list[CheckScore] = []
    approved = False
    for g2_attempt in (1, 2):
        try:
            gate2_analysis = text_backend.analyze_checks_gate2(
                probe_id=item.probe_id,
                site=item.site,
                checks=item.checks,
                signature=item.signature,
                source_excerpt=item.source_excerpt,
                watched_keys=item.watched_keys,
                ignored_params=item.ignored_params,
                captured_params=item.param_names,
                temperature=0.0 if g2_attempt == 1 else 0.2,
            )
            gate2_state_obj: dict[str, Any] = {
                "site": item.site,
                "signature": item.signature,
                "watched_keys": list(item.watched_keys),
                "checks": list(item.checks),
            }
            if item.site == "decorator":
                gate2_state_obj["captured_inputs"] = list(item.param_names)
                if item.ignored_params:
                    gate2_state_obj["ignored_params"] = list(item.ignored_params)
            gate2_state = json.dumps(gate2_state_obj, ensure_ascii=False)
            gate2_scores = decision_backend.evaluate_checks(
                probe_id=f"{item.probe_id}#gate2",
                checks=get_nouls("check_gate2_nouls"),
                snapshot_json=gate2_state,
                analysis=gate2_analysis,
            )
        except BackendError as exc:
            return (
                ProbeCheckResult(
                    probe_id=item.probe_id,
                    file_line=item.file_line,
                    site=item.site,
                    probe_type=item.probe_type.value,
                    status="failed",
                    gate="gate2",
                    reason=redactor.scrub_text(str(exc)),
                    checks=item.checks,
                ),
                None,
            )

        approved = bool(gate2_scores) and all(_gate2_noul_ok(s) for s in gate2_scores)
        if approved:
            break

    if not approved:
        failed_nouls = [s.check for s in gate2_scores if not _gate2_noul_ok(s)]
        audit_reason = (
            f"{gate2_analysis} (failed: {'; '.join(failed_nouls)})".strip()
            if gate2_analysis
            else "; ".join(failed_nouls)
        )
        return (
            ProbeCheckResult(
                probe_id=item.probe_id,
                file_line=item.file_line,
                site=item.site,
                probe_type=item.probe_type.value,
                status="failed",
                gate="gate2",
                reason=redactor.scrub_text(audit_reason or "Rejected by Gate 2 check audit"),
                checks=item.checks,
            ),
            None,
        )

    # Gate 3: Dry-run calibration against manual fixture(s) or synthesized samples (1x synthetic retry).
    fixture_source = "fixture" if fixture is not None else "synthetic"
    max_attempts = 1 if fixture is not None else 2
    retry_feedback = ""
    pass_case_scores: list[list[CheckScore]] = []
    fail_case_scores: list[list[CheckScore]] = []
    bad_pass_indices: list[int] = []
    bad_fail_indices: list[int] = []
    pass_ok = False
    fail_ok = False
    gate3_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            if fixture is not None:
                raw_pass_cases, raw_fail_cases = fixture
            else:
                p_snap, f_snap = text_backend.generate_calibration_fixtures_gate3(
                    probe_id=item.probe_id,
                    site=item.site,
                    checks=item.checks,
                    signature=item.signature,
                    source_excerpt=item.source_excerpt,
                    watched_keys=item.watched_keys,
                    ignored_params=item.ignored_params,
                    captured_params=item.param_names,
                    temperature=0.0 if attempt == 1 else 0.3,
                    retry_feedback=retry_feedback,
                )
                raw_pass_cases, raw_fail_cases = (p_snap,), (f_snap,)
        except BackendError as exc:
            if fixture is None and attempt == 1 and isinstance(exc, ModelTruncatedError):
                retry_feedback = (
                    "Previous synthetic fixture JSON truncated (finish_reason='length'); "
                    "keep field values concise while satisfying every check."
                )
                continue
            err_msg = str(exc)
            is_shape_err = (
                "Gate 3 synthesis missing" in err_msg or "Expected JSON object" in err_msg
            )
            if fixture is None and attempt == 1 and is_shape_err:
                retry_feedback = err_msg
                continue
            gate3_error = err_msg
            break

        try:
            pass_snaps = [
                _validate_gate3_snapshot(
                    snap,
                    item=item,
                    label="passing_snapshot" if len(raw_pass_cases) == 1 else f"passing_snapshot[{idx}]",
                )
                for idx, snap in enumerate(raw_pass_cases)
            ]
            fail_snaps = [
                _validate_gate3_snapshot(
                    snap,
                    item=item,
                    label="failing_snapshot" if len(raw_fail_cases) == 1 else f"failing_snapshot[{idx}]",
                )
                for idx, snap in enumerate(raw_fail_cases)
            ]
        except BackendError as exc:
            if fixture is None and attempt == 1:
                retry_feedback = str(exc)
                continue
            gate3_error = str(exc)
            break

        try:
            pass_case_scores = [
                _evaluate_sample(
                    item=item,
                    snapshot_json=snap,
                    decision_backend=decision_backend,
                    text_backend=text_backend,
                )
                for snap in pass_snaps
            ]
            fail_case_scores = [
                _evaluate_sample(
                    item=item,
                    snapshot_json=snap,
                    decision_backend=decision_backend,
                    text_backend=text_backend,
                )
                for snap in fail_snaps
            ]
        except BackendError as exc:
            gate3_error = str(exc)
            break

        bad_pass_indices = [
            idx
            for idx, scores in enumerate(pass_case_scores)
            if not (scores and all(s.passed for s in scores))
        ]
        bad_fail_indices = [
            idx
            for idx, scores in enumerate(fail_case_scores)
            if not (scores and any(not s.passed for s in scores))
        ]
        pass_ok = bool(pass_case_scores) and not bad_pass_indices
        fail_ok = bool(fail_case_scores) and not bad_fail_indices
        if pass_ok and fail_ok:
            break
        if fixture is None and attempt == 1:
            fb_parts: list[str] = []
            if not pass_ok:
                first_pass_scores = pass_case_scores[0] if pass_case_scores else []
                failed_pass = [
                    f"{s.check!r} (p={s.probability:.2f})"
                    for s in first_pass_scores
                    if not s.passed
                ]
                fb_parts.append(
                    f"passing_snapshot failed checks: {', '.join(failed_pass) or 'empty scores'}"
                )
            if not fail_ok:
                fb_parts.append("failing_snapshot did not fail any check")
            retry_feedback = "; ".join(fb_parts)
            continue

    if gate3_error is not None:
        return (
            ProbeCheckResult(
                probe_id=item.probe_id,
                file_line=item.file_line,
                site=item.site,
                probe_type=item.probe_type.value,
                status="failed",
                gate="gate3",
                reason=redactor.scrub_text(gate3_error),
                checks=item.checks,
            ),
            None,
        )

    pass_dicts = _format_case_scores(pass_case_scores)
    fail_dicts = _format_case_scores(fail_case_scores)

    if not (pass_ok and fail_ok):
        parts = []
        if not pass_ok:
            if len(pass_case_scores) == 1:
                parts.append("passing fixture did not pass all checks")
            else:
                parts.append(f"passing fixture case(s) {bad_pass_indices} did not pass all checks")
        if not fail_ok:
            if len(fail_case_scores) == 1:
                parts.append("failing fixture did not fail any check")
            else:
                parts.append(f"failing fixture case(s) {bad_fail_indices} did not fail any check")
        return (
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
            ),
            None,
        )

    passed_reason, pass_min_p, fail_max_p = _format_gate3_pass_reason(
        fixture_source, pass_case_scores, fail_case_scores
    )
    lock_entry: dict[str, Any] = {
        "probe_id": item.probe_id,
        "site": item.site,
        "kind": item.probe_type.value,
        "file_line": item.file_line,
        "checks": list(item.checks),
        "watched_keys": list(item.watched_keys),
        "param_names": list(item.param_names),
        "checks_hash": item.checks_hash,
        "source_hash": item.source_hash,
        "combined_hash": effective_combined_hash,
        "source": fixture_source,
        "pass_min_p": pass_min_p,
        "fail_max_p": fail_max_p,
        "approved_at": utc_now_iso(),
    }
    if item.ignored_params:
        lock_entry["ignored_params"] = list(item.ignored_params)
    return (
        ProbeCheckResult(
            probe_id=item.probe_id,
            file_line=item.file_line,
            site=item.site,
            probe_type=item.probe_type.value,
            status="passed",
            gate="done",
            reason=passed_reason,
            checks=item.checks,
            pass_scores=pass_dicts,
            fail_scores=fail_dicts,
        ),
        lock_entry,
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
    base_root = root.resolve()
    py_files, checked_dir_prefixes, explicit_deleted_prefixes, full_root_scan = _resolve_scan_targets(
        paths, base_root
    )
    discovered, parse_errors = _parse_and_discover(py_files, base_root)

    lock_path = cfg.resolve_lock_path()
    lock_data = _load_lockfile(lock_path)
    locked_probes: dict[str, Any] = dict(lock_data.get("probes", {}))
    fixtures_dir = lock_path.parent / "fixtures"
    redactor = SecretRedactor.from_environment(
        cfg.redact,
        cwd=base_root,
        extra_dirs=(*cfg.models_decision.env_search_dirs, Path.cwd()),
    )

    error_file_prefixes = {_rel_display_path(pf, base_root) for pf in parse_errors}
    checked_prefixes = {_rel_display_path(pf, base_root) for pf in py_files} - error_file_prefixes
    discovered_ids = {p.probe_id for p in discovered}

    protected_locked_ids: set[str] = set()
    for pid, meta in locked_probes.items():
        if not isinstance(meta, dict):
            continue
        meta_fl = str(meta.get("file_line", ""))
        if meta_fl and _matches_file_prefixes(meta_fl, error_file_prefixes):
            protected_locked_ids.add(pid)

    unscanned_locked_ids: set[str] = set()
    if not full_root_scan:
        lazy_cache: dict[Path, tuple[bool, set[str]]] = {}
        for pid, meta in locked_probes.items():
            if pid not in discovered_ids or pid in protected_locked_ids or not isinstance(meta, dict):
                continue
            meta_fl = str(meta.get("file_line", ""))
            if (
                not meta_fl
                or _matches_file_prefixes(meta_fl, checked_prefixes)
                or _matches_dir_prefixes(meta_fl, checked_dir_prefixes)
                or _matches_file_or_dir_prefixes(meta_fl, explicit_deleted_prefixes)
            ):
                continue
            meta_file = _file_from_file_line(meta_fl)
            candidate = (Path(meta_file) if Path(meta_file).is_absolute() else (base_root / meta_file)).resolve()
            if not candidate.exists() or not candidate.is_file():
                continue
            if candidate not in lazy_cache:
                cand_disc, cand_errs = _parse_and_discover([candidate], base_root)
                lazy_cache[candidate] = (not cand_errs, {p.probe_id for p in cand_disc})
            parsed_ok, cand_ids = lazy_cache[candidate]
            if not parsed_ok:
                protected_locked_ids.add(pid)
            elif pid in cand_ids:
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
            reason=redactor.scrub_text(err),
        )
        for pf, err in sorted(parse_errors.items())
    ]

    decision_backend: DecisionBackend | None = None
    text_backend: TextBackend | None = None
    try:
        decision_backend = DecisionBackend(
            cfg.models_decision,
            cfg.decision,
            domain_context=cfg.domain_context,
            transport=transport,
        )
        text_backend = TextBackend(
            cfg.models_text,
            cfg.text,
            domain_context=cfg.domain_context,
            transport=transport,
        )

        discovered_results: list[ProbeCheckResult | None] = [None] * len(discovered)
        uncached_tasks: list[tuple[int, DiscoveredProbe, str, tuple[str, str] | None]] = []

        for idx, item in enumerate(discovered):
            # Gate 1: Static literals, budgets, duplicate IDs.
            gate1_err: str | None = None
            if item.literal_error:
                gate1_err = item.literal_error
            elif item.probe_id in protected_locked_ids or item.probe_id in unscanned_locked_ids:
                gate1_err = f"Duplicate probe_id '{item.probe_id}' already locked in another file"
            elif id_counts[item.probe_id] > 1:
                gate1_err = f"Duplicate probe_id '{item.probe_id}' ({id_counts[item.probe_id]} sites)"
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
                if item.probe_id not in protected_locked_ids and item.probe_id not in unscanned_locked_ids:
                    locked_probes.pop(item.probe_id, None)
                discovered_results[idx] = ProbeCheckResult(
                    probe_id=item.probe_id,
                    file_line=item.file_line,
                    site=item.site,
                    probe_type=item.probe_type.value,
                    status="failed",
                    gate="gate1",
                    reason=redactor.scrub_text(gate1_err),
                    checks=item.checks,
                )
                continue

            effective_combined_hash, fixture, fixture_error = compute_effective_combined_hash(
                item,
                fixtures_dir=fixtures_dir,
                pass_threshold=cfg.decision.pass_threshold,
                domain_context=cfg.domain_context,
            )
            if fixture_error is not None:
                locked_probes.pop(item.probe_id, None)
                discovered_results[idx] = ProbeCheckResult(
                    probe_id=item.probe_id,
                    file_line=item.file_line,
                    site=item.site,
                    probe_type=item.probe_type.value,
                    status="failed",
                    gate="gate3",
                    reason=redactor.scrub_text(fixture_error),
                    checks=item.checks,
                )
                continue

            existing = locked_probes.get(item.probe_id)
            watched_ok = item.site != "here" or (
                isinstance(existing, dict) and existing.get("watched_keys") == list(item.watched_keys)
            )
            params_ok = item.site != "decorator" or (
                isinstance(existing, dict) and existing.get("param_names") == list(item.param_names)
            )
            if (
                not force
                and isinstance(existing, dict)
                and existing.get("combined_hash") == effective_combined_hash
                and existing.get("site") == item.site
                and existing.get("kind") == item.probe_type.value
                and watched_ok
                and params_ok
            ):
                existing["file_line"] = item.file_line
                pass_p = existing.get("pass_min_p")
                fail_p = existing.get("fail_max_p")
                if isinstance(pass_p, (int, float)) and isinstance(fail_p, (int, float)):
                    src_label = str(existing.get("source", "locked"))
                    cached_reason = (
                        f"Unchanged since last approved lock entry "
                        f"({src_label}: pass min p={float(pass_p):.2f}, fail max p={float(fail_p):.2f})"
                    )
                else:
                    cached_reason = "Unchanged since last approved lock entry"
                discovered_results[idx] = ProbeCheckResult(
                    probe_id=item.probe_id,
                    file_line=item.file_line,
                    site=item.site,
                    probe_type=item.probe_type.value,
                    status="cached",
                    gate="cache",
                    reason=cached_reason,
                    checks=item.checks,
                )
                continue

            uncached_tasks.append((idx, item, effective_combined_hash, fixture))

        if len(uncached_tasks) <= 1 or cfg.check.max_workers <= 1:
            for idx, item, eff_hash, fixture in uncached_tasks:
                res, lock_entry = _evaluate_uncached_probe(
                    item,
                    effective_combined_hash=eff_hash,
                    fixture=fixture,
                    cfg=cfg,
                    decision_backend=decision_backend,
                    text_backend=text_backend,
                    redactor=redactor,
                )
                discovered_results[idx] = res
                if lock_entry is not None:
                    locked_probes[item.probe_id] = lock_entry
                else:
                    locked_probes.pop(item.probe_id, None)
        else:
            workers = min(cfg.check.max_workers, len(uncached_tasks))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = [
                    (
                        idx,
                        item,
                        pool.submit(
                            _evaluate_uncached_probe,
                            item,
                            effective_combined_hash=eff_hash,
                            fixture=fixture,
                            cfg=cfg,
                            decision_backend=decision_backend,
                            text_backend=text_backend,
                            redactor=redactor,
                        ),
                    )
                    for idx, item, eff_hash, fixture in uncached_tasks
                ]
                for idx, item, fut in future_map:
                    res, lock_entry = fut.result()
                    discovered_results[idx] = res
                    if lock_entry is not None:
                        locked_probes[item.probe_id] = lock_entry
                    else:
                        locked_probes.pop(item.probe_id, None)

        results.extend(r for r in discovered_results if r is not None)
    except BackendError as exc:
        raise BackendError(redactor.scrub_text(str(exc))) from exc
    finally:
        if decision_backend is not None:
            decision_backend.close()
        if text_backend is not None:
            text_backend.close()

    # Remove stale entries for checked files/dirs and explicitly passed deleted targets (preserving parse errors).
    if full_root_scan:
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
                    or (
                        not _matches_file_prefixes(str(meta.get("file_line", "")), checked_prefixes)
                        and not _matches_dir_prefixes(str(meta.get("file_line", "")), checked_dir_prefixes)
                        and not _matches_file_or_dir_prefixes(
                            str(meta.get("file_line", "")), explicit_deleted_prefixes
                        )
                    )
                )
            )
        }

    lock_data = {"version": 1, "probes": locked_probes}
    _save_lockfile(lock_path, lock_data)

    all_ok = all(r.status in {"passed", "cached"} for r in results)
    return all_ok, results
