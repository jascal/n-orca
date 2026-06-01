"""Verification pipeline for N-Orca architectures.

Five stages, in order:

1. Naming      — every flow edge endpoint and tensor reference resolves
2. Structural  — DAG, reachability, every layer reaches an output
3. Shape       — input shapes to each layer match its op's input rule;
                  declared shapes match inferred
4. Resource    — `param_count` / `flops` / `depth` / `output_shape` /
                  `vram_estimate` invariants
5. Op          — every layer has a known op (or warning for unknown)
6. Runtime     — informational: can the Unsloth runtime backend load + fine-tune
                  this, and roughly what does a QLoRA fine-tune cost? (Populates
                  `report.runtime`; never fails on its own — backend coverage is
                  reported, not required.)

A failure in an earlier stage prevents later stages from running on that
architecture, since their preconditions would not hold. Stages 5 and 6 have no
preconditions and always run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from n_orca.ast import Architecture, Layer
from n_orca.backends import capability
from n_orca.ops import get_op, op_names, UnknownOpError, ShapeRuleError


@dataclass
class VerificationError:
    code: str
    message: str
    suggestion: str | None = None
    severity: str = "error"   # "error" | "warning"


@dataclass
class VerificationWarning(VerificationError):
    severity: str = "warning"


@dataclass
class VerificationReport:
    architecture: str
    valid: bool = True
    errors: list[VerificationError] = field(default_factory=list)
    warnings: list[VerificationError] = field(default_factory=list)
    inferred_shapes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    param_count: int = 0
    depth: int = 0
    runtime: dict[str, Any] = field(default_factory=dict)

    def add_error(self, code: str, message: str, suggestion: str | None = None) -> None:
        self.errors.append(VerificationError(code=code, message=message, suggestion=suggestion))
        self.valid = False

    def add_warning(self, code: str, message: str, suggestion: str | None = None) -> None:
        self.warnings.append(
            VerificationError(code=code, message=message, suggestion=suggestion, severity="warning")
        )

    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "valid": self.valid,
            "errors": [e.__dict__ for e in self.errors],
            "warnings": [w.__dict__ for w in self.warnings],
            "param_count": self.param_count,
            "depth": self.depth,
            "inferred_shapes": {k: list(v) for k, v in self.inferred_shapes.items()},
            "runtime": self.runtime,
        }


def verify(arch: Architecture, *, strict: bool = False) -> VerificationReport:
    """Run the full pipeline against `arch` and return a report."""
    report = VerificationReport(architecture=arch.name)

    # Stage 1 — Naming.
    if _stage_naming(arch, report):
        # Stage 2 — Structural (requires Naming to have passed).
        if _stage_structural(arch, report):
            # Stage 3 — Shape (requires Structural).
            _stage_shape(arch, report)
            # Stage 4 — Resource bounds (uses any inferred values we have).
            _stage_resources(arch, report)
    # Stage 5 — Op coverage warnings always run, no dependency.
    _stage_ops(arch, report)
    # Stage 6 — Runtime backend coverage + QLoRA VRAM estimate (informational).
    _stage_runtime(arch, report)

    if strict and report.warnings:
        for w in report.warnings:
            report.errors.append(w)
        report.warnings.clear()
        report.valid = False

    return report


# --------------------------------------------------------------------------- #
#  Stage 1 — Naming
# --------------------------------------------------------------------------- #


def _stage_naming(arch: Architecture, report: VerificationReport) -> bool:
    ok = True
    layer_names = {ly.name for ly in arch.layers}
    if not layer_names:
        report.add_error("NO_LAYERS", "architecture declares no layers")
        return False

    duplicates = _find_duplicates(ly.name for ly in arch.layers)
    for dup in duplicates:
        report.add_error(
            "DUPLICATE_LAYER",
            f"layer name {dup!r} is declared more than once",
            "give each `## layer` heading a unique name",
        )
        ok = False

    for edge in arch.flow:
        if edge.source not in layer_names:
            report.add_error(
                "UNKNOWN_LAYER_REFERENCE",
                f"flow edge source {edge.source!r} is not a declared layer",
                f"declare `## layer {edge.source}` or fix the typo",
            )
            ok = False
        if edge.target not in layer_names:
            report.add_error(
                "UNKNOWN_LAYER_REFERENCE",
                f"flow edge target {edge.target!r} is not a declared layer",
                f"declare `## layer {edge.target}` or fix the typo",
            )
            ok = False

    # Inputs and outputs should appear in `## tensors`.
    tensor_names = {t.name for t in arch.tensors}
    for ly in arch.layers:
        if ly.is_input and tensor_names and ly.name not in tensor_names:
            report.add_warning(
                "INPUT_NOT_IN_TENSORS",
                f"input layer {ly.name!r} has no matching row in `## tensors`",
                "add a row to `## tensors` declaring this input's shape and dtype",
            )
        if ly.is_output and tensor_names and ly.name not in tensor_names:
            report.add_warning(
                "OUTPUT_NOT_IN_TENSORS",
                f"output layer {ly.name!r} has no matching row in `## tensors`",
                "add a row to `## tensors` declaring this output's shape and dtype",
            )

    if not arch.inputs():
        report.add_error(
            "NO_INPUT_LAYER",
            "no layer is marked `[input]`",
            "mark the entry layer with `## layer <name> [input]`",
        )
        ok = False
    if not arch.outputs():
        report.add_error(
            "NO_OUTPUT_LAYER",
            "no layer is marked `[output]`",
            "mark the exit layer with `## layer <name> [output]`",
        )
        ok = False

    return ok


def _find_duplicates(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for n in names:
        if n in seen and n not in dupes:
            dupes.append(n)
        seen.add(n)
    return dupes


# --------------------------------------------------------------------------- #
#  Stage 2 — Structural
# --------------------------------------------------------------------------- #


def _stage_structural(arch: Architecture, report: VerificationReport) -> bool:
    ok = True
    # Adjacency.
    succ: dict[str, list[str]] = {ly.name: [] for ly in arch.layers}
    pred: dict[str, list[str]] = {ly.name: [] for ly in arch.layers}
    for e in arch.flow:
        succ[e.source].append(e.target)
        pred[e.target].append(e.source)

    # Cycle detection (DFS with 3-color).
    color: dict[str, int] = {ly.name: 0 for ly in arch.layers}   # 0=white,1=gray,2=black
    cycle_path: list[str] | None = None

    def dfs(node: str, stack: list[str]) -> bool:
        nonlocal cycle_path
        color[node] = 1
        stack.append(node)
        for nxt in succ[node]:
            if color[nxt] == 1:
                cycle_path = stack[stack.index(nxt):] + [nxt]
                return True
            if color[nxt] == 0:
                if dfs(nxt, stack):
                    return True
        stack.pop()
        color[node] = 2
        return False

    for name in [ly.name for ly in arch.layers]:
        if color[name] == 0 and dfs(name, []):
            break

    if cycle_path:
        report.add_error(
            "CYCLE_DETECTED",
            f"layer graph contains a cycle: {' -> '.join(cycle_path)}",
            "remove or re-direct one of the edges to break the cycle",
        )
        ok = False

    # Reachability from inputs.
    inputs = [ly.name for ly in arch.inputs()]
    reachable: set[str] = set()
    stack = list(inputs)
    while stack:
        n = stack.pop()
        if n in reachable:
            continue
        reachable.add(n)
        stack.extend(succ[n])

    unreachable = [ly.name for ly in arch.layers if ly.name not in reachable]
    for u in unreachable:
        report.add_error(
            "UNREACHABLE_LAYER",
            f"layer {u!r} is not reachable from any `[input]` layer",
            "connect this layer with a `## flow` edge from a reachable predecessor",
        )
        ok = False

    # Co-reachability to outputs (only meaningful when no cycle).
    if not cycle_path:
        outputs = {ly.name for ly in arch.outputs()}
        co_reach: set[str] = set()
        stack = list(outputs)
        while stack:
            n = stack.pop()
            if n in co_reach:
                continue
            co_reach.add(n)
            stack.extend(pred[n])

        for ly in arch.layers:
            if ly.name not in co_reach:
                report.add_error(
                    "LAYER_NOT_REACHING_OUTPUT",
                    f"layer {ly.name!r} has no path to any `[output]` layer",
                    "add a `## flow` edge from this layer toward an output, or remove the layer",
                )
                ok = False

    # Input layers should have no predecessors; output layers no successors.
    for ly in arch.layers:
        if ly.is_input and pred[ly.name]:
            report.add_warning(
                "INPUT_HAS_PREDECESSOR",
                f"input layer {ly.name!r} has incoming edges: {pred[ly.name]}",
                "remove the incoming `## flow` edges, or unmark this layer as `[input]`",
            )
        if ly.is_output and succ[ly.name]:
            report.add_warning(
                "OUTPUT_HAS_SUCCESSOR",
                f"output layer {ly.name!r} has outgoing edges: {succ[ly.name]}",
                "remove the outgoing `## flow` edges, or unmark this layer as `[output]`",
            )

    return ok


# --------------------------------------------------------------------------- #
#  Stage 3 — Shape inference
# --------------------------------------------------------------------------- #


def _stage_shape(arch: Architecture, report: VerificationReport) -> None:
    """Topologically traverse the layer DAG, propagating shapes through ops."""
    # Seed shapes for input layers from `## tensors`.
    shapes: dict[str, tuple[str, ...]] = {}
    for ly in arch.inputs():
        t = arch.tensor(ly.name)
        if t is not None:
            shapes[ly.name] = t.shape
        elif ly.declared_shape is not None:
            shapes[ly.name] = ly.declared_shape
        else:
            report.add_warning(
                "INPUT_SHAPE_UNDECLARED",
                f"input layer {ly.name!r} has no declared shape",
                "add a row to `## tensors` for this input",
            )

    pred: dict[str, list[str]] = {ly.name: [] for ly in arch.layers}
    succ: dict[str, list[str]] = {ly.name: [] for ly in arch.layers}
    for e in arch.flow:
        pred[e.target].append(e.source)
        succ[e.source].append(e.target)

    # Topological sort.
    indeg = {n: len(pred[n]) for n in pred}
    queue = [n for n, d in indeg.items() if d == 0]
    topo: list[str] = []
    while queue:
        n = queue.pop(0)
        topo.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)

    for name in topo:
        ly = arch.layer(name)
        if ly is None:
            continue
        if ly.is_input:
            continue
        # Compute input shapes from predecessor outputs, in flow-row order.
        ordered_preds = [e.source for e in arch.flow if e.target == name]
        in_shapes: list[tuple[str, ...]] = []
        missing_any = False
        for p in ordered_preds:
            if p not in shapes:
                missing_any = True
                break
            in_shapes.append(shapes[p])
        if missing_any:
            # Skip — earlier-stage error will explain.
            continue

        if ly.is_output:
            # Output layer just relays — must have exactly one predecessor.
            if len(in_shapes) != 1:
                report.add_error(
                    "OUTPUT_ARITY_MISMATCH",
                    f"output layer {ly.name!r} must have exactly 1 incoming edge; got {len(in_shapes)}",
                    "merge multiple inputs with an explicit `Add` or `Concat` layer first",
                )
                continue
            shapes[name] = in_shapes[0]
            # Compare with declared output tensor shape if present.
            t = arch.tensor(ly.name)
            if t is not None and t.shape != in_shapes[0]:
                report.add_error(
                    "OUTPUT_SHAPE_MISMATCH",
                    f"output {ly.name!r}: declared shape {t.shape} but inferred {in_shapes[0]}",
                    "update the `## tensors` row or fix the upstream op chain",
                )
            continue

        if ly.op is None:
            report.add_error(
                "MISSING_OP",
                f"layer {ly.name!r} has no `op:` and is not an `[input]`/`[output]`",
                f"add a bullet `- op: <Op>(args)` — see standard ops: {', '.join(op_names())}",
            )
            continue

        try:
            spec = get_op(ly.op.name)
        except UnknownOpError:
            # Unknown op — handled in Stage 5. Use declared shape if available.
            if ly.declared_shape is not None:
                shapes[name] = ly.declared_shape
            continue

        if spec.arity != -1 and len(in_shapes) != spec.arity:
            report.add_error(
                "INPUT_ARITY_MISMATCH",
                f"layer {ly.name!r}: op {spec.name} expects {spec.arity} input(s), got {len(in_shapes)}",
                "add or remove `## flow` edges so the input count matches the op",
            )
            continue

        try:
            out_shape = spec.infer(ly.op.args, in_shapes)
        except ShapeRuleError as ex:
            report.add_error(
                "SHAPE_MISMATCH",
                f"layer {ly.name!r}: {ex}",
                "check the layer's op args and incoming shapes",
            )
            continue

        shapes[name] = out_shape

        # If the user declared a shape, check it matches.
        if ly.declared_shape is not None and ly.declared_shape != out_shape:
            report.add_error(
                "DECLARED_SHAPE_MISMATCH",
                f"layer {ly.name!r}: declared shape {ly.declared_shape} differs from inferred {out_shape}",
                "remove the `shape:` bullet or fix the upstream chain",
            )

    report.inferred_shapes = {k: v for k, v in shapes.items()}


# --------------------------------------------------------------------------- #
#  Stage 4 — Resource bounds
# --------------------------------------------------------------------------- #


def _stage_resources(arch: Architecture, report: VerificationReport) -> None:
    # Build a substitution map from hyperparameter names to their default values.
    # This lets us resolve symbolic op-args like `Linear(in_dim, hidden)` to
    # concrete integers for param counting. The numeric values are estimates
    # based on declared defaults — overrideable at runtime by the host.
    hp_subs: dict[str, str] = {}
    for hp in arch.hyperparameters:
        if hp.default is not None:
            hp_subs[hp.name] = str(hp.default)

    # Compute parameter count and depth (longest path in DAG).
    total_params = 0
    for ly in arch.layers:
        if ly.op is None:
            continue
        try:
            spec = get_op(ly.op.name)
        except UnknownOpError:
            continue
        resolved_args = [_resolve_arg(a, hp_subs) for a in ly.op.args]
        ordered_preds = [e.source for e in arch.flow if e.target == ly.name]
        in_shapes = [report.inferred_shapes[p] for p in ordered_preds if p in report.inferred_shapes]
        try:
            total_params += spec.params(resolved_args, in_shapes)
        except Exception:
            pass
    report.param_count = total_params

    depth = _longest_path_depth(arch)
    report.depth = depth

    # Evaluate declared invariants.
    for inv in arch.invariants:
        if inv.kind == "param_count":
            if not _compare(total_params, inv.op, inv.value):
                report.add_error(
                    "PARAM_BUDGET_EXCEEDED",
                    f"param_count = {total_params} fails `param_count {inv.op} {inv.value}`",
                    "reduce layer widths, depths, or relax the invariant",
                )
        elif inv.kind == "depth":
            if not _compare(depth, inv.op, inv.value):
                report.add_error(
                    "DEPTH_BUDGET_EXCEEDED",
                    f"depth = {depth} fails `depth {inv.op} {inv.value}`",
                    "shorten the longest layer chain or relax the invariant",
                )
        elif inv.kind == "output_shape":
            # Compare against every output layer's inferred shape.
            for out in arch.outputs():
                inferred = report.inferred_shapes.get(out.name)
                if inferred is not None and inferred != tuple(inv.value):  # type: ignore[arg-type]
                    report.add_error(
                        "OUTPUT_SHAPE_INVARIANT",
                        f"output {out.name!r} shape {inferred} != declared invariant {tuple(inv.value)}",  # type: ignore[arg-type]
                        "adjust the architecture so the final output matches the invariant",
                    )
        elif inv.kind == "vram_estimate":
            # Enforced in Stage 6 (runtime), where the calibrated QLoRA estimate
            # is computed — keeps a single source of truth for the number.
            pass
        elif inv.kind == "flops":
            # Best-effort: skip — FLOPs estimation is out of scope for v0.1.
            report.add_warning(
                "FLOPS_NOT_IMPLEMENTED",
                "flops invariants are accepted but not yet enforced",
                "remove the flops invariant or run an external FLOPs profiler",
            )


def _resolve_arg(arg: str, hp_subs: dict[str, str]) -> str:
    """Resolve a single op-arg token by hyperparameter substitution.

    Bare hyperparameter names ('d_model') become their default values.
    Simple arithmetic ('d_model * 4') is evaluated when fully resolved.
    Anything else is passed through unchanged.
    """
    arg = arg.strip()
    if arg in hp_subs:
        return hp_subs[arg]
    # Try evaluating a simple expression with hyperparameter substitution.
    if any(c in arg for c in "+-*/"):
        try:
            substituted = arg
            # Replace longest names first to avoid partial overlaps.
            for name in sorted(hp_subs, key=len, reverse=True):
                substituted = _word_replace(substituted, name, hp_subs[name])
            # Only allow digits and basic arithmetic — no fallthrough to arbitrary eval.
            allowed = set("0123456789+-*/() .")
            if all(c in allowed for c in substituted):
                value = eval(substituted, {"__builtins__": {}}, {})  # noqa: S307
                if isinstance(value, (int, float)):
                    return str(int(value)) if value == int(value) else str(value)
        except Exception:
            pass
    return arg


def _word_replace(text: str, word: str, replacement: str) -> str:
    """Replace `word` as a whole identifier (not a substring) inside `text`."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if (
            text[i:i + len(word)] == word
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
            and (
                i + len(word) >= n
                or not (text[i + len(word)].isalnum() or text[i + len(word)] == "_")
            )
        ):
            out.append(replacement)
            i += len(word)
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _compare(value: int | float, op: str, bound) -> bool:
    try:
        b = float(bound)
    except (TypeError, ValueError):
        return True
    if op == "<=":
        return value <= b
    if op == ">=":
        return value >= b
    if op in ("==", "="):
        return value == b
    return True


def _longest_path_depth(arch: Architecture) -> int:
    pred: dict[str, list[str]] = {ly.name: [] for ly in arch.layers}
    succ: dict[str, list[str]] = {ly.name: [] for ly in arch.layers}
    for e in arch.flow:
        pred[e.target].append(e.source)
        succ[e.source].append(e.target)

    indeg = {n: len(pred[n]) for n in pred}
    queue = [n for n, d in indeg.items() if d == 0]
    topo: list[str] = []
    while queue:
        n = queue.pop(0)
        topo.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)

    dist = {n: 0 for n in pred}
    for n in topo:
        for m in succ[n]:
            if dist[n] + 1 > dist[m]:
                dist[m] = dist[n] + 1
    return max(dist.values()) if dist else 0


# --------------------------------------------------------------------------- #
#  Stage 5 — Op coverage
# --------------------------------------------------------------------------- #


def _stage_ops(arch: Architecture, report: VerificationReport) -> None:
    for ly in arch.layers:
        if ly.op is None:
            continue
        try:
            get_op(ly.op.name)
        except UnknownOpError:
            report.add_warning(
                "UNKNOWN_OP",
                f"layer {ly.name!r} uses op {ly.op.name!r} which is not in the standard library",
                f"either pick from the standard library ({', '.join(op_names())})"
                f" or implement it as a custom module in the host code",
            )


# --------------------------------------------------------------------------- #
#  Stage 6 — Runtime backend coverage (informational)
# --------------------------------------------------------------------------- #


def _stage_runtime(arch: Architecture, report: VerificationReport) -> None:
    """Report what the Unsloth runtime backend can do with this architecture.

    Records a `runtime` block on the report: which family the architecture maps
    to, the three-state Unsloth status (supported / unsupported / unknown), the
    loader class, default LoRA targets, and — for decoder LLMs above the size
    floor — a calibrated best-effort QLoRA VRAM *range*.

    Backend coverage is informational and never invalidates a design. The one
    runtime check that can be flagged is a declared `vram_estimate` invariant,
    and only as a **warning** (the estimate is a calibrated heuristic, not a
    guarantee) — promoted to an error only under `--strict`.
    """
    cap = capability.analyze(arch=arch, param_count=report.param_count)
    report.runtime = cap.to_dict()

    invariants = [inv for inv in arch.invariants if inv.kind == "vram_estimate"]
    if not invariants:
        return
    est = cap.vram_estimate
    if est is None:
        report.add_warning(
            "VRAM_ESTIMATE_NOT_APPLICABLE",
            f"a `vram_estimate` invariant is declared but no QLoRA estimate "
            f"applies: {cap.vram_note}",
            "remove the invariant, or apply it to a decoder-LLM architecture",
        )
        return
    for inv in invariants:
        # Compare the central estimate against the budget; surface the range so
        # the reader sees the uncertainty rather than trusting a point value.
        if not _compare(est.central_bytes, inv.op, inv.value):
            report.add_warning(
                "VRAM_BUDGET_EXCEEDED",
                f"estimated QLoRA VRAM ~{est.central_gib} GiB "
                f"(range {est.low_gib}–{est.high_gib} GiB) likely fails "
                f"`vram_estimate {inv.op} {inv.value}` — calibrated estimate, "
                f"not a guarantee",
                "raise the budget, lower max_seq_length / batch_size / LoRA "
                "rank, or pick a smaller base model — then measure to confirm",
            )
