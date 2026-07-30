"""Stage 6 — topology property checking.

The other five stages answer *"is this DAG well-formed and shape-consistent?"*.
This one answers *"is subgraph A structurally incapable of influencing subgraph
B?"* — the question that matters when an information boundary is the design
claim rather than an implementation detail.

Ported from `orca-lang`'s bounded model checker (see
`packages/orca-lang/docs/property-checking-design.md`), adapted from state
machines to layer DAGs. Vocabulary and error-code names are deliberately
identical so the two languages stay one family; `reachable`, `unreachable` and
`passes_through` keep their `orca-lang` spelling verbatim.

**One semantic difference matters.** `orca-lang` ignores transition guards, so
its answers are overapproximations and its messages carry a disclaimer to that
effect. A `## flow` edge carries no guard: if the row is in the table, the
tensor flows along it on every forward pass. So these checks are **exact**, and
a counterexample path is a real activation/gradient path — not a hypothesis.
Never copy `orca-lang`'s guard disclaimer into these messages; here it would be
false.
"""
from __future__ import annotations

from n_orca.ast import Architecture, Property
from n_orca.verifier.verifier import (
    PropertyResult,
    VerificationReport,
    _resolve_arg,
    hp_substitutions,
    layer_params,
)

#: Operands each kind cannot do without. A recognized kind whose required
#: operand is `None` means the parser could not match that kind's grammar, and
#: the bullet is malformed.
_REQUIRED: dict[str, tuple[str, ...]] = {
    "reachable": ("source", "target"),
    "unreachable": ("source", "target"),
    "passes_through": ("through", "source", "target"),
    "within": ("source", "target", "bound"),
    "param_fraction": ("subject", "bound"),
    "out_dim": ("subject", "bound"),
}

#: Shown in the `PROPERTY_MALFORMED` suggestion, so the fix is in the message.
_GRAMMAR: dict[str, str] = {
    "reachable": "reachable: <B> from <A>",
    "unreachable": "unreachable: <B> from <A>",
    "passes_through": "passes_through: <C> for <A> -> <B>",
    "within": "within: <B> from <A> <= <N>",
    "param_fraction": "param_fraction: <group-or-layer> <= <P>%",
    "out_dim": "out_dim: <layer> <= <expr>",
}


def check_properties(arch: Architecture, report: VerificationReport) -> None:
    """Run Stage 6 against `arch`, recording results on `report`.

    Called only when Stages 1–2 passed; see `verify()`. Group diagnostics live
    here too, and are therefore skipped along with the stage — there is no
    point telling an author their group table overlaps while their graph still
    has a cycle in it.
    """
    report.groups = {g.name: list(g.layers) for g in arch.groups}

    layer_names = {ly.name for ly in arch.layers}
    _check_groups(arch, report, layer_names)

    if not arch.properties:
        return

    succ = _adjacency(arch)
    hp_subs = hp_substitutions(arch)

    for prop in arch.properties:
        report.properties.append(
            _check_one(prop, arch, report, succ, layer_names, hp_subs)
        )


# --------------------------------------------------------------------------- #
#  Groups
# --------------------------------------------------------------------------- #


def _check_groups(
    arch: Architecture, report: VerificationReport, layer_names: set[str]
) -> None:
    owner: dict[str, str] = {}
    for group in arch.groups:
        if group.name in layer_names:
            report.add_error(
                "GROUP_NAME_COLLISION",
                f"group {group.name!r} has the same name as a declared layer",
                f"rename the group (e.g. {group.name}_group) so property operands"
                f" resolve to exactly one thing",
            )
        for member in group.layers:
            if member not in layer_names:
                report.add_error(
                    "PROPERTY_UNKNOWN_REFERENCE",
                    f"group {group.name!r} lists {member!r}, which is not a declared layer",
                    f"declare `## layer {member}` or remove it from the group",
                )
                continue
            if member in owner and owner[member] != group.name:
                report.add_warning(
                    "GROUP_OVERLAP",
                    f"layer {member!r} belongs to more than one group:"
                    f" {owner[member]!r} and {group.name!r}",
                    "make the groups disjoint — overlap usually means the boundary"
                    " being asserted is not the one intended",
                )
            else:
                owner[member] = group.name


# --------------------------------------------------------------------------- #
#  Graph helpers
# --------------------------------------------------------------------------- #


def _adjacency(arch: Architecture) -> dict[str, list[str]]:
    """Successor lists in `## flow` row order.

    Row order (not set order) is what makes every reported counterexample a
    deterministic function of the document — the paths go into error messages
    that tests and CI diff. Duplicate rows between the same pair of layers are
    legal (two tensors along one edge) and collapse here.
    """
    succ: dict[str, list[str]] = {ly.name: [] for ly in arch.layers}
    seen: set[tuple[str, str]] = set()
    for e in arch.flow:
        key = (e.source, e.target)
        if key in seen or e.source not in succ:
            continue
        seen.add(key)
        succ[e.source].append(e.target)
    return succ


def _bfs(
    succ: dict[str, list[str]],
    sources: list[str],
    *,
    exclude: frozenset[str] = frozenset(),
) -> tuple[dict[str, int], dict[str, str]]:
    """Multi-source BFS. Returns (distance-from-nearest-source, parent).

    Sources are seeded at depth 0 in the order given (group member order, i.e.
    document order), and the frontier is a FIFO, so the parent tree — and hence
    every reconstructed path — is deterministic and shortest.

    A layer reaches itself by the empty path, so every source has distance 0.
    """
    dist: dict[str, int] = {}
    parent: dict[str, str] = {}
    queue: list[str] = []
    for s in sources:
        if s in exclude or s in dist or s not in succ:
            continue
        dist[s] = 0
        queue.append(s)

    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for nxt in succ.get(node, ()):
            if nxt in exclude or nxt in dist:
                continue
            dist[nxt] = dist[node] + 1
            parent[nxt] = node
            queue.append(nxt)
    return dist, parent


def _witness(
    dist: dict[str, int], parent: dict[str, str], targets: list[str]
) -> list[str]:
    """The shortest path to the nearest reachable target, or `[]` if none.

    Ties break toward the earlier target in `targets` (document order), so a
    graph with two equidistant leaks always reports the same one.
    """
    best: str | None = None
    for t in targets:
        if t in dist and (best is None or dist[t] < dist[best]):
            best = t
    if best is None:
        return []
    path = [best]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _chain(path: list[str]) -> str:
    return " -> ".join(path)


# --------------------------------------------------------------------------- #
#  Dispatch
# --------------------------------------------------------------------------- #


def _check_one(
    prop: Property,
    arch: Architecture,
    report: VerificationReport,
    succ: dict[str, list[str]],
    layer_names: set[str],
    hp_subs: dict[str, str],
) -> PropertyResult:
    required = _REQUIRED.get(prop.kind)
    if required is None or any(getattr(prop, attr) is None for attr in required):
        return _malformed(prop, report)

    # Resolve every operand this kind uses, up front.
    operands = {
        attr: getattr(prop, attr)
        for attr in ("source", "target", "through", "subject")
        if getattr(prop, attr) is not None and attr in required
    }
    resolved: dict[str, list[str]] = {}
    for attr, name in operands.items():
        members = _resolve(arch, name, layer_names)
        if members is None:
            return _unknown_reference(prop, report, name)
        resolved[attr] = members

    if prop.kind == "reachable":
        return _check_reachable(prop, report, succ, resolved)
    if prop.kind == "unreachable":
        return _check_unreachable(prop, report, succ, resolved)
    if prop.kind == "passes_through":
        return _check_passes_through(prop, report, succ, resolved)
    if prop.kind == "within":
        return _check_within(prop, report, succ, resolved)
    if prop.kind == "param_fraction":
        return _check_param_fraction(prop, arch, report, resolved, hp_subs)
    return _check_out_dim(prop, arch, report, hp_subs, layer_names)


def _resolve(
    arch: Architecture, name: str, layer_names: set[str]
) -> list[str] | None:
    """Resolve a property operand to the layers it stands for.

    Groups first, then layers. `GROUP_NAME_COLLISION` makes the two disjoint on
    any document that passes group validation, so the order is defensive rather
    than meaningful.
    """
    group = arch.group(name)
    if group is not None:
        return list(group.layers)
    if name in layer_names:
        return [name]
    return None


def _describe(name: str, members: list[str]) -> str:
    """`'cortex' (embed, attn, ...)` for a group; `'attn'` for a bare layer."""
    if members == [name]:
        return repr(name)
    if not members:
        return f"{name!r} (empty group)"
    return f"{name!r} ({', '.join(members)})"


# --------------------------------------------------------------------------- #
#  Diagnostics shared by several kinds
# --------------------------------------------------------------------------- #


def _malformed(prop: Property, report: VerificationReport) -> PropertyResult:
    if prop.kind in _GRAMMAR:
        message = (
            f"cannot read `{prop.raw}` as a `{prop.kind}` property"
        )
        suggestion = f"write it as `{_GRAMMAR[prop.kind]}`"
    else:
        message = f"`{prop.raw}` is not a recognized property"
        suggestion = (
            "use one of: " + "; ".join(_GRAMMAR[k] for k in _GRAMMAR)
        )
    report.add_error("PROPERTY_MALFORMED", message, suggestion)
    return PropertyResult(
        kind=prop.kind, raw=prop.raw, passed=False, code="PROPERTY_MALFORMED"
    )


def _unknown_reference(
    prop: Property, report: VerificationReport, name: str
) -> PropertyResult:
    report.add_error(
        "PROPERTY_UNKNOWN_REFERENCE",
        f"property `{prop.raw}` references {name!r},"
        f" which is not a declared layer or group",
        f"declare `## layer {name}`, add a `## groups` row for it, or fix the typo",
    )
    return PropertyResult(
        kind=prop.kind, raw=prop.raw, passed=False, code="PROPERTY_UNKNOWN_REFERENCE"
    )


def _passed(prop: Property) -> PropertyResult:
    return PropertyResult(kind=prop.kind, raw=prop.raw, passed=True)


def _skipped(prop: Property) -> PropertyResult:
    return PropertyResult(kind=prop.kind, raw=prop.raw, passed=True, skipped=True)


# --------------------------------------------------------------------------- #
#  reachable / unreachable
# --------------------------------------------------------------------------- #


def _check_reachable(
    prop: Property,
    report: VerificationReport,
    succ: dict[str, list[str]],
    resolved: dict[str, list[str]],
) -> PropertyResult:
    """∃a ∈ A, ∃b ∈ B : path(a, b) — one live wire makes influence possible."""
    sources, targets = resolved["source"], resolved["target"]
    dist, parent = _bfs(succ, sources)
    path = _witness(dist, parent, targets)
    if path:
        return PropertyResult(kind=prop.kind, raw=prop.raw, passed=True, path=path)

    report.add_error(
        "PROPERTY_REACHABILITY_FAIL",
        f"property `{prop.raw}` fails: no path from"
        f" {_describe(prop.source or '', sources)} to"
        f" {_describe(prop.target or '', targets)}",
        f"add a `## flow` path from {prop.source} to {prop.target},"
        f" or drop the property if the two are meant to be independent",
    )
    return PropertyResult(
        kind=prop.kind, raw=prop.raw, passed=False, code="PROPERTY_REACHABILITY_FAIL"
    )


def _check_unreachable(
    prop: Property,
    report: VerificationReport,
    succ: dict[str, list[str]],
    resolved: dict[str, list[str]],
) -> PropertyResult:
    """∀a ∈ A, ∀b ∈ B : ¬path(a, b) — the exact complement of `reachable`.

    Directed and asymmetric: `unreachable: core from cortex` can hold while
    `unreachable: cortex from core` fails, which is precisely the situation a
    modulatory bus creates. `from` always names the source.
    """
    sources, targets = resolved["source"], resolved["target"]
    dist, parent = _bfs(succ, sources)
    path = _witness(dist, parent, targets)
    if not path:
        return _passed(prop)

    report.add_error(
        "PROPERTY_EXCLUSION_FAIL",
        f"property `{prop.raw}` fails: a path exists — {_chain(path)}",
        "remove one `## flow` edge on that path to sever the boundary,"
        " or drop the property if this influence is intentional",
    )
    return PropertyResult(
        kind=prop.kind,
        raw=prop.raw,
        passed=False,
        code="PROPERTY_EXCLUSION_FAIL",
        path=path,
    )


# --------------------------------------------------------------------------- #
#  passes_through
# --------------------------------------------------------------------------- #


def _check_passes_through(
    prop: Property,
    report: VerificationReport,
    succ: dict[str, list[str]],
    resolved: dict[str, list[str]],
) -> PropertyResult:
    """Every A→B path visits C, i.e. C is a cut set separating A from B.

    The whole of `C` is deleted at once. Deleting members one at a time would
    be a weaker check and would pass on a redundant two-lane bus that the
    author means to treat as a single boundary.

    Vacuously true when B is already unreachable from A: every A→B path does
    visit C, there being none.
    """
    sources, targets = resolved["source"], resolved["target"]
    cut = frozenset(resolved["through"])
    dist, parent = _bfs(succ, sources, exclude=cut)
    path = _witness(dist, parent, targets)
    if not path:
        return _passed(prop)

    report.add_error(
        "PROPERTY_PATH_FAIL",
        f"property `{prop.raw}` fails: a path bypasses"
        f" {_describe(prop.through or '', resolved['through'])}"
        f" — {_chain(path)}",
        f"route that path through {prop.through}, remove the bypassing edge,"
        f" or widen {prop.through} to cover the whole boundary",
    )
    return PropertyResult(
        kind=prop.kind,
        raw=prop.raw,
        passed=False,
        code="PROPERTY_PATH_FAIL",
        path=path,
    )


# --------------------------------------------------------------------------- #
#  within
# --------------------------------------------------------------------------- #


def _check_within(
    prop: Property,
    report: VerificationReport,
    succ: dict[str, list[str]],
    resolved: dict[str, list[str]],
) -> PropertyResult:
    """∃a ∈ A, ∃b ∈ B : dist(a, b) ≤ N — `reachable` plus a budget.

    The BFS is unbounded rather than depth-capped: the failure message has to
    report the *actual* shortest distance to be actionable, and a capped search
    cannot know it. The bound is applied as a comparison afterwards. Both cost
    O(V+E).
    """
    bound = _as_int(prop.bound)
    if bound is None or bound < 0:
        return _malformed(prop, report)

    sources, targets = resolved["source"], resolved["target"]
    dist, parent = _bfs(succ, sources)
    path = _witness(dist, parent, targets)

    if not path:
        report.add_error(
            "PROPERTY_DEPTH_FAIL",
            f"property `{prop.raw}` fails: no path from"
            f" {_describe(prop.source or '', sources)} to"
            f" {_describe(prop.target or '', targets)},"
            f" so no depth bound can hold",
            f"connect {prop.source} to {prop.target} with `## flow` edges,"
            f" or drop the property — raising the bound will not help",
        )
        return PropertyResult(
            kind=prop.kind, raw=prop.raw, passed=False, code="PROPERTY_DEPTH_FAIL"
        )

    actual = len(path) - 1
    if actual <= bound:
        return PropertyResult(kind=prop.kind, raw=prop.raw, passed=True, path=path)

    report.add_error(
        "PROPERTY_DEPTH_FAIL",
        f"property `{prop.raw}` fails: shortest path from {prop.source} to"
        f" {prop.target} is {actual} edges, exceeding {bound} — {_chain(path)}",
        f"shorten the chain between {prop.source} and {prop.target},"
        f" or raise the bound to {actual}",
    )
    return PropertyResult(
        kind=prop.kind,
        raw=prop.raw,
        passed=False,
        code="PROPERTY_DEPTH_FAIL",
        path=path,
    )


# --------------------------------------------------------------------------- #
#  param_fraction
# --------------------------------------------------------------------------- #


def _check_param_fraction(
    prop: Property,
    arch: Architecture,
    report: VerificationReport,
    resolved: dict[str, list[str]],
    hp_subs: dict[str, str],
) -> PropertyResult:
    """A group's parameters as a share of the whole model.

    DAG-native — `orca-lang` has no analogue, because a machine has no
    parameters. Expressed as a fraction rather than an absolute count so it
    survives a change of hyperparameters that would invalidate a fixed bound;
    that is also why it is not a `## invariants` entry.
    """
    raw_bound = (prop.bound or "").strip()
    if not raw_bound.endswith("%"):
        return _malformed(prop, report)
    try:
        budget = float(raw_bound[:-1])
    except ValueError:
        return _malformed(prop, report)

    total = report.param_count
    if total <= 0:
        # No denominator — there is nothing to claim a share of.
        return _skipped(prop)

    members = resolved["subject"]
    group_params = 0
    for name in members:
        ly = arch.layer(name)
        if ly is not None:
            group_params += layer_params(arch, ly, report, hp_subs)

    pct = 100.0 * group_params / total
    if pct <= budget:
        return _passed(prop)

    report.add_error(
        "PROPERTY_PARAM_FRACTION_FAIL",
        f"property `{prop.raw}` fails: {prop.subject} holds {group_params:,} of"
        f" {total:,} parameters = {pct:.2f}%, over the {raw_bound} budget",
        f"shrink the layers in {prop.subject}, grow the rest of the model,"
        f" or relax the budget to {pct:.2f}%",
    )
    return PropertyResult(
        kind=prop.kind,
        raw=prop.raw,
        passed=False,
        code="PROPERTY_PARAM_FRACTION_FAIL",
    )


# --------------------------------------------------------------------------- #
#  out_dim
# --------------------------------------------------------------------------- #


def _check_out_dim(
    prop: Property,
    arch: Architecture,
    report: VerificationReport,
    hp_subs: dict[str, str],
    layer_names: set[str],
) -> PropertyResult:
    """A single layer's inferred output width.

    Also DAG-native: a state has no width. The bound goes through
    `_resolve_arg`, the same hyperparameter-substitution path Stage 4 uses for
    op args, so `out_dim: nm_bus <= n_affect` and `out_dim: h <= d_model * 4`
    both work without a second expression evaluator to keep in sync.
    """
    name = prop.subject or ""
    if arch.group(name) is not None:
        report.add_error(
            "PROPERTY_MALFORMED",
            f"cannot read `{prop.raw}`: {name!r} is a group, and a group has no"
            f" output width",
            f"name a single layer instead, e.g. `out_dim: <layer> {prop.op}"
            f" {prop.bound}`",
        )
        return PropertyResult(
            kind=prop.kind, raw=prop.raw, passed=False, code="PROPERTY_MALFORMED"
        )
    if name not in layer_names:
        return _unknown_reference(prop, report, name)

    shape = report.inferred_shapes.get(name)
    if not shape:
        # Stage 3 could not infer this layer's shape and has already said why.
        # A second error about a consequence of that is noise.
        return _skipped(prop)

    actual = _as_int(_resolve_arg(shape[-1], hp_subs))
    bound = _as_int(_resolve_arg(prop.bound or "", hp_subs))
    if actual is None or bound is None:
        # A free symbolic dimension the verifier deliberately treats as opaque.
        # An unresolvable comparison is not a violation.
        return _skipped(prop)

    if actual <= bound:
        return _passed(prop)

    shown = f"{prop.bound} ({bound})" if str(prop.bound) != str(bound) else str(bound)
    report.add_error(
        "PROPERTY_WIDTH_FAIL",
        f"property `{prop.raw}` fails: layer {name!r} output last dim is"
        f" {actual}, exceeding {shown}",
        f"reduce the output width of {name}, or raise the bound to {actual}",
    )
    return PropertyResult(
        kind=prop.kind, raw=prop.raw, passed=False, code="PROPERTY_WIDTH_FAIL"
    )


def _as_int(token: str | None) -> int | None:
    if token is None:
        return None
    try:
        return int(str(token).strip())
    except (TypeError, ValueError):
        return None
