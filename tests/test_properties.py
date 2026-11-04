"""Stage 6 — topology property checking.

The fixtures below are a miniature of `examples/subcortical-value-cortex`:
a `cortex` chain, a `core` chain that reads the input directly, and a `bus`
that carries core -> cortex modulation. The point of the shape is that the
core/cortex boundary is *directed*: `unreachable: core from cortex` holds and
`unreachable: cortex from core` does not.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from n_orca.compiler import compile_mermaid
from n_orca.parser import parse
from n_orca.render import render
from n_orca.verifier import verify

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"


# --------------------------------------------------------------------------- #
#  Fixture construction
# --------------------------------------------------------------------------- #

_HEAD = """# architecture Mini

## hyperparameters

| Name     | Type | Default |
|----------|------|---------|
| d_obs    | int  | 16      |
| d_model  | int  | 32      |
| d_core   | int  | 8       |
| n_affect | int  | 4       |

## tensors

| Name  | Shape        | Dtype   |
|-------|--------------|---------|
| obs   | (B, d_obs)   | float32 |
| y     | (B, d_model) | float32 |
| v_hat | (B, 1)       | float32 |

## layer obs [input]
## layer embed
- op: Linear(d_obs, d_model)
## layer cortex_ln
- op: LayerNorm(d_model)
## layer core_sense
- op: Linear(d_obs, d_core)
## layer core_act
- op: Tanh()
## layer affect_head
- op: Linear(d_core, n_affect)
## layer value_head
- op: Linear(d_core, 1)
## layer nm_bus
- op: Linear(n_affect, n_affect)
## layer nm_receptor
- op: Linear(n_affect, d_model)
## layer nm_gain
- op: Sigmoid()
## layer modulate
- op: Mul()
## layer head
- op: Linear(d_model, d_model)
## layer y [output]
## layer v_hat [output]
"""

_GROUPS = """
## groups

| Name   | Layers                                        |
|--------|-----------------------------------------------|
| core   | core_sense, core_act, affect_head, value_head |
| cortex | embed, cortex_ln, modulate, head              |
| bus    | nm_bus, nm_receptor, nm_gain                  |
"""

_FLOW = """
## flow

| Source      | Target      | Tensor |
|-------------|-------------|--------|
| obs         | embed       | obs    |
| embed       | cortex_ln   | h      |
| obs         | core_sense  | obs    |
| core_sense  | core_act    | c      |
| core_act    | affect_head | c_act  |
| core_act    | value_head  | c_act  |
| value_head  | v_hat       | v      |
| affect_head | nm_bus      | a      |
| nm_bus      | nm_receptor | nm     |
| nm_receptor | nm_gain     | g_pre  |
| nm_gain     | modulate    | g      |
| cortex_ln   | modulate    | h_ln   |
| modulate    | head        | h_mod  |
| head        | y           | y_out  |
"""


def mini(
    properties: str | list[str] = (),
    *,
    groups: str = _GROUPS,
    extra_layers: str = "",
    extra_flow: str = "",
) -> str:
    """Build a mini architecture document.

    `properties` is a list of bullet bodies (without the leading `- `) or a
    raw section body. `extra_flow` rows are appended to `## flow`.
    """
    if isinstance(properties, str):
        props_body = properties
    else:
        props_body = "\n".join(f"- {p}" for p in properties)
    props = f"\n## properties\n{props_body}\n" if props_body else ""
    return _HEAD + extra_layers + groups + _FLOW + extra_flow + props


def check(*args, **kwargs):
    """Parse + verify a mini document, returning the report."""
    archs = parse(mini(*args, **kwargs))
    assert len(archs) == 1
    return verify(archs[0])


def codes(report) -> list[str]:
    return [e.code for e in report.errors]


def wcodes(report) -> list[str]:
    return [w.code for w in report.warnings]


def messages(report, code: str) -> list[str]:
    return [e.message for e in report.errors + report.warnings if e.code == code]


def only(report, code: str) -> str:
    got = messages(report, code)
    assert len(got) == 1, f"expected exactly one {code}, got {got}"
    return got[0]


# --------------------------------------------------------------------------- #
#  Sanity — the fixture itself is well-formed
# --------------------------------------------------------------------------- #


def test_mini_fixture_is_valid_without_properties():
    report = check()
    assert report.valid, codes(report)
    assert report.properties_checked == 0
    assert report.properties_failed == 0


def test_mini_fixture_shapes_and_params():
    report = check()
    assert report.inferred_shapes["nm_bus"] == ("B", "n_affect")
    assert report.inferred_shapes["y"] == ("B", "d_model")
    assert report.param_count == 2025
    assert report.depth == 9


# --------------------------------------------------------------------------- #
#  Parsing
# --------------------------------------------------------------------------- #


def test_groups_parse_into_ast():
    arch = parse(mini())[0]
    assert [g.name for g in arch.groups] == ["core", "cortex", "bus"]
    assert arch.group("bus").layers == ["nm_bus", "nm_receptor", "nm_gain"]
    # Member order is preserved verbatim.
    assert arch.group("core").layers == [
        "core_sense", "core_act", "affect_head", "value_head",
    ]
    assert arch.group("nope") is None


def test_all_six_property_kinds_parse():
    arch = parse(mini([
        "reachable: v_hat from obs",
        "unreachable: core from cortex",
        "passes_through: bus for core -> cortex",
        "within: y from obs <= 12",
        "param_fraction: core <= 10%",
        "out_dim: nm_bus <= n_affect",
    ]))[0]
    props = arch.properties
    assert [p.kind for p in props] == [
        "reachable", "unreachable", "passes_through",
        "within", "param_fraction", "out_dim",
    ]

    assert (props[0].source, props[0].target) == ("obs", "v_hat")
    assert (props[1].source, props[1].target) == ("cortex", "core")
    assert (props[2].through, props[2].source, props[2].target) == ("bus", "core", "cortex")
    assert (props[3].source, props[3].target, props[3].op, props[3].bound) == (
        "obs", "y", "<=", "12",
    )
    assert (props[4].subject, props[4].op, props[4].bound) == ("core", "<=", "10%")
    assert (props[5].subject, props[5].op, props[5].bound) == ("nm_bus", "<=", "n_affect")

    # `raw` is the verbatim bullet, so messages can quote the author back.
    assert props[2].raw == "passes_through: bus for core -> cortex"


def test_malformed_bullets_do_not_raise_at_parse_time():
    arch = parse(mini([
        "reachable: v_hat",
        "eventually: y from obs",
        "param_fraction: core <= 1",
    ]))[0]
    assert len(arch.properties) == 3
    assert arch.properties[0].raw == "reachable: v_hat"


def test_unknown_sections_are_still_ignored_silently():
    src = mini(["reachable: v_hat from obs"]) + "\n## notes\n- this is prose\n"
    report = verify(parse(src)[0])
    assert report.valid, codes(report)


# --------------------------------------------------------------------------- #
#  Render round-trip
# --------------------------------------------------------------------------- #


def test_render_round_trips_groups_and_properties():
    src = mini([
        "reachable: v_hat from obs",
        "unreachable: core from cortex",
        "passes_through: bus for core -> cortex",
        "within: y from obs <= 12",
        "param_fraction: core <= 10%",
        "out_dim: nm_bus <= n_affect",
    ])
    arch = parse(src)[0]
    again = parse(render(arch))[0]
    assert again.groups == arch.groups
    assert again.properties == arch.properties


def test_render_unchanged_when_no_groups_or_properties():
    arch = parse(mini(groups=""))[0]
    out = render(arch)
    assert "## groups" not in out
    assert "## properties" not in out


# --------------------------------------------------------------------------- #
#  Group diagnostics
# --------------------------------------------------------------------------- #


def test_group_overlap_is_a_warning_naming_the_layer_and_both_groups():
    groups = """
## groups

| Name   | Layers                                        |
|--------|-----------------------------------------------|
| core   | core_sense, core_act, affect_head, value_head |
| bus    | nm_bus, nm_receptor, nm_gain                  |
| cortex | embed, cortex_ln, modulate, head, nm_gain     |
"""
    report = check(["unreachable: core from cortex"], groups=groups)
    assert "GROUP_OVERLAP" in wcodes(report)
    msg = only(report, "GROUP_OVERLAP")
    assert "nm_gain" in msg
    assert "bus" in msg and "cortex" in msg
    # Warning only — still valid.
    assert report.valid, codes(report)


def test_group_name_collision_is_an_error():
    groups = """
## groups

| Name  | Layers            |
|-------|-------------------|
| embed | embed, cortex_ln  |
"""
    report = check(groups=groups)
    assert "GROUP_NAME_COLLISION" in codes(report)
    assert not report.valid
    assert "embed" in only(report, "GROUP_NAME_COLLISION")


def test_group_member_that_is_not_a_layer_is_unknown_reference():
    groups = """
## groups

| Name | Layers                  |
|------|-------------------------|
| core | core_sense, core_ghost  |
"""
    report = check(groups=groups)
    assert "PROPERTY_UNKNOWN_REFERENCE" in codes(report)
    assert "core_ghost" in only(report, "PROPERTY_UNKNOWN_REFERENCE")


def test_strict_promotes_group_overlap_to_error():
    groups = """
## groups

| Name   | Layers                       |
|--------|------------------------------|
| bus    | nm_bus, nm_receptor          |
| wider  | nm_receptor, nm_gain         |
"""
    arch = parse(mini(groups=groups))[0]
    lenient = verify(arch)
    assert lenient.valid
    strict = verify(arch, strict=True)
    assert not strict.valid
    assert "GROUP_OVERLAP" in codes(strict)


# --------------------------------------------------------------------------- #
#  reachable
# --------------------------------------------------------------------------- #


def test_reachable_passes():
    report = check(["reachable: v_hat from obs"])
    assert report.valid, codes(report)
    assert report.properties_checked == 1
    assert report.properties_failed == 0


def test_reachable_fails_with_no_path():
    report = check(["reachable: core from cortex"])
    assert "PROPERTY_REACHABILITY_FAIL" in codes(report)
    assert report.properties_failed == 1
    msg = only(report, "PROPERTY_REACHABILITY_FAIL")
    assert "reachable: core from cortex" in msg


def test_reachable_is_existential_over_group_members():
    # `bus` reaches `cortex` only via nm_gain -> modulate. One pair is enough.
    report = check(["reachable: cortex from bus"])
    assert report.valid, codes(report)


# --------------------------------------------------------------------------- #
#  unreachable — the asymmetric boundary
# --------------------------------------------------------------------------- #


def test_unreachable_passes_on_the_real_isolation_boundary():
    report = check(["unreachable: core from cortex"])
    assert report.valid, codes(report)
    assert report.properties_failed == 0


def test_unreachable_is_directed_and_asymmetric():
    """core -> cortex is a design edge; cortex -> core must not exist."""
    ok = check(["unreachable: core from cortex"])
    assert ok.valid, codes(ok)

    bad = check(["unreachable: cortex from core"])
    assert not bad.valid
    assert "PROPERTY_EXCLUSION_FAIL" in codes(bad)


def test_unreachable_failure_carries_the_counterexample_path():
    report = check(["unreachable: cortex from core"])
    msg = only(report, "PROPERTY_EXCLUSION_FAIL")
    assert "affect_head -> nm_bus -> nm_receptor -> nm_gain -> modulate" in msg


def test_unreachable_layer_to_layer():
    ok = check(["unreachable: core_sense from cortex_ln"])
    assert ok.valid, codes(ok)
    bad = check(["unreachable: modulate from affect_head"])
    assert "PROPERTY_EXCLUSION_FAIL" in codes(bad)
    assert "affect_head -> nm_bus -> nm_receptor -> nm_gain -> modulate" in only(
        bad, "PROPERTY_EXCLUSION_FAIL"
    )


def test_reachable_and_unreachable_are_exact_complements():
    for src, tgt in [("cortex", "core"), ("core", "cortex"), ("obs", "y")]:
        r = check([f"reachable: {tgt} from {src}"])
        u = check([f"unreachable: {tgt} from {src}"])
        assert r.properties_failed + u.properties_failed == 1, (src, tgt)


def test_counterexample_path_is_shortest_and_deterministic():
    first = only(check(["unreachable: cortex from core"]), "PROPERTY_EXCLUSION_FAIL")
    for _ in range(5):
        assert only(
            check(["unreachable: cortex from core"]), "PROPERTY_EXCLUSION_FAIL"
        ) == first
    # Shortest witness: affect_head is a `core` member one hop from the bus.
    assert "core_sense" not in first


def test_counterexample_path_is_exposed_structurally():
    report = check(["unreachable: cortex from core"])
    result = report.properties[0]
    assert result.passed is False
    assert result.code == "PROPERTY_EXCLUSION_FAIL"
    assert result.path == [
        "affect_head", "nm_bus", "nm_receptor", "nm_gain", "modulate",
    ]


# --------------------------------------------------------------------------- #
#  passes_through
# --------------------------------------------------------------------------- #


def test_passes_through_passes_when_the_cut_set_is_the_only_route():
    report = check(["passes_through: bus for core -> cortex"])
    assert report.valid, codes(report)


def test_passes_through_catches_a_bypass_edge_around_the_named_cut():
    extra_layers = """
## layer bypass_proj
- op: Linear(n_affect, d_model)
"""
    extra_flow = """| affect_head | bypass_proj | a_by |
| bypass_proj | modulate    | by   |
"""
    report = check(
        ["passes_through: bus for core -> cortex"],
        extra_layers=extra_layers,
        extra_flow=extra_flow,
    )
    assert "PROPERTY_PATH_FAIL" in codes(report)
    msg = only(report, "PROPERTY_PATH_FAIL")
    assert "affect_head -> bypass_proj -> modulate" in msg
    assert "passes_through: bus for core -> cortex" in msg


def test_passes_through_deletes_the_whole_cut_set_not_one_member():
    """Two parallel bus lanes: deleting either alone leaves a route."""
    extra_layers = """
## layer nm_lane_b
- op: Linear(n_affect, d_model)
"""
    extra_flow = """| affect_head | nm_lane_b | a_b  |
| nm_lane_b   | modulate  | g_b  |
"""
    groups = """
## groups

| Name   | Layers                                        |
|--------|-----------------------------------------------|
| core   | core_sense, core_act, affect_head, value_head |
| cortex | embed, cortex_ln, modulate, head              |
| bus    | nm_bus, nm_receptor, nm_gain, nm_lane_b       |
"""
    ok = check(
        ["passes_through: bus for core -> cortex"],
        groups=groups,
        extra_layers=extra_layers,
        extra_flow=extra_flow,
    )
    assert ok.valid, codes(ok)

    # Naming only one lane as the cut leaves the other lane as a bypass.
    narrow = groups.replace("nm_bus, nm_receptor, nm_gain, nm_lane_b", "nm_bus, nm_receptor, nm_gain")
    bad = check(
        ["passes_through: bus for core -> cortex"],
        groups=narrow,
        extra_layers=extra_layers,
        extra_flow=extra_flow,
    )
    assert "PROPERTY_PATH_FAIL" in codes(bad)
    assert "affect_head -> nm_lane_b -> modulate" in only(bad, "PROPERTY_PATH_FAIL")


def test_passes_through_holds_vacuously_when_no_path_exists():
    # No cortex layer reaches any core layer at all, so every such path
    # (there are none) trivially visits the bus.
    report = check(["passes_through: bus for cortex -> core"])
    assert report.valid, codes(report)


# --------------------------------------------------------------------------- #
#  within
# --------------------------------------------------------------------------- #


def test_within_passes():
    report = check(["within: y from obs <= 12"])
    assert report.valid, codes(report)


def test_within_fails_too_far_and_reports_the_actual_distance():
    report = check(["within: y from obs <= 4"])
    assert "PROPERTY_DEPTH_FAIL" in codes(report)
    msg = only(report, "PROPERTY_DEPTH_FAIL")
    assert "5" in msg
    assert "within: y from obs <= 4" in msg


def test_within_fails_distinctly_when_no_path_exists():
    report = check(["within: core from cortex <= 99"])
    assert "PROPERTY_DEPTH_FAIL" in codes(report)
    msg = only(report, "PROPERTY_DEPTH_FAIL")
    assert "no path" in msg.lower()


def test_within_zero_is_reflexive():
    report = check(["within: obs from obs <= 0"])
    assert report.valid, codes(report)


def test_within_rejects_a_non_integer_bound():
    report = check(["within: y from obs <= many"])
    assert "PROPERTY_MALFORMED" in codes(report)


# --------------------------------------------------------------------------- #
#  param_fraction
# --------------------------------------------------------------------------- #


def test_param_fraction_passes():
    report = check(["param_fraction: core <= 10%"])
    assert report.valid, codes(report)


def test_param_fraction_fails_with_percentage_and_raw_counts():
    report = check(["param_fraction: core <= 1%"])
    assert "PROPERTY_PARAM_FRACTION_FAIL" in codes(report)
    msg = only(report, "PROPERTY_PARAM_FRACTION_FAIL")
    assert "181" in msg          # group params
    assert "2,025" in msg        # total params
    assert "8.9" in msg          # actual percentage


def test_param_fraction_accepts_a_fractional_percent():
    report = check(["param_fraction: core <= 8.94%"])
    assert report.valid, codes(report)
    assert check(["param_fraction: core <= 8.9%"]).properties_failed == 1


def test_param_fraction_works_on_a_single_layer():
    report = check(["param_fraction: head <= 60%"])
    assert report.valid, codes(report)
    assert check(["param_fraction: head <= 10%"]).properties_failed == 1


def test_param_fraction_requires_a_percent_sign():
    report = check(["param_fraction: core <= 1"])
    assert "PROPERTY_MALFORMED" in codes(report)
    assert "param_fraction: core <= 1" in only(report, "PROPERTY_MALFORMED")


# --------------------------------------------------------------------------- #
#  out_dim
# --------------------------------------------------------------------------- #


def test_out_dim_passes_against_a_hyperparameter_bound():
    report = check(["out_dim: nm_bus <= n_affect"])
    assert report.valid, codes(report)


def test_out_dim_passes_against_an_arithmetic_expression():
    report = check(["out_dim: head <= d_core * 4"])
    assert report.valid, codes(report)


def test_out_dim_fails():
    report = check(["out_dim: head <= n_affect"])
    assert "PROPERTY_WIDTH_FAIL" in codes(report)
    msg = only(report, "PROPERTY_WIDTH_FAIL")
    assert "head" in msg
    assert "32" in msg and "4" in msg


def test_out_dim_accepts_a_literal_bound():
    assert check(["out_dim: nm_bus <= 4"]).valid
    assert check(["out_dim: nm_bus <= 3"]).properties_failed == 1


def test_out_dim_on_a_group_is_malformed():
    report = check(["out_dim: cortex <= 512"])
    assert "PROPERTY_MALFORMED" in codes(report)
    assert "cortex" in only(report, "PROPERTY_MALFORMED")


def test_out_dim_skips_an_unresolvable_bound():
    # `S` is a free symbolic dimension the verifier treats as opaque.
    report = check(["out_dim: nm_bus <= S"])
    assert report.valid, codes(report)
    assert report.properties_failed == 0
    assert report.properties_checked == 0
    assert report.properties[0].skipped is True


def test_out_dim_skips_a_hyperparameter_declared_without_a_default():
    """No default means no substitution, so the comparison is unresolvable.

    Distinct from the free-symbol case: the name *is* a declared
    hyperparameter, it just has nothing to resolve to.
    """
    src = mini(["out_dim: nm_bus <= d_budget"]).replace(
        "| n_affect | int  | 4       |",
        "| n_affect | int  | 4       |\n| d_budget | int  |         |",
    )
    arch = parse(src)[0]
    assert arch.hyperparameter("d_budget").default is None
    report = verify(arch)
    assert report.valid, codes(report)
    assert report.properties[0].skipped is True


def test_out_dim_failure_names_the_inferred_shape():
    msg = only(check(["out_dim: head <= n_affect"]), "PROPERTY_WIDTH_FAIL")
    assert "(B, d_model)" in msg
    suggestion = next(
        e.suggestion for e in check(["out_dim: head <= n_affect"]).errors
        if e.code == "PROPERTY_WIDTH_FAIL"
    )
    assert "trailing dimension" in suggestion


# --------------------------------------------------------------------------- #
#  Malformed / unknown references
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bullet", [
    "reachable: v_hat",
    "reachable: from obs",
    "unreachable: core cortex",
    "passes_through: bus for core",
    "passes_through: bus core -> cortex",
    "within: y from obs",
    "within: y from obs <= ",
    "eventually: y from obs",
    "param_fraction: core",
    "out_dim: nm_bus",
    "just some prose",
])
def test_malformed_bullets_report_property_malformed(bullet):
    report = check([bullet])
    assert "PROPERTY_MALFORMED" in codes(report), bullet


@pytest.mark.parametrize("bullet,bad", [
    ("reachable: nowhere from obs", "nowhere"),
    ("unreachable: core from ghost", "ghost"),
    ("passes_through: phantom for core -> cortex", "phantom"),
    ("within: y from ghost <= 3", "ghost"),
    ("param_fraction: ghost <= 1%", "ghost"),
    ("out_dim: ghost <= 4", "ghost"),
])
def test_unknown_operands_report_unknown_reference(bullet, bad):
    report = check([bullet])
    assert "PROPERTY_UNKNOWN_REFERENCE" in codes(report), bullet
    assert bad in only(report, "PROPERTY_UNKNOWN_REFERENCE")


def test_every_new_code_carries_a_suggestion():
    seen: set[str] = set()
    fixtures = [
        (["reachable: core from cortex"], _GROUPS),
        (["unreachable: cortex from core"], _GROUPS),
        (["passes_through: value_head for core -> cortex"], _GROUPS),
        (["within: y from obs <= 1"], _GROUPS),
        (["param_fraction: core <= 1%"], _GROUPS),
        (["out_dim: head <= 4"], _GROUPS),
        (["reachable: nowhere from obs"], _GROUPS),
        (["just some prose"], _GROUPS),
        ([], """
## groups

| Name  | Layers           |
|-------|------------------|
| embed | embed, cortex_ln |
"""),
        ([], """
## groups

| Name  | Layers                |
|-------|-----------------------|
| a     | nm_bus, nm_receptor   |
| b     | nm_receptor, nm_gain  |
"""),
    ]
    for props, groups in fixtures:
        report = check(props, groups=groups)
        for diag in report.errors + report.warnings:
            seen.add(diag.code)
            assert diag.suggestion, f"{diag.code} has no suggestion"

    expected = {
        "PROPERTY_REACHABILITY_FAIL", "PROPERTY_EXCLUSION_FAIL",
        "PROPERTY_PATH_FAIL", "PROPERTY_DEPTH_FAIL",
        "PROPERTY_PARAM_FRACTION_FAIL", "PROPERTY_WIDTH_FAIL",
        "PROPERTY_UNKNOWN_REFERENCE", "PROPERTY_MALFORMED",
        "GROUP_NAME_COLLISION", "GROUP_OVERLAP",
    }
    assert expected <= seen, expected - seen


# --------------------------------------------------------------------------- #
#  Degenerate operands (design §4.6)
# --------------------------------------------------------------------------- #


def test_empty_group_quantifies_vacuously():
    groups = _GROUPS + """
| empty  |                                               |
"""
    # `reachable`/`within` are existential -> false over an empty set.
    assert "PROPERTY_REACHABILITY_FAIL" in codes(
        check(["reachable: empty from obs"], groups=groups)
    )
    # `unreachable`/`passes_through` are universal -> true over an empty set.
    assert check(["unreachable: empty from obs"], groups=groups).valid
    assert check(["passes_through: bus for empty -> cortex"], groups=groups).valid


def test_overlapping_operands_fail_unreachable_by_reflexivity():
    report = check(["unreachable: cortex from cortex"])
    assert "PROPERTY_EXCLUSION_FAIL" in codes(report)
    assert check(["reachable: cortex from cortex"]).valid


def test_a_layer_reaches_itself():
    assert check(["reachable: obs from obs"]).valid
    assert "PROPERTY_EXCLUSION_FAIL" in codes(check(["unreachable: obs from obs"]))


def test_self_reference_failure_explains_the_empty_path():
    """A one-layer witness needs saying out loud — no edge removal can fix it."""
    msg = only(check(["unreachable: obs from obs"]), "PROPERTY_EXCLUSION_FAIL")
    assert "empty path" in msg
    assert "both operands" in msg
    # The bare-path phrasing would read as a truncated message here.
    assert "a path exists — obs" not in msg
    suggestion = next(
        e.suggestion for e in check(["unreachable: obs from obs"]).errors
        if e.code == "PROPERTY_EXCLUSION_FAIL"
    )
    assert "disjoint" in suggestion


def test_partially_overlapping_groups_report_the_shared_layer_not_a_path():
    groups = _GROUPS + """
| shared | modulate, core_sense                           |
"""
    report = check(["unreachable: shared from cortex"], groups=groups)
    msg = only(report, "PROPERTY_EXCLUSION_FAIL")
    # `modulate` is in both `cortex` and `shared`, so the shortest witness is
    # the shared layer itself, not the longer cortex -> core_sense route.
    assert "'modulate'" in msg
    assert "empty path" in msg


# --------------------------------------------------------------------------- #
#  Several properties over the same operands
# --------------------------------------------------------------------------- #


def test_multiple_properties_on_the_same_group_pair_are_checked_independently():
    """Each bullet gets its own verdict; none is inferred from a sibling."""
    report = check([
        "reachable: cortex from core",        # passes — the bus
        "unreachable: cortex from core",     # fails  — same fact, opposite claim
        "passes_through: bus for core -> cortex",   # passes
        "within: cortex from core <= 2",     # fails — shortest is 4
        "within: cortex from core <= 4",     # passes
    ])
    assert report.properties_checked == 5
    assert report.properties_failed == 2
    assert [r.passed for r in report.properties] == [True, False, True, False, True]
    assert [r.code for r in report.properties] == [
        None, "PROPERTY_EXCLUSION_FAIL", None, "PROPERTY_DEPTH_FAIL", None,
    ]


def test_contradictory_pair_always_yields_exactly_one_failure():
    for a, b in [("core", "cortex"), ("cortex", "core"), ("bus", "cortex")]:
        report = check([f"reachable: {b} from {a}", f"unreachable: {b} from {a}"])
        assert report.properties_failed == 1, (a, b)


def test_repeated_identical_properties_each_report():
    report = check(["unreachable: cortex from core"] * 3)
    assert report.properties_failed == 3
    assert len(messages(report, "PROPERTY_EXCLUSION_FAIL")) == 3


# --------------------------------------------------------------------------- #
#  Determinism under fan-in
# --------------------------------------------------------------------------- #


def test_earliest_target_wins_under_many_equidistant_parallel_paths():
    """With 12 equal-length leaks, the witness must be stable and predictable.

    Every `leak_NN` sits one hop from `affect_head`, so all 12 tie at distance
    1. The rule is document order: the target listed first in the group wins,
    and within that, the earliest `## flow` row.
    """
    n = 12
    extra_layers = "".join(
        f"\n## layer leak_{i:02d}\n- op: Linear(n_affect, d_model)\n" for i in range(n)
    )
    extra_flow = "".join(
        f"| affect_head | leak_{i:02d} | a{i:02d} |\n| leak_{i:02d} | modulate | g{i:02d} |\n"
        for i in range(n)
    )
    # Group members deliberately listed in reverse of the flow-row order, so a
    # naive "first row wins" implementation would disagree with the spec.
    members = ", ".join(f"leak_{i:02d}" for i in reversed(range(n)))
    groups = _GROUPS + f"""
| leaks  | {members} |
"""
    reports = [
        check(["unreachable: leaks from core"],
              groups=groups, extra_layers=extra_layers, extra_flow=extra_flow)
        for _ in range(6)
    ]
    paths = {tuple(r.properties[0].path) for r in reports}
    assert len(paths) == 1, f"non-deterministic witness: {paths}"
    # Group member order decides the tie, not flow-row order.
    assert reports[0].properties[0].path == ["affect_head", f"leak_{n - 1:02d}"]


def test_wide_fan_in_add_still_resolves_a_shortest_witness():
    """Many predecessors into one layer: the witness stays the shortest route."""
    n = 8
    extra_layers = "".join(
        f"\n## layer fan_{i:02d}\n- op: Linear(d_model, d_model)\n" for i in range(n)
    )
    extra_flow = "".join(
        f"| cortex_ln | fan_{i:02d} | f{i:02d} |\n| fan_{i:02d} | modulate | m{i:02d} |\n"
        for i in range(n)
    )
    report = check(
        ["within: modulate from cortex_ln <= 1", "reachable: modulate from cortex_ln"],
        extra_layers=extra_layers,
        extra_flow=extra_flow,
    )
    # The direct cortex_ln -> modulate edge is still the shortest route.
    assert report.valid, codes(report)
    assert report.properties[1].path == ["cortex_ln", "modulate"]


# --------------------------------------------------------------------------- #
#  Stage ordering
# --------------------------------------------------------------------------- #


def _no_property_diagnostics(report) -> None:
    for diag in report.errors + report.warnings:
        assert not diag.code.startswith("PROPERTY_"), diag.code
        assert not diag.code.startswith("GROUP_"), diag.code
    assert report.properties == []
    assert report.properties_checked == 0


def test_stage_6_skipped_silently_when_structural_fails():
    """A cycle would make `unreachable` fail everywhere — stay quiet instead."""
    extra_flow = "| head | cortex_ln | loop |\n"
    groups = """
## groups

| Name  | Layers           |
|-------|------------------|
| embed | embed, cortex_ln |
"""
    report = check(
        ["unreachable: cortex from core", "reachable: v_hat from obs"],
        groups=groups,
        extra_flow=extra_flow,
    )
    assert "CYCLE_DETECTED" in codes(report)
    _no_property_diagnostics(report)


def test_stage_6_skipped_silently_when_naming_fails():
    extra_flow = "| ghost_layer | head | g |\n"
    report = check(
        ["reachable: v_hat from obs", "just some prose"],
        extra_flow=extra_flow,
    )
    assert "UNKNOWN_LAYER_REFERENCE" in codes(report)
    _no_property_diagnostics(report)


def test_stage_6_still_runs_when_only_the_shape_stage_fails():
    extra_layers = """
## layer bad_add
- op: Add()
"""
    extra_flow = """| affect_head | bad_add | a  |
| cortex_ln   | bad_add | h  |
| bad_add     | modulate| ba |
"""
    report = check(
        ["unreachable: cortex from core"],
        extra_layers=extra_layers,
        extra_flow=extra_flow,
    )
    assert "SHAPE_MISMATCH" in codes(report)
    assert "PROPERTY_EXCLUSION_FAIL" in codes(report)


def test_out_dim_skipped_when_the_layer_has_no_inferred_shape():
    extra_layers = """
## layer mystery
- op: TotallyUnknownOp(7)
## layer mystery2
- op: Linear(d_model, d_model)
"""
    extra_flow = """| cortex_ln | mystery  | m  |
| mystery   | mystery2 | m2 |
| mystery2  | modulate | m3 |
"""
    report = check(
        ["out_dim: mystery <= 1"],
        extra_layers=extra_layers,
        extra_flow=extra_flow,
    )
    assert "PROPERTY_WIDTH_FAIL" not in codes(report)
    assert "UNKNOWN_OP" in wcodes(report)


def test_param_fraction_skipped_when_total_is_zero():
    src = """# architecture NoParams

## tensors

| Name | Shape  | Dtype   |
|------|--------|---------|
| a    | (B, 4) | float32 |
| b    | (B, 4) | float32 |

## layer a [input]
## layer act
- op: ReLU()
## layer b [output]

## groups

| Name | Layers |
|------|--------|
| g    | act    |

## flow

| Source | Target | Tensor |
|--------|--------|--------|
| a      | act    | a      |
| act    | b      | h      |

## properties
- param_fraction: g <= 0%
"""
    report = verify(parse(src)[0])
    assert report.valid, codes(report)
    assert report.properties_failed == 0


# --------------------------------------------------------------------------- #
#  Report / to_dict
# --------------------------------------------------------------------------- #


def test_report_to_dict_carries_properties_and_groups():
    report = check([
        "reachable: v_hat from obs",
        "unreachable: cortex from core",
    ])
    d = report.to_dict()
    assert d["groups"]["bus"] == ["nm_bus", "nm_receptor", "nm_gain"]
    props = d["properties"]
    assert props["checked"] == 2
    assert props["passed"] == 1
    assert props["failed"] == 1
    results = props["results"]
    assert results[0]["passed"] is True
    assert results[1]["code"] == "PROPERTY_EXCLUSION_FAIL"
    assert results[1]["path"][0] == "affect_head"
    # The dict must be JSON-serializable — the CLI and MCP both dump it.
    json.dumps(d)


def test_report_to_dict_shape_unchanged_when_no_properties():
    report = check(groups="")
    d = report.to_dict()
    assert d["properties"]["checked"] == 0
    assert d["groups"] == {}


# --------------------------------------------------------------------------- #
#  Mermaid
# --------------------------------------------------------------------------- #


def test_mermaid_emits_one_subgraph_per_group():
    out = compile_mermaid(parse(mini())[0])
    assert 'subgraph g_core["core"]' in out
    assert 'subgraph g_cortex["cortex"]' in out
    assert 'subgraph g_bus["bus"]' in out
    assert out.count("subgraph ") == 3
    assert out.count("\n    end") == 3


def test_mermaid_declares_grouped_nodes_inside_their_subgraph():
    out = compile_mermaid(parse(mini())[0])
    lines = out.splitlines()
    start = lines.index('    subgraph g_bus["bus"]')
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "end")
    block = "\n".join(lines[start:end])
    for member in ("nm_bus", "nm_receptor", "nm_gain"):
        assert f"{member}[" in block


def test_mermaid_keeps_ungrouped_nodes_and_all_edges():
    arch = parse(mini())[0]
    out = compile_mermaid(arch)
    # `obs`, `y`, `v_hat` belong to no group.
    assert "obs((" in out
    assert "v_hat((" in out
    for e in arch.flow:
        assert f"    {e.source} -- " in out or f"    {e.source} --> " in out


def test_mermaid_emits_a_doubly_grouped_layer_once_in_the_first_group():
    groups = """
## groups

| Name  | Layers                |
|-------|-----------------------|
| a     | nm_bus, nm_receptor   |
| b     | nm_receptor, nm_gain  |
"""
    out = compile_mermaid(parse(mini(groups=groups))[0])
    assert out.count("    nm_receptor[") == 1
    lines = out.splitlines()
    a_start = lines.index('    subgraph g_a["a"]')
    b_start = lines.index('    subgraph g_b["b"]')
    node = next(i for i, ln in enumerate(lines) if ln.startswith("        nm_receptor["))
    assert a_start < node < b_start


def test_mermaid_unchanged_without_groups():
    ungrouped = compile_mermaid(parse(mini(groups=""))[0])
    assert "subgraph" not in ungrouped
    # Every node declaration is at the original 4-space indent.
    for ln in ungrouped.splitlines():
        if ln.strip() and ln.startswith(" "):
            assert ln.startswith("    ") and not ln.startswith("     ")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "n_orca.cli.main", *args],
        cwd=REPO, capture_output=True, text=True, check=False,
    )


def test_cli_prints_the_property_summary_line(tmp_path):
    f = tmp_path / "p.n.orca.md"
    f.write_text(mini([
        "reachable: v_hat from obs",
        "unreachable: core from cortex",
        "passes_through: bus for core -> cortex",
    ]), encoding="utf-8")
    r = _cli("verify", str(f))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Properties: 3 checked, 0 failed" in r.stdout


def test_cli_reports_failures_in_the_summary_line(tmp_path):
    f = tmp_path / "p.n.orca.md"
    f.write_text(mini([
        "reachable: v_hat from obs",
        "unreachable: cortex from core",
    ]), encoding="utf-8")
    r = _cli("verify", str(f))
    assert r.returncode == 1
    assert "Properties: 2 checked, 1 failed" in r.stdout
    assert "affect_head -> nm_bus" in r.stdout


def test_cli_omits_the_property_line_without_properties(tmp_path):
    f = tmp_path / "p.n.orca.md"
    f.write_text(mini(), encoding="utf-8")
    r = _cli("verify", str(f))
    assert r.returncode == 0
    assert "Properties:" not in r.stdout


def test_cli_json_carries_properties_and_groups(tmp_path):
    f = tmp_path / "p.n.orca.md"
    f.write_text(mini(["unreachable: cortex from core"]), encoding="utf-8")
    r = _cli("verify", "--json", str(f))
    payload = json.loads(r.stdout)
    assert payload[0]["properties"]["failed"] == 1
    assert payload[0]["groups"]["core"][0] == "core_sense"


def test_cli_strict_promotes_group_overlap(tmp_path):
    groups = """
## groups

| Name  | Layers                |
|-------|-----------------------|
| a     | nm_bus, nm_receptor   |
| b     | nm_receptor, nm_gain  |
"""
    f = tmp_path / "p.n.orca.md"
    f.write_text(mini(groups=groups), encoding="utf-8")
    assert _cli("verify", str(f)).returncode == 0
    strict = _cli("verify", "--strict", str(f))
    assert strict.returncode == 1
    assert "GROUP_OVERLAP" in strict.stdout


# --------------------------------------------------------------------------- #
#  MCP
# --------------------------------------------------------------------------- #


def test_mcp_verify_markdown_carries_properties_and_groups():
    pytest.importorskip("mcp")
    from n_orca.mcp_server import verify_markdown

    fn = getattr(verify_markdown, "fn", verify_markdown)
    payload = fn(source=mini(["unreachable: cortex from core"]))
    arch = payload["architectures"][0]
    assert arch["properties"]["failed"] == 1
    assert arch["properties"]["results"][0]["code"] == "PROPERTY_EXCLUSION_FAIL"
    assert arch["groups"]["bus"] == ["nm_bus", "nm_receptor", "nm_gain"]


# --------------------------------------------------------------------------- #
#  Shipped examples
# --------------------------------------------------------------------------- #

POSITIVE = EXAMPLES / "subcortical-value-cortex.n.orca.md"
LEAK = EXAMPLES / "subcortical-value-cortex-leak.n.orca.md"
FULL = EXAMPLES / "subcortical-value-cortex-full.n.orca.md"


def test_positive_example_verifies_clean_with_all_six_kinds():
    arch = parse(POSITIVE.read_text(encoding="utf-8"))[0]
    report = verify(arch)
    assert report.valid, codes(report)
    assert report.properties_checked == 6
    assert report.properties_failed == 0
    assert {p.kind for p in arch.properties} == {
        "reachable", "unreachable", "passes_through",
        "within", "param_fraction", "out_dim",
    }
    assert {g.name for g in arch.groups} == {"core", "cortex", "bus"}


def test_positive_example_boundary_is_asymmetric():
    src = POSITIVE.read_text(encoding="utf-8")
    flipped = src.replace(
        "- unreachable: core from cortex",
        "- unreachable: cortex from core",
    )
    assert flipped != src
    report = verify(parse(flipped)[0])
    assert "PROPERTY_EXCLUSION_FAIL" in codes(report)


def test_leak_example_fails_for_exactly_the_right_reason():
    report = verify(parse(LEAK.read_text(encoding="utf-8"))[0])
    assert not report.valid
    assert "PROPERTY_EXCLUSION_FAIL" in codes(report)
    assert "cortex_ln -> critic_fc -> feedback_proj -> core_add" in only(
        report, "PROPERTY_EXCLUSION_FAIL"
    )
    # The file is well-formed: it is wrong *only* in the asserted boundary.
    assert set(codes(report)) == {"PROPERTY_EXCLUSION_FAIL"}


def test_leak_example_differs_from_the_positive_only_by_the_feedback_path():
    pos = POSITIVE.read_text(encoding="utf-8")
    leak = LEAK.read_text(encoding="utf-8")
    assert "critic_fc" not in pos
    assert "critic_fc" in leak
    assert "feedback_proj" in leak


def test_positive_example_mermaid_shows_the_boundary():
    out = compile_mermaid(parse(POSITIVE.read_text(encoding="utf-8"))[0])
    assert 'subgraph g_core["core"]' in out
    assert 'subgraph g_cortex["cortex"]' in out
    committed = (EXAMPLES / "subcortical-value-cortex.mmd").read_text(encoding="utf-8")
    assert committed.strip() == out.strip()


# --------------------------------------------------------------------------- #
#  Regression — pre-existing examples are untouched
# --------------------------------------------------------------------------- #

BASELINE = Path(__file__).parent / "data" / "example_verify_baseline.txt"
MERMAID_BASELINE = Path(__file__).parent / "data" / "example_mermaid_baseline.txt"


def _pre_existing_examples() -> list[Path]:
    # Files added after the baseline was frozen (dory1-4b declares groups).
    new = {POSITIVE.name, LEAK.name, FULL.name, "dory1-4b.n.orca.md"}
    files = sorted(EXAMPLES.glob("*.n.orca.md")) + sorted(
        (EXAMPLES / "hf-generated").glob("*.n.orca.md")
    )
    return [f for f in files if f.name not in new]


def _verify_text(path: Path) -> str:
    import contextlib
    import io

    from n_orca.cli.main import _print_human_report

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for arch in parse(path.read_text(encoding="utf-8")):
            _print_human_report(verify(arch))
    return buf.getvalue()


def test_pre_existing_examples_verify_identically():
    expected = BASELINE.read_text(encoding="utf-8")
    chunks = []
    for f in _pre_existing_examples():
        chunks.append(f"### {f.relative_to(REPO)}\n{_verify_text(f)}")
    assert "".join(chunks) == expected


def test_pre_existing_examples_compile_mermaid_identically():
    expected = MERMAID_BASELINE.read_text(encoding="utf-8")
    chunks = []
    for f in _pre_existing_examples():
        parts = [compile_mermaid(a) for a in parse(f.read_text(encoding="utf-8"))]
        chunks.append(f"### {f.relative_to(REPO)}\n" + "\n\n".join(parts) + "\n")
    assert "".join(chunks) == expected


def test_pre_existing_examples_declare_no_groups_or_properties():
    """The baseline only proves stability if these files really are untouched."""
    for f in _pre_existing_examples():
        for arch in parse(f.read_text(encoding="utf-8")):
            assert arch.groups == []
            assert arch.properties == []


# --------------------------------------------------------------------------
# The full architecture of record.
#
# POSITIVE is a minimal fixture: it exercises groups, the bus bottleneck and
# param_fraction, and its value head lives inside `core`. FULL is the design
# it was reduced from — the critic sits in the *cortex*, reads cortical
# latents, and regresses onto the core's affect signal. That bridge is the
# whole mechanism, so it gets its own guarded example.
# --------------------------------------------------------------------------


def _full():
    return parse(FULL.read_text(encoding="utf-8"))[0]


def test_full_example_verifies_clean():
    report = verify(_full())
    assert report.valid
    assert [p.raw for p in report.properties if not p.passed] == []
    assert len(report.properties) == 8


def test_full_example_critic_is_cortical_not_subcortical():
    """The bridge: the critic must read cortical latents, never core_act."""
    arch = _full()
    groups = {g.name: set(g.layers) for g in arch.groups}
    assert "critic_fc" in groups["critic"]
    assert not (groups["critic"] & groups["core"])
    sources = {e.source for e in arch.flow if e.target == "critic_fc"}
    assert sources == {"cortex_ln"}


def test_full_example_excludes_core_from_critic():
    """critic is the trainable module closest to the core signal — pin it."""
    props = {p.raw for p in verify(_full()).properties}
    assert "unreachable: core from critic" in props


def test_full_example_core_reads_only_privileged_inputs():
    arch = _full()
    core = next(set(g.layers) for g in arch.groups if g.name == "core")
    incoming = {e.source for e in arch.flow if e.target in core} - set(core)
    assert incoming == {"obs", "soma_in"}
