"""Render an `Architecture` AST back to canonical `.n.orca.md` markdown.

The output is guaranteed to round-trip through `parse()`. Tables are aligned
for readability — wide enough for the longest cell in each column.
"""
from __future__ import annotations

from n_orca.ast import Architecture, Hyperparameter, Tensor, Layer, FlowEdge


def render(arch: Architecture) -> str:
    """Render `arch` to markdown."""
    lines: list[str] = []
    lines.append(f"# architecture {arch.name}")
    if arch.description:
        lines.append("")
        lines.append(f"> {arch.description}")

    if arch.hyperparameters:
        lines.append("")
        lines.append("## hyperparameters")
        lines.append("")
        lines.extend(_render_table(
            ["Name", "Type", "Default"],
            [[hp.name, hp.type, _format_default(hp.default)] for hp in arch.hyperparameters],
        ))

    if arch.tensors:
        lines.append("")
        lines.append("## tensors")
        lines.append("")
        lines.extend(_render_table(
            ["Name", "Shape", "Dtype"],
            [[t.name, _render_shape(t.shape), t.dtype] for t in arch.tensors],
        ))

    for ly in arch.layers:
        lines.append("")
        lines.append(_render_layer_heading(ly))
        if ly.description:
            lines.append(f"> {ly.description}")
        if ly.op is not None:
            args = ", ".join(ly.op.args)
            lines.append(f"- op: {ly.op.name}({args})" if args else f"- op: {ly.op.name}()")
        if ly.declared_shape is not None:
            lines.append(f"- shape: {_render_shape(ly.declared_shape)}")
        if ly.params:
            parts = ", ".join(f"{k}: {v}" for k, v in ly.params.items())
            lines.append(f"- params: {{{parts}}}")

    if arch.flow:
        lines.append("")
        lines.append("## flow")
        lines.append("")
        lines.extend(_render_table(
            ["Source", "Target", "Tensor"],
            [[e.source, e.target, e.tensor] for e in arch.flow],
        ))

    if arch.invariants:
        lines.append("")
        lines.append("## invariants")
        for inv in arch.invariants:
            if inv.kind == "output_shape":
                lines.append(f"- output_shape: {_render_shape(tuple(inv.value))}")  # type: ignore[arg-type]
            else:
                lines.append(f"- {inv.kind} {inv.op} {inv.value}")

    if arch.verification_rules:
        lines.append("")
        lines.append("## verification rules")
        for rule in arch.verification_rules:
            lines.append(f"- {rule}")

    lines.append("")
    return "\n".join(lines)


def _render_layer_heading(layer: Layer) -> str:
    markers: list[str] = []
    if layer.is_input:
        markers.append("[input]")
    if layer.is_output:
        markers.append("[output]")
    suffix = " " + " ".join(markers) if markers else ""
    return f"## layer {layer.name}{suffix}"


def _render_shape(shape: tuple[str, ...]) -> str:
    return "(" + ", ".join(shape) + ")"


def _format_default(default) -> str:
    if default is None:
        return ""
    if isinstance(default, bool):
        return "true" if default else "false"
    return str(default)


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a pipe-delimited markdown table, padded for visual alignment."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def pad(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [pad(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        out.append(pad(row + [""] * (len(headers) - len(row))))
    return out
