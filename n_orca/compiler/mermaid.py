"""Mermaid flowchart compiler.

Emits a `flowchart TD` (top-down) diagram with one node per layer. Tensor
flow edges carry the tensor name and (if available) the inferred shape, so
the diagram is self-documenting.

Declared `## groups` become `subgraph` blocks. That is the highest-leverage
part of property checking for a human reader: an asserted isolation boundary
stops being a claim in a table and becomes a box in the picture.
"""
from __future__ import annotations

from n_orca.ast import Architecture, Layer
from n_orca.verifier import verify


def compile_mermaid(arch: Architecture) -> str:
    """Compile `arch` to a Mermaid flowchart string."""
    report = verify(arch)
    shapes = report.inferred_shapes

    lines: list[str] = []
    lines.append(f"%% architecture {arch.name}")
    lines.append("flowchart TD")

    # Mermaid requires a node to be declared inside its subgraph before any
    # edge mentions it, so grouped nodes come first, then ungrouped ones, then
    # every edge. A layer listed in two groups is drawn in the first that
    # claims it — Mermaid subgraphs cannot overlap, and `GROUP_OVERLAP` has
    # already warned about the ambiguity.
    claimed: set[str] = set()
    for group in arch.groups:
        members = [
            name for name in group.layers
            if name not in claimed and arch.layer(name) is not None
        ]
        if not members:
            continue
        lines.append(f"    subgraph g_{_mid(group.name)}[{_quote(group.name)}]")
        for name in members:
            claimed.add(name)
            ly = arch.layer(name)
            if ly is not None:
                lines.append(f"        {_node(ly)}")
        lines.append("    end")

    for ly in arch.layers:
        if ly.name not in claimed:
            lines.append(f"    {_node(ly)}")

    for edge in arch.flow:
        src = _mid(edge.source)
        tgt = _mid(edge.target)
        out_shape = shapes.get(edge.source)
        shape_str = _format_shape(out_shape) if out_shape else ""
        label_parts = [p for p in (edge.tensor, shape_str) if p]
        if label_parts:
            label = " : ".join(label_parts)
            lines.append(f"    {src} -- \"{label}\" --> {tgt}")
        else:
            lines.append(f"    {src} --> {tgt}")

    # Style input/output nodes distinctly.
    inputs = [_mid(ly.name) for ly in arch.inputs()]
    outputs = [_mid(ly.name) for ly in arch.outputs()]
    if inputs:
        lines.append(f"    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;")
        lines.append(f"    class {','.join(inputs)} input;")
    if outputs:
        lines.append(f"    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;")
        lines.append(f"    class {','.join(outputs)} output;")

    return "\n".join(lines)


def _node(ly: Layer) -> str:
    """One node declaration: `id[label]`, or `id((label))` for an input/output."""
    node_id = _mid(ly.name)
    if ly.is_input:
        return f"{node_id}(({_quote(f'{ly.name}<br/>[input]')}))"
    if ly.is_output:
        return f"{node_id}(({_quote(f'{ly.name}<br/>[output]')}))"
    return f"{node_id}[{_quote(f'{ly.name}<br/>{_format_op(ly)}')}]"


def _mid(name: str) -> str:
    """Sanitize a layer name to a valid Mermaid node id."""
    return "".join(c if c.isalnum() else "_" for c in name)


def _quote(label: str) -> str:
    """Wrap a node label in double quotes for Mermaid v10+.

    Required when the label contains brackets, parens, commas, or anything
    that the flowchart grammar might otherwise interpret as syntax.
    Embedded double quotes are encoded with the HTML entity Mermaid accepts.
    """
    escaped = label.replace('"', "&quot;")
    return f'"{escaped}"'


def _format_op(layer) -> str:
    if layer.op is None:
        return "&nbsp;"
    if not layer.op.args:
        return f"{layer.op.name}()"
    return f"{layer.op.name}({', '.join(layer.op.args)})"


def _format_shape(shape: tuple[str, ...]) -> str:
    return "(" + ",".join(shape) + ")"
