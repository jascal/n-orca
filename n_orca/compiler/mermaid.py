"""Mermaid flowchart compiler.

Emits a `flowchart TD` (top-down) diagram with one node per layer. Tensor
flow edges carry the tensor name and (if available) the inferred shape, so
the diagram is self-documenting.
"""
from __future__ import annotations

from n_orca.ast import Architecture
from n_orca.verifier import verify


def compile_mermaid(arch: Architecture) -> str:
    """Compile `arch` to a Mermaid flowchart string."""
    report = verify(arch)
    shapes = report.inferred_shapes

    lines: list[str] = []
    lines.append(f"%% architecture {arch.name}")
    lines.append("flowchart TD")

    for ly in arch.layers:
        node_id = _mid(ly.name)
        if ly.is_input:
            label = _quote(f"{ly.name}<br/>[input]")
            shape = f"(({label}))"
        elif ly.is_output:
            label = _quote(f"{ly.name}<br/>[output]")
            shape = f"(({label}))"
        else:
            op_str = _format_op(ly)
            label = _quote(f"{ly.name}<br/>{op_str}")
            shape = f"[{label}]"
        lines.append(f"    {node_id}{shape}")

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
