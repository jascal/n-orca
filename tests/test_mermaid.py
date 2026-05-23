"""Tests for the Mermaid compiler."""
from __future__ import annotations

from n_orca.compiler import compile_mermaid
from n_orca.parser import parse_file


def test_mermaid_renders_for_all_examples():
    """Every shipped example must compile to a non-empty Mermaid diagram."""
    import glob
    from pathlib import Path

    examples = sorted(glob.glob("examples/*.n.orca.md"))
    assert examples, "no examples found"
    for path in examples:
        archs = parse_file(path)
        for arch in archs:
            mermaid = compile_mermaid(arch)
            assert "flowchart TD" in mermaid
            for ly in arch.layers:
                # Sanitized layer id must appear at least once.
                assert any(
                    ly.name in line or _mid(ly.name) in line
                    for line in mermaid.splitlines()
                ), f"layer {ly.name} missing in mermaid output for {path}"


def _mid(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def test_mermaid_marks_input_and_output_classes():
    archs = parse_file("examples/simple-mlp.n.orca.md")
    mermaid = compile_mermaid(archs[0])
    assert "classDef input" in mermaid
    assert "classDef output" in mermaid


def test_mermaid_includes_inferred_shapes_on_edges():
    archs = parse_file("examples/simple-mlp.n.orca.md")
    mermaid = compile_mermaid(archs[0])
    # Edge labels include tensor name + a (...) shape annotation.
    assert "(B,in_dim)" in mermaid or "(B, in_dim)" in mermaid or "B,in_dim" in mermaid


def test_mermaid_node_labels_are_quoted_for_v10_grammar():
    """Regression: Mermaid v10+ rejects `[input]` inside unquoted node labels.

    Without quoting, `x((x<br/>[input]))` parses as opening a new SQS shape
    inside an existing double-circle. The compiler must wrap labels in
    double-quotes so brackets, parens, commas etc. are treated as text.
    """
    archs = parse_file("examples/simple-mlp.n.orca.md")
    mermaid = compile_mermaid(archs[0])
    # Input node should be wrapped: x(("x<br/>[input]"))
    assert '(("x<br/>[input]"))' in mermaid
    # Output node likewise.
    assert '(("y<br/>[output]"))' in mermaid
    # Op nodes carry parens in their op() form — must also be quoted.
    assert '"fc1<br/>Linear(in_dim, hidden)"' in mermaid
