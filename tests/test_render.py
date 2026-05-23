"""Round-trip tests: parse -> render -> parse must preserve structure."""
from __future__ import annotations

import glob

from n_orca.parser import parse, parse_file
from n_orca.render import render
from n_orca.verifier import verify


def test_render_roundtrips_all_examples():
    paths = sorted(glob.glob("examples/*.n.orca.md"))
    assert paths, "no examples found"
    for path in paths:
        for arch in parse_file(path):
            rendered = render(arch)
            [round_tripped] = parse(rendered)
            assert round_tripped.name == arch.name
            assert len(round_tripped.layers) == len(arch.layers)
            assert [(e.source, e.target, e.tensor) for e in round_tripped.flow] == [
                (e.source, e.target, e.tensor) for e in arch.flow
            ]
            r1 = verify(arch)
            r2 = verify(round_tripped)
            assert r1.param_count == r2.param_count
            assert r1.depth == r2.depth
            assert r1.valid == r2.valid


def test_render_aligned_table_format():
    src = """
# architecture A
## hyperparameters
| Name | Type | Default |
|------|------|---------|
| d | int | 4 |
## tensors
| Name | Shape | Dtype |
|------|-------|-------|
| x | (B, d) | float32 |
| y | (B, d) | float32 |
## layer x [input]
## layer y [output]
## flow
| Source | Target | Tensor |
|--------|--------|--------|
| x | y | z |
"""
    [arch] = parse(src)
    out = render(arch)
    # The table separator row uses dashes between pipes.
    assert "|------" in out or "|--" in out
    # Layer heading carries the marker.
    assert "## layer x [input]" in out
    assert "## layer y [output]" in out
