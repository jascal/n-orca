"""CLI tests for `n-orca hf` subcommands."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cli(*args, stdin: str | None = None):
    return subprocess.run(
        [sys.executable, "-m", "n_orca.cli.main", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        input=stdin,
    )


def test_hf_convert_from_stdin_writes_markdown(tmp_path):
    cfg = {
        "model_type": "gpt2",
        "n_embd": 64, "n_layer": 2, "n_head": 4, "n_inner": 128,
        "n_positions": 32, "vocab_size": 1000,
    }
    out = tmp_path / "model.n.orca.md"
    mmd = tmp_path / "model.mmd"
    result = _cli(
        "hf", "convert", "-",
        "--out", str(out),
        "--mermaid", str(mmd),
        stdin=json.dumps(cfg),
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert mmd.exists()
    text = out.read_text(encoding="utf-8")
    assert "# architecture" in text
    assert "## layer input_ids [input]" in text
    assert "flowchart TD" in mmd.read_text(encoding="utf-8")


def test_hf_convert_pipes_to_stdout():
    cfg = '{"model_type": "bert", "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4, "intermediate_size": 64, "max_position_embeddings": 16, "vocab_size": 100}'
    result = _cli("hf", "convert", "-", "--no-verify", stdin=cfg)
    assert result.returncode == 0, result.stderr
    assert "# architecture" in result.stdout
    assert "## layer input_ids [input]" in result.stdout


def test_hf_convert_unsupported_model_fails():
    result = _cli("hf", "convert", "-", stdin='{"model_type": "fake_arch_v99"}')
    assert result.returncode != 0
    assert "no adapter" in result.stderr.lower() or "error" in result.stderr.lower()


def test_hf_convert_name_override():
    cfg = '{"model_type": "gpt2", "n_embd": 32, "n_layer": 1, "n_head": 4, "n_inner": 64, "n_positions": 16, "vocab_size": 100}'
    result = _cli("hf", "convert", "-", "--name", "MyOverride",
                  "--no-verify", stdin=cfg)
    assert result.returncode == 0
    assert "# architecture MyOverride" in result.stdout
