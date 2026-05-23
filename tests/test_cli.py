"""End-to-end CLI tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cli(*args):
    """Run `python -m n_orca.cli.main` with `args` and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "n_orca.cli.main", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_version():
    result = _cli("--version")
    assert result.returncode == 0
    assert "n-orca" in result.stdout


def test_verify_simple_mlp():
    result = _cli("verify", "examples/simple-mlp.n.orca.md")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALID" in result.stdout
    assert "SimpleMLP" in result.stdout


def test_verify_json_output():
    result = _cli("verify", "examples/simple-mlp.n.orca.md", "--json")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data[0]["architecture"] == "SimpleMLP"
    assert data[0]["valid"] is True
    assert data[0]["param_count"] > 0


def test_compile_mermaid():
    result = _cli("compile", "mermaid", "examples/simple-mlp.n.orca.md")
    assert result.returncode == 0
    assert "flowchart TD" in result.stdout


def test_compile_pytorch():
    result = _cli("compile", "pytorch", "examples/simple-mlp.n.orca.md")
    assert result.returncode == 0
    assert "class SimpleMLP(nn.Module):" in result.stdout


def test_info_summary():
    result = _cli("info", "examples/simple-mlp.n.orca.md")
    assert result.returncode == 0
    assert "Architecture: SimpleMLP" in result.stdout
    assert "Hyperparameters:" in result.stdout
    assert "Layers:" in result.stdout


def test_verify_nonexistent_file_fails():
    result = _cli("verify", "examples/does-not-exist.n.orca.md")
    assert result.returncode != 0
