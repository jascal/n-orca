"""Smoke tests for the n-orca MCP server.

These tests speak the real MCP protocol over stdio against the n-orca
server (no network — the tools that touch HF Hub aren't called here).
The mcp package is an optional dependency, so the whole module skips
when it isn't installed.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

HAS_MCP = importlib.util.find_spec("mcp") is not None
pytestmark = pytest.mark.skipif(not HAS_MCP, reason="mcp not installed")


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(coro):
    return asyncio.run(coro)


async def _connect():
    """Yield an initialized MCP ClientSession against `n_orca.mcp_server`."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import sys

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "n_orca.mcp_server"],
    )
    return params


async def _with_session(call):
    """Helper: open an MCP session, invoke `call(session)`, return its result."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = await _connect()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await call(session)


def test_list_tools_includes_core_set():
    async def _go(session):
        tools = await session.list_tools()
        return {t.name for t in tools.tools}

    names = _run(_with_session(_go))
    expected = {
        "list_architectures", "hf_search", "hf_info", "convert_from_hf",
        "verify_markdown", "compile_mermaid", "compile_pytorch", "render_markdown",
    }
    assert expected.issubset(names)


def test_list_architectures_reports_registered_adapters():
    async def _go(session):
        result = await session.call_tool("list_architectures", {})
        return json.loads(result.content[0].text)

    data = _run(_with_session(_go))
    names = {a["adapter"] for a in data["adapters"]}
    assert {"Gpt2Adapter", "LlamaFamilyAdapter", "BertAdapter", "EsmAdapter"}.issubset(names)


def test_verify_markdown_path():
    async def _go(session):
        result = await session.call_tool("verify_markdown", {
            "path": str(REPO_ROOT / "examples" / "simple-mlp.n.orca.md"),
        })
        return json.loads(result.content[0].text)

    data = _run(_with_session(_go))
    arch = data["architectures"][0]
    assert arch["valid"] is True
    assert arch["param_count"] == 203_530


def test_compile_mermaid_via_mcp():
    async def _go(session):
        result = await session.call_tool("compile_mermaid", {
            "path": str(REPO_ROOT / "examples" / "simple-mlp.n.orca.md"),
        })
        return json.loads(result.content[0].text)

    data = _run(_with_session(_go))
    assert "flowchart TD" in data["mermaid"][0]


def test_convert_from_hf_with_inline_config_via_mcp(tmp_path):
    """Use a tiny canned GPT-2 config so the tool runs without network."""
    # The convert_from_hf tool only accepts a model_id string, not a dict.
    # However the verify+render path can be exercised via verify_markdown.
    # We separately confirm convert_from_hf accepts a path to local config.json.
    cfg = {
        "model_type": "gpt2",
        "n_embd": 32, "n_layer": 1, "n_head": 4, "n_inner": 64,
        "n_positions": 16, "vocab_size": 100,
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    async def _go(session):
        result = await session.call_tool("convert_from_hf", {
            "model_id": str(cfg_path),
            "out_markdown": str(tmp_path / "out.n.orca.md"),
        })
        return json.loads(result.content[0].text)

    data = _run(_with_session(_go))
    assert "error" not in data, data.get("error")
    assert data["report"]["valid"] is True
    assert (tmp_path / "out.n.orca.md").exists()


def test_build_world_model_mot_with_timestep_dim():
    """Test MCP build_world_model for mot variant with custom timestep_dim (nit from PR #8 review)."""
    async def _go(session):
        result = await session.call_tool("build_world_model", {
            "variant": "mot",
            "embed_dim": 32,
            "n_heads": 2,
            "timestep_dim": 16,
            "h1_dim": 32,
            "h2_dim": 16,
        })
        return json.loads(result.content[0].text)

    data = _run(_with_session(_go))
    assert "error" not in data, data.get("error")
    assert data["report"]["valid"] is True
    assert data["architecture_name"] == "MoTDenoiseStep"
