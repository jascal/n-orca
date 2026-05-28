"""N-Orca MCP server.

Exposes n-orca's verifier, compilers, and Hugging Face integration as
tools an LLM client (e.g. Claude Code) can invoke directly.

Run with `python -m n_orca.mcp_server` over stdio. Register with Claude:

    claude mcp add n-orca \\
        /Users/allans/code/n-orca/.venv/bin/python -m n_orca.mcp_server

Tools exposed:

- list_architectures      — registered HF adapters and what they cover
- hf_search               — search the Hugging Face Hub
- hf_info                 — model metadata + config.json + adapter dispatch
- convert_from_hf         — HF model_id → verified .n.orca.md + Mermaid
- verify_markdown         — verify a .n.orca.md string or file
- compile_mermaid         — emit a Mermaid diagram
- compile_pytorch         — emit a runnable nn.Module
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from n_orca import __version__, sae as _sae, world_models as _wm
from n_orca.compiler import compile_mermaid as _compile_mermaid, compile_pytorch as _compile_pytorch
from n_orca.hf import HfClient, HfClientError, convert as _convert, UnsupportedModelError
from n_orca.hf.adapters import list_adapters
from n_orca.parser import parse, parse_file, ParseError
from n_orca.render import render
from n_orca.verifier import verify

app = FastMCP("n-orca")


@app.tool()
def list_architectures() -> dict[str, Any]:
    """List all registered HF adapters and the model families they cover.

    Use this first when asked which Hugging Face models n-orca can convert.
    """
    adapters = []
    for a in list_adapters():
        adapters.append({
            "adapter": type(a).__name__,
            "model_types": list(a.model_types),
        })
    return {"version": __version__, "adapters": adapters}


@app.tool()
def hf_search(
    query: str | None = None,
    task: str | None = None,
    library: str | None = None,
    limit: int = 20,
    sort: str = "downloads",
) -> dict[str, Any]:
    """Search the Hugging Face Hub.

    Args:
        query: free-text search (e.g. "gpt2", "esm2 protein")
        task: pipeline_tag filter (e.g. "text-generation", "fill-mask")
        library: library filter (e.g. "transformers")
        limit: max results
        sort: "downloads" | "likes" | "lastModified"
    """
    try:
        results = HfClient().search(
            query=query, task=task, library=library, limit=limit, sort=sort,
        )
    except HfClientError as ex:
        return {"error": str(ex)}
    return {
        "results": [
            {
                "id": r.id,
                "pipeline_tag": r.pipeline_tag,
                "library_name": r.library_name,
                "downloads": r.downloads,
                "likes": r.likes,
                "tags": r.tags[:10],
            }
            for r in results
        ]
    }


@app.tool()
def hf_info(
    model_id: str,
    revision: str | None = None,
    include_config: bool = True,
) -> dict[str, Any]:
    """Show metadata for a Hub model: revision, files, config.json, and whether
    n-orca has an adapter that can convert it."""
    try:
        info = HfClient().info(model_id, revision=revision, include_config=include_config)
    except HfClientError as ex:
        return {"error": str(ex)}
    from n_orca.hf.adapters import get_adapter
    adapter = get_adapter(info.config) if info.config else None
    return {
        "id": info.id,
        "sha": info.sha,
        "pipeline_tag": info.pipeline_tag,
        "library_name": info.library_name,
        "downloads": info.downloads,
        "likes": info.likes,
        "last_modified": info.last_modified,
        "tags": info.tags[:15],
        "siblings": info.siblings[:25],
        "config": info.config,
        "supported_by": type(adapter).__name__ if adapter else None,
    }


@app.tool()
def convert_from_hf(
    model_id: str,
    revision: str | None = None,
    name: str | None = None,
    out_markdown: str | None = None,
    out_mermaid: str | None = None,
) -> dict[str, Any]:
    """Convert a Hugging Face model to a verified n-orca architecture document.

    Reads `config.json` only — no weights or model code are executed.

    Args:
        model_id: HF Hub model id (e.g. "gpt2", "meta-llama/Llama-2-7b-hf").
                  May also be a path to a local config.json.
        revision: optional commit sha / tag / branch to pin
        name: override the architecture name in the document
        out_markdown: if set, write the markdown to this path
        out_mermaid: if set, write the Mermaid diagram to this path

    Returns the rendered markdown, mermaid string, verification report,
    and (if requested) the paths written.
    """
    try:
        result = _convert(model_id, revision=revision, name=name)
    except (UnsupportedModelError, HfClientError) as ex:
        return {"error": str(ex)}

    written: dict[str, str] = {}
    if out_markdown:
        path = result.write_markdown(out_markdown)
        written["markdown"] = str(path)
    if out_mermaid:
        path = result.write_mermaid(out_mermaid)
        written["mermaid"] = str(path)

    return {
        "architecture_name": result.architecture.name,
        "markdown": result.markdown,
        "mermaid": result.mermaid,
        "report": result.report.to_dict(),
        "written": written,
        "model_id": result.model_id,
        "revision": result.revision,
    }


@app.tool()
def verify_markdown(
    source: str | None = None,
    path: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Parse and verify a `.n.orca.md` document.

    Provide either `source` (the markdown text) or `path` (a file on disk).
    """
    try:
        if path:
            archs = parse_file(path)
        elif source is not None:
            archs = parse(source)
        else:
            return {"error": "either `source` or `path` is required"}
    except ParseError as ex:
        return {"error": f"parse error: {ex}"}

    return {"architectures": [verify(a, strict=strict).to_dict() for a in archs]}


@app.tool()
def compile_mermaid(
    source: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Compile a `.n.orca.md` document to a Mermaid flowchart."""
    try:
        archs = _resolve(source=source, path=path)
    except ParseError as ex:
        return {"error": f"parse error: {ex}"}
    except FileNotFoundError as ex:
        return {"error": str(ex)}
    return {"mermaid": [_compile_mermaid(a) for a in archs]}


@app.tool()
def compile_pytorch(
    source: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Compile a `.n.orca.md` document to a PyTorch nn.Module source string."""
    try:
        archs = _resolve(source=source, path=path)
    except ParseError as ex:
        return {"error": f"parse error: {ex}"}
    except FileNotFoundError as ex:
        return {"error": str(ex)}
    return {"pytorch": [_compile_pytorch(a) for a in archs]}


@app.tool()
def render_markdown(source: str | None = None, path: str | None = None) -> dict[str, Any]:
    """Round-trip a `.n.orca.md` document through the canonical renderer.

    Useful for cleaning up hand-written markdown into the canonical form.
    """
    try:
        archs = _resolve(source=source, path=path)
    except ParseError as ex:
        return {"error": f"parse error: {ex}"}
    except FileNotFoundError as ex:
        return {"error": str(ex)}
    return {"markdown": [render(a) for a in archs]}


@app.tool()
def build_sae(
    variant: str,
    input_dim: int = 768,
    n_features: int = 16384,
    k: int = 64,
    l1_coeff: float = 1e-2,
    theta_init: float = 0.1,
    l0_coeff: float = 5e-3,
    n_heads: int = 4,
    attn_dropout: float = 0.0,
    n_labels: int = 100,
    aux_weight: float = 0.1,
    name: str | None = None,
    out_markdown: str | None = None,
    out_mermaid: str | None = None,
) -> dict[str, Any]:
    """Build a sparse-autoencoder architecture and verify it.

    Args:
        variant: "topk" | "l1" | "jumprelu" | "attn_topk" | "supervised_topk" | "gated"
        input_dim: the activation dim the SAE reconstructs
        n_features: dictionary width (typical: 16k–1M, expansion factor 4-32x)
        k: top-k threshold ("topk", "attn_topk", "supervised_topk")
        l1_coeff: L1 penalty coefficient ("l1" only — metadata)
        theta_init: initial threshold ("jumprelu" only)
        l0_coeff: L0 penalty coefficient ("jumprelu" only — metadata)
        n_heads: MultiHeadAttention head count ("attn_topk" only)
        attn_dropout: attention dropout ("attn_topk" only)
        n_labels: classifier output dim ("supervised_topk" only)
        aux_weight: auxiliary loss weight ("supervised_topk" only — metadata)
        name: override architecture name
        out_markdown / out_mermaid: optional file paths to write

    The "attn_topk" variant prepends a MultiHeadAttention block (with residual
    + LayerNorm) before the SAE encoder, so each position can see cross-
    sequence context. Its input/output tensors are 3D `(B, T, input_dim)`
    instead of 2D — consumers must feed per-token / per-residue / per-agent
    activations, NOT pooled. Mirrors econ-sae's AttnWorldModel structural
    unlock (Phase 1.6: conjunctive mAUC 0.84 → 0.97).

    The "supervised_topk" variant adds an auxiliary per-label classifier head
    off the sparse latents; the compiled forward returns `(x_hat, y_logits)`.
    Joint reconstruction + BCE-on-labels loss (loss weight is metadata).
    Mirrors econ-sae Phase 5.1's regime-supervised SAE (regime mAUC
    0.885 → 0.972 — biggest single-phase jump in econ-sae's journey).
    """
    variant = (variant or "").lower()
    builder_kw: dict[str, Any] = {"input_dim": input_dim, "n_features": n_features}
    if name:
        builder_kw["name"] = name
    if variant == "topk":
        builder_kw["k"] = k
        arch = _sae.topk_sae(**builder_kw)
    elif variant == "l1":
        builder_kw["l1_coeff"] = l1_coeff
        arch = _sae.l1_sae(**builder_kw)
    elif variant == "jumprelu":
        builder_kw["theta_init"] = theta_init
        builder_kw["l0_coeff"] = l0_coeff
        arch = _sae.jumprelu_sae(**builder_kw)
    elif variant in ("attn_topk", "attn-topk"):
        builder_kw["k"] = k
        builder_kw["n_heads"] = n_heads
        builder_kw["attn_dropout"] = attn_dropout
        arch = _sae.attn_topk_sae(**builder_kw)
    elif variant in ("supervised_topk", "supervised-topk", "sup_topk"):
        builder_kw["k"] = k
        builder_kw["n_labels"] = n_labels
        builder_kw["aux_weight"] = aux_weight
        arch = _sae.supervised_topk_sae(**builder_kw)
    elif variant in ("gated", "gated_sae"):
        arch = _sae.gated_sae(**builder_kw)
    else:
        return {"error": (
            f"unknown variant {variant!r}; expected 'topk' | 'l1' | "
            f"'jumprelu' | 'attn_topk' | 'supervised_topk' | 'gated'"
        )}

    return _materialise(arch, out_markdown=out_markdown, out_mermaid=out_mermaid)


@app.tool()
def build_world_model(
    variant: str = "baseline",
    input_dim: int = 43,
    h1_dim: int = 96,
    h2_dim: int = 48,
    hidden_dims: list[int] | None = None,
    embed_dim: int = 64,
    n_heads: int = 4,
    out_dim: int = 23,
    dropout: float = 0.0,
    name: str | None = None,
    out_markdown: str | None = None,
    out_mermaid: str | None = None,
) -> dict[str, Any]:
    """Build one of econ-sae's non-LLM world models.

    Args:
        variant: "baseline" (2-hidden-layer MLP) | "deep" (N-hidden-layer MLP)
                 | "attn" (per-agent self-attention + MLP)
        input_dim / out_dim: per-agent state dimensionality (econ-sae default 43 / 23)
        h1_dim / h2_dim: MLP widths (baseline & attn variants)
        hidden_dims: list of hidden widths (deep variant only)
        embed_dim / n_heads / dropout: attention variant only
        name: override architecture name
    """
    variant = (variant or "").lower()
    common: dict[str, Any] = {"input_dim": input_dim, "out_dim": out_dim}
    if name:
        common["name"] = name
    if variant in ("baseline", "world_model", "wm"):
        arch = _wm.world_model(h1_dim=h1_dim, h2_dim=h2_dim, **common)
    elif variant in ("deep", "deep_world_model"):
        dims = tuple(hidden_dims) if hidden_dims else (192, 128, 64)
        arch = _wm.deep_world_model(hidden_dims=dims, **common)
    elif variant in ("attn", "attn_world_model"):
        arch = _wm.attn_world_model(
            embed_dim=embed_dim, n_heads=n_heads,
            h1_dim=h1_dim, h2_dim=h2_dim, dropout=dropout, **common,
        )
    else:
        return {"error": f"unknown variant {variant!r}; expected 'baseline' | 'deep' | 'attn'"}

    return _materialise(arch, out_markdown=out_markdown, out_mermaid=out_mermaid)


def _materialise(arch, *, out_markdown: str | None, out_mermaid: str | None) -> dict[str, Any]:
    """Common SAE/world-model post-build: verify, render, optionally write."""
    rep = verify(arch)
    md = render(arch)
    mmd = _compile_mermaid(arch)
    written: dict[str, str] = {}
    if out_markdown:
        Path(out_markdown).write_text(md, encoding="utf-8")
        written["markdown"] = out_markdown
    if out_mermaid:
        Path(out_mermaid).write_text(mmd, encoding="utf-8")
        written["mermaid"] = out_mermaid
    return {
        "architecture_name": arch.name,
        "markdown": md,
        "mermaid": mmd,
        "report": rep.to_dict(),
        "written": written,
    }


def _resolve(*, source: str | None, path: str | None):
    if path:
        return parse_file(path)
    if source is not None:
        return parse(source)
    raise ValueError("either `source` or `path` is required")


def main() -> None:
    """Entry point for `python -m n_orca.mcp_server`."""
    app.run()


if __name__ == "__main__":
    main()
