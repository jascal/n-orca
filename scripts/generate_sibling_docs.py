"""Generate docs/architectures/ in sae-forge and bio-sae.

For every model family the sibling project supports:
- Fetch the real config.json from Hugging Face (cached)
- Convert via n-orca to a verified .n.orca.md + Mermaid diagram
- Drop the files in <repo>/docs/architectures/<model>.{n.orca.md,mmd}
- Generate an index.md summarising parameter counts, layer counts,
  and the originating model_id.

Models that need an adapter n-orca does not yet ship (Whisper encoder-decoder,
Qwen3-MoE, etc.) get a stub entry in the index linking to the GitHub-issue-
style placeholder under <repo>/docs/architectures/_unsupported/.

Run from the n-orca repo root:
    .venv/bin/python scripts/generate_sibling_docs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from n_orca.hf import HfClient, HfClientError, convert, UnsupportedModelError
from n_orca.hf.adapters import get_adapter


@dataclass
class ModelTarget:
    label: str                        # filename slug, e.g. "gpt2-small"
    model_id: str                     # HF Hub id (canonical, used for provenance)
    fallback_config: dict | None = None
    note: str | None = None           # extra context for the index


@dataclass
class Generated:
    label: str
    model_id: str
    revision: str | None = None
    valid: bool = False
    params: int = 0
    layers: int = 0
    adapter: str | None = None
    error: str | None = None
    note: str | None = None


SAE_FORGE_MODELS: list[ModelTarget] = [
    ModelTarget("gpt2-small",  "openai-community/gpt2",
                note="OpenAI GPT-2 small — sae-forge's `Gpt2Adapter` reference"),
    ModelTarget("gpt2-medium", "openai-community/gpt2-medium",
                note="GPT-2 medium (24-layer)"),
    ModelTarget("llama-2-7b",  "NousResearch/Llama-2-7b-hf",
                fallback_config={
                    "_name_or_path": "meta-llama/Llama-2-7b-hf",
                    "model_type": "llama", "architectures": ["LlamaForCausalLM"],
                    "hidden_size": 4096, "num_hidden_layers": 32,
                    "num_attention_heads": 32, "intermediate_size": 11008,
                    "max_position_embeddings": 4096, "vocab_size": 32000,
                },
                note="LLaMA-2 7B — sae-forge's `LlamaAdapter` reference"),
    ModelTarget("gemma-2-2b",  "google/gemma-2-2b",
                fallback_config={
                    "_name_or_path": "google/gemma-2-2b",
                    "model_type": "gemma2", "architectures": ["Gemma2ForCausalLM"],
                    "hidden_size": 2304, "num_hidden_layers": 26,
                    "num_attention_heads": 8, "intermediate_size": 9216,
                    "max_position_embeddings": 8192, "vocab_size": 256000,
                },
                note="Gemma-2 2B — sae-forge's `Gemma2Adapter` reference"),
    ModelTarget("qwen2-7b",    "Qwen/Qwen2-7B",
                note="Qwen2 7B — sae-forge's `Qwen2Adapter` reference"),
    ModelTarget("qwen3-4b",    "Qwen/Qwen3-4B",
                fallback_config={
                    "_name_or_path": "Qwen/Qwen3-4B",
                    "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                    "hidden_size": 2560, "num_hidden_layers": 36,
                    "num_attention_heads": 32, "intermediate_size": 9728,
                    "max_position_embeddings": 32768, "vocab_size": 151936,
                },
                note="Qwen3 4B — sae-forge's `Qwen3Adapter` reference"),
    ModelTarget("esm2-650m",   "facebook/esm2_t33_650M_UR50D",
                note="ESM-2 650M — shared with bio-sae for protein SAE work"),
]

SAE_FORGE_UNSUPPORTED: list[ModelTarget] = [
    ModelTarget("whisper-base", "openai/whisper-base",
                note="Encoder-decoder speech model — n-orca v0 does not yet"
                     " model encoder-decoder topologies; would need a"
                     " `WhisperAdapter` covering log-mel CNN stem + encoder"
                     " + cross-attention decoder."),
    ModelTarget("qwen3-moe", "Qwen/Qwen3-30B-A3B",
                note="Qwen3 mixture-of-experts decoder — n-orca v0 represents"
                     " FFN as a dense block; a `MoeFeedForward` op (with `n_experts`"
                     " and `top_k` parameters) would be the minimum required addition."),
]

# bio-sae's `configs/` lists 8M and 650M. The full public ESM-2 ladder
# is documented here for completeness.
BIO_SAE_MODELS: list[ModelTarget] = [
    ModelTarget("esm2-t6-8m",     "facebook/esm2_t6_8M_UR50D",
                note="ESM-2 8M parameters (6 layers, d=320) — bio-sae's"
                     " `configs/esm2_t6_8M.yaml` reference. Smallest publicly"
                     " released ESM-2 — useful for fast iteration."),
    ModelTarget("esm2-t12-35m",   "facebook/esm2_t12_35M_UR50D",
                note="ESM-2 35M parameters (12 layers, d=480)."),
    ModelTarget("esm2-t30-150m",  "facebook/esm2_t30_150M_UR50D",
                note="ESM-2 150M parameters (30 layers, d=640)."),
    ModelTarget("esm2-t33-650m",  "facebook/esm2_t33_650M_UR50D",
                note="ESM-2 650M parameters (33 layers, d=1280) — bio-sae's"
                     " `configs/esm2_t33_650M_layer24.yaml` reference."),
    ModelTarget("esm2-t36-3b",    "facebook/esm2_t36_3B_UR50D",
                note="ESM-2 3B parameters (36 layers, d=2560)."),
    ModelTarget("esm2-t48-15b",   "facebook/esm2_t48_15B_UR50D",
                note="ESM-2 15B parameters (48 layers, d=5120) — largest released."),
]


def generate_for(repo: Path, models: list[ModelTarget], *,
                 client: HfClient, unsupported: list[ModelTarget] | None = None) -> list[Generated]:
    out_dir = repo / "docs" / "architectures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Generated] = []
    for m in models:
        rows.append(_generate_one(m, out_dir, client))

    if unsupported:
        stub_dir = out_dir / "_unsupported"
        stub_dir.mkdir(exist_ok=True)
        for m in unsupported:
            stub_path = stub_dir / f"{m.label}.md"
            stub_path.write_text(_stub_unsupported(m), encoding="utf-8")
            rows.append(Generated(
                label=m.label, model_id=m.model_id, valid=False,
                error="no n-orca adapter yet", note=m.note,
            ))

    _write_index(repo, rows)
    return rows


def _generate_one(target: ModelTarget, out_dir: Path, client: HfClient) -> Generated:
    config = None
    revision = None
    # Prefer the live Hub for an authoritative config.
    try:
        info = client.info(target.model_id, include_config=True)
        config = info.config or None
        revision = info.sha
    except HfClientError as ex:
        print(f"  [warn] {target.model_id}: hub fetch failed ({ex.__class__.__name__}); falling back", file=sys.stderr)
        config = None

    if not config:
        config = target.fallback_config
    if not config:
        return Generated(label=target.label, model_id=target.model_id,
                         error="no config available (Hub failed; no fallback)",
                         note=target.note)

    try:
        result = convert(config, name=_pascal(target.label))
    except UnsupportedModelError as ex:
        return Generated(label=target.label, model_id=target.model_id,
                         error=str(ex), note=target.note)

    md_path = out_dir / f"{target.label}.n.orca.md"
    mmd_path = out_dir / f"{target.label}.mmd"
    result.write_markdown(md_path)
    result.write_mermaid(mmd_path)

    adapter = get_adapter(config)
    return Generated(
        label=target.label,
        model_id=target.model_id,
        revision=revision,
        valid=result.report.valid,
        params=result.report.param_count,
        layers=len(result.architecture.layers),
        adapter=type(adapter).__name__ if adapter else None,
        note=target.note,
    )


def _pascal(s: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in s.replace("-", "_").split("_") if p) or "Model"


def _stub_unsupported(m: ModelTarget) -> str:
    return f"""# {m.label} — not yet supported by n-orca v0.1

**Model:** [`{m.model_id}`](https://huggingface.co/{m.model_id})

{m.note or ""}

Once an adapter is added under `n_orca/hf/adapters/`, regenerate this
directory with:

```bash
.venv/bin/python scripts/generate_sibling_docs.py
```
"""


def _write_index(repo: Path, rows: list[Generated]) -> None:
    out = repo / "docs" / "architectures" / "README.md"
    lines: list[str] = []
    lines.append(f"# {repo.name} — model architectures, in n-orca")
    lines.append("")
    lines.append(
        "Each architecture below has been converted from its Hugging Face"
        " `config.json` into a verified [n-orca](https://github.com/jascal/n-orca)"
        " document. The same conversion runs via the `n-orca` MCP server"
        " (registered with Claude Code as `n-orca`) — call the"
        " `convert_from_hf` tool with any HF model id to regenerate."
    )
    lines.append("")
    lines.append("Each entry has:")
    lines.append("")
    lines.append("- `<name>.n.orca.md` — the verified n-orca architecture document")
    lines.append("- `<name>.mmd` — a Mermaid flowchart of the architecture")
    lines.append("")
    lines.append("## Supported")
    lines.append("")
    lines.append("| Model | Hub id | n-orca adapter | Layers | Params | Document |")
    lines.append("|-------|--------|----------------|-------:|-------:|----------|")
    for r in rows:
        if r.error:
            continue
        params = f"{r.params:,}" if r.params else "—"
        rev = f" @ `{r.revision[:8]}`" if r.revision else ""
        lines.append(
            f"| **{r.label}** | [`{r.model_id}`](https://huggingface.co/{r.model_id}){rev} |"
            f" `{r.adapter or '—'}` | {r.layers:,} | {params} |"
            f" [`{r.label}.n.orca.md`]({r.label}.n.orca.md) /"
            f" [`{r.label}.mmd`]({r.label}.mmd) |"
        )

    unsupported_rows = [r for r in rows if r.error]
    if unsupported_rows:
        lines.append("")
        lines.append("## Not yet supported")
        lines.append("")
        lines.append(
            "These model families ship in the parent project's adapter list"
            " but n-orca v0.1 does not yet model their topology."
            " See [`_unsupported/`](_unsupported/) for per-model context."
        )
        lines.append("")
        lines.append("| Model | Hub id | Why |")
        lines.append("|-------|--------|-----|")
        for r in unsupported_rows:
            lines.append(
                f"| {r.label} | [`{r.model_id}`](https://huggingface.co/{r.model_id}) |"
                f" {r.note or r.error} |"
            )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Generated by `scripts/generate_sibling_docs.py` in"
                 " [n-orca](https://github.com/jascal/n-orca)."
                 " Re-run after upgrading n-orca to refresh._")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-forge", type=Path,
                        default=Path("/Users/allans/code/sae-forge"))
    parser.add_argument("--bio-sae", type=Path,
                        default=Path("/Users/allans/code/bio-sae"))
    args = parser.parse_args()

    client = HfClient()

    print(f"Generating sae-forge docs in {args.sae_forge}/docs/architectures/")
    rows = generate_for(args.sae_forge, SAE_FORGE_MODELS, client=client,
                        unsupported=SAE_FORGE_UNSUPPORTED)
    _print_summary("sae-forge", rows)

    print()
    print(f"Generating bio-sae docs in {args.bio_sae}/docs/architectures/")
    rows = generate_for(args.bio_sae, BIO_SAE_MODELS, client=client)
    _print_summary("bio-sae", rows)
    return 0


def _print_summary(project: str, rows: list[Generated]) -> None:
    print(f"\n  {project} — {len(rows)} entries")
    for r in rows:
        if r.error:
            print(f"    [stub]  {r.label:<22}  {r.model_id:<45}  ({r.error[:60]})")
        else:
            print(
                f"    [OK]    {r.label:<22}  {r.model_id:<45}"
                f"  adapter={r.adapter:<22}  params={r.params:>14,}  layers={r.layers:>4}"
            )


if __name__ == "__main__":
    sys.exit(main())
