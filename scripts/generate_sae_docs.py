"""Generate docs/architectures/ in polygram, sm-sae, and econ-sae.

For each project, ship the SAE variants the project trains or consumes,
plus (for econ-sae) the non-LLM world-model substrates the SAE reads from.

All architectures verify clean and ship as `.n.orca.md` + `.mmd` pairs,
linked by a per-project index README.

Run from the n-orca repo root:
    .venv/bin/python scripts/generate_sae_docs.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from n_orca import sae, world_models
from n_orca.compiler import compile_mermaid
from n_orca.render import render
from n_orca.verifier import verify


@dataclass
class DocEntry:
    slug: str
    title: str
    builder_doc: str         # how the user would reproduce this
    arch: object             # Architecture
    valid: bool = False
    params: int = 0
    layers: int = 0


def _build(label: str, slug: str, builder_doc: str, arch) -> DocEntry:
    rep = verify(arch)
    return DocEntry(
        slug=slug,
        title=label,
        builder_doc=builder_doc,
        arch=arch,
        valid=rep.valid,
        params=rep.param_count,
        layers=len(arch.layers),
    )


def polygram_entries() -> list[DocEntry]:
    """Polygram consumes SAEs trained externally; document the variants it
    can re-encode via its quantum-inspired rungs."""
    return [
        _build(
            "Top-k SAE (GPT-2 small substrate)",
            "sae-topk-gpt2",
            "n_orca.sae.topk_sae(input_dim=768, n_features=16384, k=64)",
            sae.topk_sae(input_dim=768, n_features=16384, k=64, name="TopKSae_GPT2"),
        ),
        _build(
            "Top-k SAE (Pythia 70m substrate)",
            "sae-topk-pythia",
            "n_orca.sae.topk_sae(input_dim=512, n_features=8192, k=32)",
            sae.topk_sae(input_dim=512, n_features=8192, k=32, name="TopKSae_Pythia"),
        ),
        _build(
            "JumpReLU SAE (Gemma 2 substrate)",
            "sae-jumprelu-gemma",
            "n_orca.sae.jumprelu_sae(input_dim=2304, n_features=16384)",
            sae.jumprelu_sae(input_dim=2304, n_features=16384, name="JumpReLUSae_Gemma2"),
        ),
        _build(
            "L1 SAE (LLaMA-2 substrate)",
            "sae-l1-llama",
            "n_orca.sae.l1_sae(input_dim=4096, n_features=32768)",
            sae.l1_sae(input_dim=4096, n_features=32768, name="L1Sae_Llama2"),
        ),
    ]


def sm_sae_entries() -> list[DocEntry]:
    """sm-sae trains three SAE variants over the Standard-Model tensor bundle
    (61 particles × 9-dim embedding) and a tiny GPT-2 cascade host."""
    return [
        _build(
            "Top-k SAE (SM embedding substrate)",
            "sae-topk-sm",
            "n_orca.sae.topk_sae(input_dim=61, n_features=256, k=8)",
            sae.topk_sae(input_dim=61, n_features=256, k=8, name="TopKSae_SM"),
        ),
        _build(
            "L1 SAE (SM embedding substrate)",
            "sae-l1-sm",
            "n_orca.sae.l1_sae(input_dim=61, n_features=256)",
            sae.l1_sae(input_dim=61, n_features=256, name="L1Sae_SM"),
        ),
        _build(
            "JumpReLU SAE (SM embedding substrate)",
            "sae-jumprelu-sm",
            "n_orca.sae.jumprelu_sae(input_dim=61, n_features=256)",
            sae.jumprelu_sae(input_dim=61, n_features=256, name="JumpReLUSae_SM"),
        ),
    ]


def econ_sae_entries() -> list[DocEntry]:
    """econ-sae trains the same three SAE variants on the H1 activations of
    a small world-model MLP. Document both layers of the stack."""
    return [
        # World-model substrates.
        _build(
            "WorldModel (baseline 2-hidden-layer MLP)",
            "world-model",
            "n_orca.world_models.world_model(input_dim=43, h1_dim=96, h2_dim=48, out_dim=23)",
            world_models.world_model(),
        ),
        _build(
            "DeepWorldModel (3-hidden-layer MLP)",
            "deep-world-model",
            "n_orca.world_models.deep_world_model(input_dim=43, hidden_dims=(192, 128, 64), out_dim=23)",
            world_models.deep_world_model(),
        ),
        _build(
            "AttnWorldModel (per-agent self-attention)",
            "attn-world-model",
            "n_orca.world_models.attn_world_model(input_dim=43, embed_dim=64, n_heads=4,"
            " h1_dim=192, h2_dim=128, out_dim=23)",
            world_models.attn_world_model(),
        ),
        # SAEs trained on H1 activations of WorldModel (h1_dim=96).
        _build(
            "Top-k SAE (over WorldModel H1)",
            "sae-topk-econ",
            "n_orca.sae.topk_sae(input_dim=96, n_features=512, k=16)",
            sae.topk_sae(input_dim=96, n_features=512, k=16, name="TopKSae_Econ"),
        ),
        _build(
            "L1 SAE (over WorldModel H1)",
            "sae-l1-econ",
            "n_orca.sae.l1_sae(input_dim=96, n_features=512)",
            sae.l1_sae(input_dim=96, n_features=512, name="L1Sae_Econ"),
        ),
        _build(
            "JumpReLU SAE (over WorldModel H1)",
            "sae-jumprelu-econ",
            "n_orca.sae.jumprelu_sae(input_dim=96, n_features=512)",
            sae.jumprelu_sae(input_dim=96, n_features=512, name="JumpReLUSae_Econ"),
        ),
    ]


def write_project(repo: Path, intro: str, entries: list[DocEntry]) -> None:
    out_dir = repo / "docs" / "architectures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for e in entries:
        (out_dir / f"{e.slug}.n.orca.md").write_text(render(e.arch), encoding="utf-8")
        (out_dir / f"{e.slug}.mmd").write_text(compile_mermaid(e.arch), encoding="utf-8")

    index = out_dir / "README.md"
    lines: list[str] = []
    lines.append(f"# {repo.name} — model architectures, in n-orca")
    lines.append("")
    lines.append(intro)
    lines.append("")
    lines.append(
        "Each entry has a verified [n-orca](https://github.com/jascal/n-orca)"
        " architecture document (`.n.orca.md`) and a Mermaid flowchart (`.mmd`)."
        " Regenerate with `python scripts/generate_sae_docs.py` from the n-orca repo,"
        " or call the `build_sae` / `build_world_model` tools on the registered"
        " `n-orca` MCP server."
    )
    lines.append("")
    lines.append("| Architecture | Builder | Layers | Params | Document |")
    lines.append("|--------------|---------|-------:|-------:|----------|")
    for e in entries:
        status = "" if e.valid else " ⚠"
        lines.append(
            f"| **{e.title}**{status} | `{e.builder_doc}` | {e.layers} | "
            f"{e.params:,} | [`{e.slug}.n.orca.md`]({e.slug}.n.orca.md) /"
            f" [`{e.slug}.mmd`]({e.slug}.mmd) |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Generated by `scripts/generate_sae_docs.py` in"
                 " [n-orca](https://github.com/jascal/n-orca)._")
    lines.append("")
    index.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {len(entries)} entries to {out_dir}")


PROJECT_INTROS = {
    "polygram": (
        "[polygram](https://github.com/jascal/polygram) re-encodes feature"
        " dictionaries from publicly available SAEs (via SAE-Lens, Anthropic"
        " releases, etc.) using quantum-inspired MPS / HEA rungs. The SAE"
        " variants below are the most common upstream sources — they describe"
        " the **input** topology polygram consumes, not models polygram trains"
        " itself."
    ),
    "sm-sae": (
        "[sm-sae](https://github.com/jascal/sm-sae) trains sparse autoencoders"
        " over the Standard Model tensor bundle — a synthetic substrate with a"
        " known feature factorisation (charge ⊗ color ⊗ flavor ⊗ generation)."
        " The three SAE variants below are the ones evaluated against this"
        " ground-truth basis. See `smsae/sae/models.py` for the trainer-side"
        " implementations."
    ),
    "econ-sae": (
        "[econ-sae](https://github.com/jascal/econ-sae) trains SAEs over the"
        " hidden activations of small world-model MLPs that simulate a"
        " stock-flow-consistent macroeconomy. The non-LLM **world models** are"
        " documented first — their H1 hidden layer is the SAE training"
        " substrate. The three SAE variants follow."
    ),
}


# Sibling repos live alongside n-orca in the shared workspace dir; override any
# of them with the corresponding flag.
_CODE_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polygram", type=Path, default=_CODE_ROOT / "polygram")
    parser.add_argument("--sm-sae",   type=Path, default=_CODE_ROOT / "sm-sae")
    parser.add_argument("--econ-sae", type=Path, default=_CODE_ROOT / "econ-sae")
    args = parser.parse_args()

    print(f"Generating polygram docs at {args.polygram}/docs/architectures/")
    write_project(args.polygram, PROJECT_INTROS["polygram"], polygram_entries())

    print(f"Generating sm-sae docs at {args.sm_sae}/docs/architectures/")
    write_project(args.sm_sae, PROJECT_INTROS["sm-sae"], sm_sae_entries())

    print(f"Generating econ-sae docs at {args.econ_sae}/docs/architectures/")
    write_project(args.econ_sae, PROJECT_INTROS["econ-sae"], econ_sae_entries())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
