"""JEPA / V-JEPA / I-JEPA / LeWorldModel family adapter.

JEPA models are *joint-embedding predictive architectures*: a ViT-style
**encoder** maps a (masked) context view into a latent representation, and a
lighter **predictor** forecasts the latents of unseen / future targets in that
same embedding space. Training never reconstructs pixels — the loss lives in
latent space, against an EMA "target" encoder.

This adapter normalizes two very different config schemas into one
encoder → predictor DAG:

* **V-JEPA 2 / I-JEPA** (`facebook/vjepa2-*`, `facebook/ijepa_*`) — a flat
  Hugging Face `transformers` config. The encoder fields sit at the top level
  (`hidden_size`, `num_hidden_layers`, …) and the predictor fields are prefixed
  `pred_` (`pred_hidden_size`, `pred_num_hidden_layers`, …). V-JEPA 2 is a
  *video* model (a Conv3d tubelet embedding over `frames_per_clip`); I-JEPA is
  an *image* model.

* **LeWorldModel / LeWM** (`quentinll/lewm-*`) — a nested Hydra-style config
  keyed by `_target_`, with `encoder` / `predictor` / `action_encoder` /
  `projector` sub-dicts. These are action-conditioned world models built on a
  ViT-Tiny encoder, and they carry a LeJEPA SIGReg latent regularizer.

What is faithfully modeled in the AST: the patch/tubelet embedding, the encoder
and predictor transformer stacks (distinct widths/depths), the latent
projection between them, optional additive **action conditioning**, and an
optional **projector** head. What is captured as metadata / verification rules
rather than topology (mirroring how the LLaMA adapter treats RoPE/RMSNorm): the
learnable predictor **mask tokens**, the EMA **stop-gradient** target branch,
the **SIGReg** regularizer, and rotary/sincos positions — these are either
parameters without a graph input or pure loss terms.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from n_orca.ast import (
    Architecture,
    Hyperparameter,
    Tensor,
    Layer,
    FlowEdge,
    OpCall,
    Invariant,
)
from n_orca.hf.adapters.base import HfAdapter, register
from n_orca.hf.adapters._blocks import pre_norm_decoder_block


# ViT presets used by LeWM's `vit_hf` encoder (size -> dim, depth, heads).
_VIT_SIZES: dict[str, tuple[int, int, int]] = {
    "tiny": (192, 12, 3),
    "small": (384, 12, 6),
    "base": (768, 12, 12),
    "large": (1024, 24, 16),
    "huge": (1280, 32, 16),
    "giant": (1408, 40, 16),
    "gigantic": (1664, 48, 16),
}


@dataclass
class _JepaSpec:
    """Normalized, schema-agnostic view of a JEPA checkpoint."""
    family: str                 # "vjepa2" | "ijepa" | "lewm"
    is_video: bool
    # encoder
    d_model: int
    n_layer: int
    n_heads: int
    d_ff: int
    patch: int
    image: int
    in_chans: int
    frames: int
    tubelet: int
    # predictor
    pred_dim: int
    pred_layers: int
    pred_heads: int
    pred_d_ff: int
    mask_tokens: int
    dropout: float
    # optional pieces
    action_dim: int | None = None
    proj_hidden: int | None = None
    has_sigreg: bool = False
    pos_kind: str = "rope"      # "rope" | "sincos" | "learned" | "none"
    name_hint: str = ""

    @property
    def num_patches(self) -> int:
        """Token count after patch/tubelet embedding (folded to an int)."""
        spatial = (self.image // self.patch) ** 2
        temporal = (self.frames // self.tubelet) if self.is_video else 1
        return spatial * temporal


# --------------------------------------------------------------------------- #
#  Config parsing
# --------------------------------------------------------------------------- #


def _is_lewm(config: dict[str, Any]) -> bool:
    target = str(config.get("_target_") or "").lower()
    if any(s in target for s in ("lewm", "leworldmodel")):
        return True
    enc, pred = config.get("encoder"), config.get("predictor")
    return isinstance(enc, dict) and isinstance(pred, dict)


def _parse(config: dict[str, Any]) -> _JepaSpec:
    if _is_lewm(config):
        return _parse_lewm(config)
    return _parse_hf(config)


def _parse_hf(config: dict[str, Any]) -> _JepaSpec:
    mt = (config.get("model_type") or "").lower()
    archs = " ".join(config.get("architectures") or []).lower()
    is_video = (
        "vjepa" in mt or "v-jepa" in mt or "v_jepa" in mt or "vjepa" in archs
        or config.get("frames_per_clip") is not None
        or config.get("tubelet_size") is not None
    )
    family = "vjepa2" if is_video else ("ijepa" if "ijepa" in (mt + archs) else "jepa")

    d_model = int(config.get("hidden_size", 1024))
    n_layer = int(config.get("num_hidden_layers", 24))
    n_heads = int(config.get("num_attention_heads", 16))
    mlp_ratio = float(config.get("mlp_ratio", 4) or 4)
    d_ff = int(config.get("intermediate_size") or round(d_model * mlp_ratio))
    patch = int(config.get("patch_size", 16))
    image = int(config.get("crop_size") or config.get("image_size") or 224)
    in_chans = int(config.get("in_chans") or config.get("num_channels") or 3)
    frames = int(config.get("frames_per_clip", 16)) if is_video else 1
    tubelet = int(config.get("tubelet_size", 2)) if is_video else 1

    # When a checkpoint ships only the encoder (e.g. the HF I-JEPA config),
    # default the predictor to the encoder's width/heads. d_model is always
    # divisible by n_heads, so the compiled predictor attention stays valid.
    pred_dim = int(config.get("pred_hidden_size") or d_model)
    pred_layers = int(config.get("pred_num_hidden_layers") or max(1, n_layer // 2))
    pred_heads = int(config.get("pred_num_attention_heads") or n_heads)
    pred_mlp_ratio = float(config.get("pred_mlp_ratio") or 4.0)
    pred_d_ff = int(round(pred_dim * pred_mlp_ratio))
    mask_tokens = int(config.get("pred_num_mask_tokens") or 0)

    dropout = float(
        config.get("hidden_dropout_prob")
        or config.get("attention_probs_dropout_prob")
        or 0.0
    )

    return _JepaSpec(
        family=family, is_video=is_video,
        d_model=d_model, n_layer=n_layer, n_heads=n_heads, d_ff=d_ff,
        patch=patch, image=image, in_chans=in_chans, frames=frames, tubelet=tubelet,
        pred_dim=pred_dim, pred_layers=pred_layers, pred_heads=pred_heads,
        pred_d_ff=pred_d_ff, mask_tokens=mask_tokens, dropout=dropout,
        action_dim=None, proj_hidden=None, has_sigreg=False,
        pos_kind="rope" if is_video else "sincos",
        name_hint=_name_hint(config),
    )


def _parse_lewm(config: dict[str, Any]) -> _JepaSpec:
    enc = config.get("encoder") or {}
    pred = config.get("predictor") or {}
    act = config.get("action_encoder") or {}
    proj = config.get("projector") or {}

    size = str(enc.get("size", "tiny")).lower()
    d_model, n_layer, n_heads = _VIT_SIZES.get(size, _VIT_SIZES["tiny"])
    d_ff = d_model * 4
    patch = int(enc.get("patch_size", 14))
    image = int(enc.get("image_size", 224))

    pred_dim = int(pred.get("hidden_dim") or pred.get("input_dim") or d_model)
    pred_layers = int(pred.get("depth", 6))
    pred_heads = int(pred.get("heads", 8))
    pred_d_ff = int(pred.get("mlp_dim") or pred_dim * 4)
    dropout = float(pred.get("dropout", 0.0) or 0.0)

    action_dim = int(act.get("input_dim")) if isinstance(act, dict) and act.get("input_dim") else None
    proj_hidden = int(proj.get("hidden_dim")) if isinstance(proj, dict) and proj.get("hidden_dim") else None

    return _JepaSpec(
        family="lewm", is_video=False,
        d_model=d_model, n_layer=n_layer, n_heads=n_heads, d_ff=d_ff,
        patch=patch, image=image, in_chans=3,
        frames=int(pred.get("num_frames", 1)), tubelet=1,
        pred_dim=pred_dim, pred_layers=pred_layers, pred_heads=pred_heads,
        pred_d_ff=pred_d_ff, mask_tokens=0, dropout=dropout,
        action_dim=action_dim, proj_hidden=proj_hidden, has_sigreg=True,
        pos_kind="sincos", name_hint=_name_hint(config),
    )


def _name_hint(config: dict[str, Any]) -> str:
    return str(config.get("_name_or_path") or config.get("name_or_path") or "")


# --------------------------------------------------------------------------- #
#  Adapter
# --------------------------------------------------------------------------- #


class JepaAdapter(HfAdapter):
    """V-JEPA 2 / I-JEPA / LeWorldModel — encoder + joint-embedding predictor."""

    model_types = (
        "jepa", "ijepa", "i-jepa", "i_jepa",
        "v-jepa", "vjepa", "vjepa2", "v_jepa2", "vje pa2",
        "leworldmodel", "lewm",
    )

    def matches(self, config: dict[str, Any]) -> bool:
        mt = (config.get("model_type") or "").lower()
        archs = [a.lower() for a in (config.get("architectures") or [])]
        keys = {k.lower() for k in self.model_types if k.strip()}
        if mt and (mt in keys or any(k in mt for k in keys)):
            return True
        if any(any(k in a for k in keys) for a in archs):
            return True
        # LeWorldModel ships a Hydra config with no model_type/architectures.
        target = str(config.get("_target_") or "").lower()
        if any(s in target for s in ("lewm", "leworldmodel", "jepa")):
            return True
        enc, pred = config.get("encoder"), config.get("predictor")
        if isinstance(enc, dict) and isinstance(pred, dict):
            blob = json.dumps(config).lower()
            if any(s in blob for s in ("jepa", "lewm", "worldmodel", "world_model")):
                return True
        return False

    def build(self, config: dict[str, Any], *, name: str | None = None) -> Architecture:
        spec = _parse(config)
        arch = Architecture(
            name=name or _derive_name(spec),
            description=_describe(spec),
            hyperparameters=_hyperparameters(spec),
            tensors=_tensors(spec),
        )

        # ---- Encoder: patch/tubelet embed -> ViT pre-norm blocks -> LN ----
        arch.layers.append(Layer(
            name="pixel_values", is_input=True,
            description=(
                "Raw video clip (B, C, T, H, W)" if spec.is_video
                else "Raw image (B, C, H, W)"
            ),
        ))
        if spec.is_video:
            arch.layers.append(Layer(
                name="patch_embed",
                description="Conv3d tubelet embedding -> token sequence",
                op=OpCall("TubeletEmbed",
                          [str(spec.in_chans), "d_model",
                           str(spec.tubelet), str(spec.patch)]),
            ))
        else:
            arch.layers.append(Layer(
                name="patch_embed",
                description="Conv2d patch embedding -> token sequence",
                op=OpCall("PatchEmbed",
                          [str(spec.in_chans), "d_model", str(spec.patch)]),
            ))
        arch.flow.append(FlowEdge("pixel_values", "patch_embed", "pixels"))

        prev = "patch_embed"
        for i in range(spec.n_layer):
            layers, edges, prev = pre_norm_decoder_block(
                index=i, prev_layer=prev,
                d_model="d_model", n_heads="n_heads",
                d_ff="d_ff", dropout="dropout", prefix="enc",
            )
            arch.layers.extend(layers)
            arch.flow.extend(edges)

        arch.layers.append(Layer(
            name="enc_norm",
            description="Final encoder LayerNorm — context representation",
            op=OpCall("LayerNorm", ["d_model"]),
        ))
        arch.flow.append(FlowEdge(prev, "enc_norm", "enc_y"))

        # Encoder latents are an output in their own right: the context
        # representation a downstream probe / SAE would read.
        arch.layers.append(Layer(
            name="encoder_latents", is_output=True,
            description="Context-view latents (the JEPA representation)",
        ))
        arch.flow.append(FlowEdge("enc_norm", "encoder_latents", "context"))

        # ---- Predictor: project to predictor width, blocks, project back ----
        arch.layers.append(Layer(
            name="pred_in",
            description="Linear bridge from encoder dim into the predictor width",
            op=OpCall("Linear", ["d_model", "pred_dim"]),
        ))
        arch.flow.append(FlowEdge("enc_norm", "pred_in", "context"))

        pred_prev = "pred_in"
        if spec.action_dim is not None:
            # Action-conditioned world model (LeWM / V-JEPA 2-AC): embed the
            # action and add it to every context token. The host tiles the
            # per-step action embedding across the token axis (see rules).
            arch.layers.append(Layer(
                name="actions", is_input=True,
                description="Per-step control / action vector (broadcast over tokens)",
            ))
            arch.layers.append(Layer(
                name="action_embed",
                description="Embed actions into the predictor width",
                op=OpCall("Linear", ["action_dim", "pred_dim"]),
            ))
            arch.layers.append(Layer(
                name="add_action",
                description="Additive action conditioning",
                op=OpCall("Add", []),
            ))
            arch.flow.append(FlowEdge("actions", "action_embed", "a"))
            arch.flow.append(FlowEdge("pred_in", "add_action", "ctx_tok"))
            arch.flow.append(FlowEdge("action_embed", "add_action", "a_emb"))
            pred_prev = "add_action"

        for i in range(spec.pred_layers):
            layers, edges, pred_prev = pre_norm_decoder_block(
                index=i, prev_layer=pred_prev,
                d_model="pred_dim", n_heads="pred_heads",
                d_ff="pred_d_ff", dropout="dropout", prefix="pred",
            )
            arch.layers.extend(layers)
            arch.flow.extend(edges)

        arch.layers.append(Layer(
            name="pred_norm",
            description="Final predictor LayerNorm",
            op=OpCall("LayerNorm", ["pred_dim"]),
        ))
        arch.layers.append(Layer(
            name="pred_out",
            description="Project predictor latents back to the encoder dim",
            op=OpCall("Linear", ["pred_dim", "d_model"]),
        ))
        arch.flow.append(FlowEdge(pred_prev, "pred_norm", "pred_y"))
        arch.flow.append(FlowEdge("pred_norm", "pred_out", "pred_yn"))

        last = "pred_out"
        if spec.proj_hidden is not None:
            # LeWM projector / pred_proj head (BatchNorm1d in the original;
            # rendered here as a plain MLP — the norm is applied host-side).
            arch.layers.append(Layer(
                name="proj_fc1", op=OpCall("Linear", ["d_model", "proj_hidden"])))
            arch.layers.append(Layer(
                name="proj_act", op=OpCall("ReLU", [])))
            arch.layers.append(Layer(
                name="proj_fc2", op=OpCall("Linear", ["proj_hidden", "d_model"])))
            arch.flow.append(FlowEdge("pred_out", "proj_fc1", "p1"))
            arch.flow.append(FlowEdge("proj_fc1", "proj_act", "p2"))
            arch.flow.append(FlowEdge("proj_act", "proj_fc2", "p3"))
            last = "proj_fc2"

        arch.layers.append(Layer(
            name="predicted_latents", is_output=True,
            description="Predicted target-view latents (joint-embedding forecast)",
        ))
        arch.flow.append(FlowEdge(last, "predicted_latents", "prediction"))

        # Both outputs live in the encoder's embedding space (B, N, d_model),
        # so a single output_shape invariant covers them.
        n = str(spec.num_patches)
        arch.invariants.append(Invariant("output_shape", "=", ("B", n, "d_model")))
        arch.verification_rules.extend(_verification_rules(spec))
        return arch


# --------------------------------------------------------------------------- #
#  Builders for the declarative sections
# --------------------------------------------------------------------------- #


def _hyperparameters(spec: _JepaSpec) -> list[Hyperparameter]:
    hps = [
        Hyperparameter("d_model", "int", spec.d_model),
        Hyperparameter("n_layer", "int", spec.n_layer),
        Hyperparameter("n_heads", "int", spec.n_heads),
        Hyperparameter("d_ff", "int", spec.d_ff),
        Hyperparameter("pred_dim", "int", spec.pred_dim),
        Hyperparameter("pred_layers", "int", spec.pred_layers),
        Hyperparameter("pred_heads", "int", spec.pred_heads),
        Hyperparameter("pred_d_ff", "int", spec.pred_d_ff),
        Hyperparameter("patch_size", "int", spec.patch),
        Hyperparameter("image_size", "int", spec.image),
        Hyperparameter("in_chans", "int", spec.in_chans),
        Hyperparameter("num_patches", "int", spec.num_patches),
        Hyperparameter("pred_mask_tokens", "int", spec.mask_tokens),
        Hyperparameter("dropout", "float", spec.dropout),
    ]
    if spec.is_video:
        hps.insert(11, Hyperparameter("frames_per_clip", "int", spec.frames))
        hps.insert(12, Hyperparameter("tubelet_size", "int", spec.tubelet))
    if spec.action_dim is not None:
        hps.append(Hyperparameter("action_dim", "int", spec.action_dim))
    if spec.proj_hidden is not None:
        hps.append(Hyperparameter("proj_hidden", "int", spec.proj_hidden))
    return hps


def _tensors(spec: _JepaSpec) -> list[Tensor]:
    n = str(spec.num_patches)
    if spec.is_video:
        pixels = Tensor("pixel_values",
                        ("B", str(spec.in_chans), str(spec.frames),
                         str(spec.image), str(spec.image)), "float32")
    else:
        pixels = Tensor("pixel_values",
                        ("B", str(spec.in_chans), str(spec.image), str(spec.image)),
                        "float32")
    tensors = [
        pixels,
        Tensor("encoder_latents", ("B", n, "d_model"), "float32"),
        Tensor("predicted_latents", ("B", n, "d_model"), "float32"),
    ]
    if spec.action_dim is not None:
        tensors.insert(1, Tensor("actions", ("B", n, "action_dim"), "float32"))
    return tensors


def _verification_rules(spec: _JepaSpec) -> list[str]:
    rules = [
        "latent-dim-consistency: encoder_latents and predicted_latents share "
        "the embedding dim d_model — the predictor forecasts in the encoder's "
        "representation space, never in pixel space.",
        "joint-embedding: the training loss compares predicted_latents against "
        "an EMA target-encoder's latents (L1/L2 in feature space).",
        "stop-gradient: the target encoder is an EMA copy of the context "
        "encoder; gradients do not flow into the target branch (not modeled as "
        "a forward node here).",
    ]
    if spec.mask_tokens:
        rules.append(
            f"mask-tokens: the predictor queries {spec.mask_tokens} learnable "
            "mask tokens for masked/target positions (pred_num_mask_tokens); "
            "modeled here as in-place latent forecasting over the N context "
            "tokens."
        )
    if spec.action_dim is not None:
        rules.append(
            "action-conditioning: the predictor is conditioned on an embedded "
            "action added to the context tokens; the per-step action embedding "
            "is broadcast across the N token axis upstream."
        )
        rules.append(
            f"prediction-horizon: the predictor rolls the latent state forward "
            f"over {spec.frames} frame(s) of control (predictor.num_frames)."
        )
    if spec.has_sigreg:
        rules.append(
            "sigreg-regularizer: a LeJEPA SIGReg (sketched isotropic-Gaussian) "
            "penalty regularizes the latent distribution to prevent collapse; "
            "it is a loss term, not part of the forward graph rendered here."
        )
    pos = {
        "rope": "positions: encoder/predictor attention uses 3D rotary position "
                "embeddings (RoPE), applied inside attention — left implicit here.",
        "sincos": "positions: fixed 2D sin-cos position embeddings are added to "
                  "the patch tokens — left implicit here (parameter-free).",
        "learned": "positions: learned absolute position embeddings (omitted "
                   "from the topology in this generated form).",
        "none": "positions: no explicit positional encoding.",
    }
    rules.append(pos.get(spec.pos_kind, pos["none"]))
    return rules


def _describe(spec: _JepaSpec) -> str:
    family = {
        "vjepa2": "V-JEPA 2 video world model",
        "ijepa": "I-JEPA image model",
        "lewm": "LeWorldModel (LeWM) action-conditioned world model",
        "jepa": "JEPA model",
    }.get(spec.family, "JEPA model")
    modality = "video clips" if spec.is_video else "images"
    bits = [
        f"{family} — joint-embedding predictive architecture over {modality}.",
        f"A {spec.n_layer}-block ViT encoder ({spec.d_model}d, {spec.n_heads} heads)",
        f"feeds a {spec.pred_layers}-block predictor ({spec.pred_dim}d,"
        f" {spec.pred_heads} heads) that forecasts target latents in the"
        " encoder's embedding space.",
    ]
    if spec.action_dim is not None:
        bits.append(
            f"The predictor is action-conditioned (action_dim={spec.action_dim})"
            f" with a {spec.proj_hidden}-wide projector head."
            if spec.proj_hidden is not None else
            f"The predictor is action-conditioned (action_dim={spec.action_dim})."
        )
    bits.append(
        "Mask tokens, the EMA target encoder, and the latent regularizer are"
        " captured as verification rules (see below)."
        " Generated by `n-orca hf convert` from a Hugging Face config."
    )
    return " ".join(bits)


def _derive_name(spec: _JepaSpec) -> str:
    if spec.name_hint:
        bare = spec.name_hint.split("/")[-1].replace("-", "_").replace(".", "_")
        return _Pascal(bare)
    tag = {"vjepa2": "VJepa2", "ijepa": "IJepa", "lewm": "LeWorldModel"}.get(
        spec.family, "Jepa")
    return f"{tag}_{spec.n_layer}L_{spec.d_model}d"


def _Pascal(s: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in s.replace("-", "_").split("_") if p) or "Model"


register(JepaAdapter())
