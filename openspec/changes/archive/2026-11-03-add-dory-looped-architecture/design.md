## Context

Dory1.0-4B is a looped (latent-recurrent) nGPT decoder:

    s_0 = I(emb(x));  s_k = R(s_{k-1} + h_in), k = 1..K;  logits = H(O(s_K))

where `h_in = s_0` is the Input block's output and `R`'s weights are shared
across the K = `n_recurrent_loops` applications. The config's `layer_types`
lists *sublayers* (attention or `mlp`), so the 16/40/16 split is the 8/20/8
blocks NVIDIA describes. No modeling code was available (not in transformers
main, not in the checkpoint repo); the config, README and NVIDIA's diagram are
the only sources.

## Decisions

1. **Unroll the loop.** n-orca graphs are DAGs and the verifier rejects
   cycles, so each loop is emitted as its own copy of the Recurrent block,
   prefixed `loop<k>_`. The loop count is taken from the config (2 in the
   published checkpoint, NVIDIA's recommended default); changing it changes
   only how many copies are emitted.
2. **Groups as the visual structure.** `input_block`, `recurrent_loop_<k>`
   and `output_block` are declared in `## groups`, so Mermaid draws them as
   subgraphs. This is the main thing that makes a ~340-node diagram readable.
3. **Re-injection as an explicit `Add`.** Each loop starts with
   `loop<k>_inject`, fed by the previous state and by `h_in`. For loop 1 both
   inputs come from the same node (`s_0 = h_in`); the duplicate edge is kept on
   purpose because it matches the diagram. Any scaling or renormalization of
   that sum is not public, so none is drawn.
4. **nGPT sublayer as `f -> Add -> LayerNorm`.** The update
   `h <- Norm(h + alpha*(Norm(f(h)) - h))` is drawn with three nodes, not
   four: the inner `Norm(f(h))` is folded into the `Add` (the alpha-lerp)
   description. This keeps the node count down without hiding the
   hypersphere renormalization.
5. **New `GroupedQueryAttention` op, not a reuse of LFM3's.**
   `HeadGatedGroupedQueryAttention` derives `head_dim = d/n_heads` and adds a
   gate; Dory has `head_dim = 256` (4096-wide Q against hidden 2560) and its
   config does not establish an output gate (`attention_output_gate_logit_scale_init`
   is null). A generic op with an explicit `head_dim` gives correct attention
   parameters and is reusable by other GQA models.

## Known approximations

- Parameters are counted once per unrolled loop (n-orca has no weight-tying),
  and SwiGLU is rendered as a two-matrix `FeedForward`; the reported ~4.20B is
  therefore not the checkpoint's unique count (~3.6B estimated).
- nGPT learned scalings (alpha, `s_qk`, logit scale) are described but not
  modelled; PyTorch output is topology-only for GQA, RoPE and window masks.

## Follow-ups (not in scope)

- Weight-tying / shared-module support would fix the parameter count; this
  adapter is the obvious first consumer.
- A dedicated normalized-residual op could replace the `Add` + `LayerNorm`
  pair for nGPT-style models.
