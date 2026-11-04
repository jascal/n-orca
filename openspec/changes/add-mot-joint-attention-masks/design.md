## Context

In a Mixture-of-Transformers block, AR and DM tokens are concatenated into one
sequence and run through ONE global self-attention. Each token is projected
with its own tower's Q/K/V/O weights, and a mask makes AR rows causal over AR
only while DM rows see everything. Two facts follow from that mask:

- The AR rows of the global attention are exactly **causal self-attention over
  AR with AR-tower weights**. No DM information can reach them.
- The DM rows are **DM queries attending, unmasked, over [AR keys/values ;
  DM keys/values]**.

So the global op decomposes into the two-tower shape `mot_denoise_step`
already has: `ar_mha` (AR tower) and `dm_mha` (DM tower). What is missing is
the AR → DM edge, the causal mask, and emission that concatenates the keys.

```
today                              this change
ar_x → ar_ln → ar_mha → ar_add     ar_x → ar_ln → ar_mha(causal) → ar_add → ar_out
dm_x → dm_ln → dm_mha → dm_add            └──────────┐
                                   dm_x → dm_ln → dm_mha(q=dm, kv=[ar;dm]) → dm_add → …
```

## Goals / Non-Goals

**Goals:**
- Emitted PyTorch implements the MoT masks: AR causal, DM bidirectional over
  concat(AR, DM), and no DM → AR dependence.
- The invariants are **checked by the verifier**, not just described:
  topology properties on the builder's output.
- No parameter-count change, and no behaviour change for any existing
  three-argument `MultiHeadAttention` document.

**Non-Goals:**
- Injecting the timestep embedding into the DM path (needs a broadcast add or
  adaLN op); recorded as a follow-up.
- A single global joint-attention op with a split back into towers (see
  Decision 1).
- Multi-block stacks, mRoPE, video/action tokenization.

## Decisions

1. **Keep two towers; make the DM op two-input.** `DualStreamJointAttention`
   takes `(dm, ar)` and computes DM rows only.
   *Alternative:* one global op over `concat(ar, dm)` followed by splitting the
   result back into AR and DM rows. It is closer to a literal MoT
   implementation, but n-orca has no slice op and its shapes would need
   symbolic `S_ar + S_dm` arithmetic. The decomposition above computes the
   same AR rows exactly, and the diagram keeps the two towers visible.

2. **Known approximation: which weights project AR keys for DM queries.** In
   real MoT, AR tokens are projected with AR-tower K/V weights even when DM
   queries attend to them. Here `dm_mha` is one `nn.MultiheadAttention` whose
   K/V weights project both AR and DM tokens. Parameter totals are unchanged
   and still correct (one Q/K/V/O set per tower); the function differs only in
   that one projection. Stated in the op and builder descriptions.
   *Alternative:* give `dm_mha` a second K/V projection for AR tokens. Rejected
   because it would add `2d² + 2d` parameters that do not exist in the real
   model.

3. **Argument order from `## flow` rows.** As with `GRUCell`, the op takes
   inputs in flow-row order: `dm` first, then `ar`. The builder emits
   `dm_ln → dm_mha` before `ar_ln → dm_mha`. The shape rule cannot tell the
   two apart by name (both last dims are `d_model`), so the error text and
   docs state the order. For the builder, a test pins it.

4. **AR keys come from `ar_ln`.** Within one joint-attention layer, the AR
   keys/values are computed from the layer input, which is the LayerNorm output,
   not from the AR attention's output. So the edge is `ar_ln → dm_mha`.

5. **Causal as an opt-in token: `MultiHeadAttention(d, h, dropout, causal)`.**
   Only the literal 4th argument `causal` enables the mask; anything else, or
   no 4th argument, keeps today's behaviour. Unrecognised op-arg tokens already
   pass through hyperparameter resolution unchanged.
   *Alternative:* a separate `CausalSelfAttention` op. Rejected: it duplicates
   MHA's shape rule, parameters and emission for one flag.

6. **Emission.**
   - Causal MHA:
     `self.m(x, x, x, attn_mask=torch.ones(S, S, dtype=torch.bool, device=x.device).triu(1))[0]`,
     with `S = x.shape[1]`. In PyTorch's boolean mask, `True` blocks a position.
   - Joint: `kv = torch.cat([ar, dm], dim=1)`, then `self.j(dm, kv, kv)[0]`.
   - `DualStreamJointAttention` leaves the compiler's single-input
     attention special case and uses the `{NAME}` call template.

7. **Invariants as properties on the builder's output.** `mot_denoise_step`
   declares `ar_tower` / `dm_tower` groups (also drawn as Mermaid subgraphs)
   and:
   - `unreachable: ar_out from dm_x_noisy`: no DM → AR backflow.
   - `unreachable: ar_out from t`: the timestep does not touch AR.
   - `reachable: dm_x_denoised from ar_x`: DM is conditioned on AR.
   - `passes_through: dm_mha for ar_x -> dm_x_denoised`: the only AR → DM
     route is the joint attention.

   The graph-level properties back up the tensor-level test (Decision 8).

8. **Tests check the masks numerically, not just the graph.** For the compiled
   module: perturbing a later AR token must not change earlier `ar_out`
   positions; perturbing `dm_x_noisy` must not change `ar_out`; perturbing
   `ar_x` must change `dm_x_denoised`.

## Risks / Trade-offs

- [The denoise-step example leaves the frozen baseline] → It is changing
  anyway, and it gains dedicated tests: properties pass, parameter count
  unchanged, numeric mask checks. The reasoner example has no groups, so it
  stays in the baseline with its entry regenerated. Every other entry must
  stay byte-identical.
- [Hand-written one-input `DualStreamJointAttention` documents break] → Only
  builder output uses it in the repo. The arity error explains the second
  edge.
- [Swapped `(dm, ar)` rows verify when `S_dm` and `S_ar` are symbolic] →
  Same-shaped inputs cannot be told apart. Documented, and pinned by a builder
  test.

## Migration Plan

Regenerate both examples from the builders. Rollback is reverting the PR.

## Open Questions

- None blocking. The timestep-injection gap is tracked separately.
