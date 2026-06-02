## Context

n-orca currently provides four world model builders (baseline, deep, attn_world_model, temporal_world_model) plus six SAE variants (including advanced attn_topk_sae, supervised_topk_sae with multi-output, gated_sae). These are used as the canonical, verifiable source of truth for SAE substrates in econ-sae (and planned for sm-sae/sae-forge/polygram consumption). The temporal addition (per OpenSpec add-temporal-world-model, PR #3) introduced explicit `hidden_in` / `hidden_out` state tensors + external carry for cross-period regime features.

Cosmos 3 (NVIDIA's latest unified world foundation model for Physical AI, ~2026) introduces the **Mixture of Transformers (MoT)** architecture as its defining pattern for combining autoregressive reasoning (VLM-like) with diffusion/flow-matching generation (world simulation, video, audio, action policies) in a single model. Key elements (from technical report + subagent research):

- Dual-tower per block: AR (reasoner, causal next-token) and DM (generator, denoising) towers with independent LayerNorms, attn projections, and MLPs.
- Joint self-attention as the fusion point: AR attends only over AR (causal); DM attends bidirectionally over concat(AR, DM) for conditioning.
- Unified token sequence: AR prefix (text + ViT understanding) + DM (noisy continuous latents from VAE/tokenizers for video/audio/action).
- Timestep conditioning for diffusion steps (flow-matching velocity prediction).
- Advanced 3D multimodal RoPE (mRoPE with t/h/w, modality offsets, FPS modulation for physical time alignment across rates).
- Encoders: ViT for AR vision, VAE (video/audio) for DM, compact geometric action reps.
- External iteration: denoising loop over timesteps (like temporal's external hidden carry); per-step graph is a DAG.

A subagent was explicitly spawned (2026-06-02, task 019e8a6d-f3b7-77f1-baf1-8786d6306436) to investigate Cosmos 3/MoT, its potential as SAE ground-truth substrate (rich physics/multimodal/diffusion factors with derivable GT from synthetic data + reasoner labels), feasibility of a new world-sae/cosmos-sae sibling, and concrete n-orca extensions needed. Report confirms high potential for both, with detailed plan. Primary recommendation: start with OpenSpec + n-orca support (builders/examples/MCP/ops) so n-orca remains the spec layer; then scaffold the -sae repo.

This design addresses the need for MoT patterns in n-orca world models so future SAE work (in new repo or existing) can use verified .n.orca.md specs for "Cosmos-style" hosts that produce interesting activations (reasoner concepts, DM latents at specific noise levels, cross-tower conjunctives, spatio-temporal physics features).

## Goals / Non-Goals

**Goals:**
- Define n-orca DAG representations for MoT dual-tower + joint-attention fusion + diffusion timestep conditioning, using explicit tensors for AR/DM streams and timestep (enables per-denoise-step graphs; diffusion schedule / full rollout external, like temporal state carry).
- Enable builder(s) that produce verifiable Architectures compatible with existing world models (reuse/extend Linear/LN/MHA/Positional/Patch/Tubelet/Add etc.).
- Support key MoT invariants in examples/docs (AR causal integrity; DM conditioned on AR context without backflow to AR).
- Provide clear path for MCP, skill, 1-3 examples, tests, docs, and OpenSpec.
- Align with econ-sae "world model as substrate" + extend to Physical AI / multimodal / diffusion factors (new GT tiers: physics dynamics, cross-modal, diffusion-stage).
- Keep n-orca the shared source of truth (no sibling edits in this change).

**Non-Goals:**
- Full end-to-end Cosmos implementation or 16B/64B scale (toy d_model=64-256, short sequences; real models via HF/diffusers in consumers).
- Implementing the iterative denoising loop or custom VAE/mRoPE inside n-orca (per-step DAG + external caller loop; mRoPE as note or Positional extension).
- New full multimodal encoder ops beyond stubs (build on existing PatchEmbed/TubeletEmbed + Linear for action).
- Changes outside n-orca (world-sae scaffold, sae-forge adapter, econ updates are later).
- Classic sparse MoE routing (Cosmos MoT is dual-tower shared-attn mixture, not per-token experts).
- Breaking existing builders or verifier for 2D assumptions (3D video already supported via Tubelet etc.).

## Decisions

**Decision: Model MoT as per-denoise-step DAGs with explicit AR/DM tensors + timestep tensor (external iteration for the diffusion schedule).**

Rationale: Matches n-orca's strength (verified forward DAGs of named tensors/layers/flows) and the temporal precedent (explicit hidden_in/out + external carry/unroll in econ-sae). A single "denoise step" is a pure DAG: AR path (causal self), DM path (full joint over AR+DM), timestep embed injected to DM. The full diffusion (T steps) or video rollout is unrolled by caller (like temporal periods). This keeps specs small/verifiable while capturing the MoT pattern. Report explicitly notes: "represent single denoise step + timestep as n-orca subgraph".

Alternatives considered:
- Full unrolled diffusion steps in one Architecture: Rejected — explodes size for 50-1000 steps; n-orca not an execution engine.
- Custom "DiffusionStep" composite op: Rejected for v1 — prefer explicit layers/flows for inspectability (like temporal uses explicit Linear `hidden_update` for state carry).
- Treating as two separate models (reasoner + generator): Rejected — misses the joint-attn "mixture" fusion which is the key MoT innovation for conditioning.

**Decision: Use/extend existing ops + minimal targeted new ops for dual-tower and joint attention.**

Rationale: AR/DM paths can be modeled with existing LayerNorm + Linear/FeedForward + MHA (with care for masks/concat in flows). For joint: either (a) custom flow description in the .n.orca.md example (two MHA calls + concat logic via tensors) or (b) new `DualStreamJointAttention` op (preferred for clean builder emission and PyTorch special case like current MHA). New `TimestepEmbed` (sinusoidal or Linear+act) for diffusion conditioning (standard in DiT). Reuse TubeletEmbed/Patch for video latents, Positional for mRoPE notes. Keeps diff small; compiler already handles multi-output, 3D, custom MHA.

**Decision: Expose hyperparameters like `d_model`, `n_heads`, `timestep_dim`, `num_ar_tokens` (stub), `dm_stream_len` etc. in builders and MCP. Support "mot_denoise_step", "mot_reasoner", "mot_omni_stub" variants.**

Rationale: Mirrors temporal (hidden_dim, num_layers, gru_hidden). Allows generating different slices (reasoner-only for VLM part; single denoise for generator core; fuller with encoder stubs). Report sketches exact usage for `n-orca build-world-model mot ...`.

**Decision: Architecture will have dual stream tensors (e.g. `ar_x (B, S_ar, D)`, `dm_x (B, S_dm, D)`, `t (B,)`) and dual-ish outputs or main DM output + optional AR. Document "AR causal invariant" in invariants / docstring.**

Rationale: Existing supervised SAE already does multi-output tuples. Temporal shows extra state tensors. Verifier/compiler support demonstrated. In PyTorch emission: forward can take ar_x, dm_x_noisy, t and return denoised (DM always runs both towers internally for conditioning). Mermaid will show the cross edges naturally.

**Decision: Start with one primary builder + example (mot_denoise_step) + reasoner variant; expand in tasks if green.**

Rationale: Smallest safe slice per principles (like temporal started basic). Full omni with action/audio encoders later or as separate example.

## Risks / Trade-offs

- [Risk] Joint attention concat logic + causal vs bidirectional masks must be correctly emitted in PyTorch and described in flows. Shape threading for mixed AR/DM seqs. → Mitigation: Model explicitly in builder (separate Q/K/V paths or custom op with documented mask); add test asserting no AR pollution; reuse/extend _torch_call_mha pattern.
- [Trade-off] mRoPE / FPS modulation / 3D posembs are complex; full symbolic support hard. → Acceptable: Use PositionalEmbedding + docstring notes + modality/timestep offsets in example; real impl in consumer HF code. n-orca captures the high-level pattern.
- [Risk] Verifier may need minor extension for "causal integrity" or dual-tower invariants (new concept). → Mitigation: Start with existing structural + shape rules; add custom note in Architecture.invariants or docstring. Tests will catch.
- [Risk] Diffusion timestep as scalar or (B,) ; conditioning add vs cross-attn. Scale of video tokens (T*H*W). → Low: treat t as (B,) tensor like batch; examples use small S; note external VAE in docs.
- [Risk] New -sae repo (world-sae) may have high activation volume / weak GT (derived from synthetic + reasoner, not exact equations like econ/sm). → Mitigation: Scope n-orca to small verifiable specs first; GT vocab in subagent report is starting point (physics, objects, motion, diffusion factors, conjunctive). Subagent assessed "strong" overall.
- Low risk overall: purely additive (new builders/ops/examples like temporal + advanced SAEs). No existing code paths changed. Subagent report provides full grounding + risk section.

## Migration Plan

- Additive only: new functions in world_models.py, new cases/variants in mcp_server, new files under examples/, new test funcs, doc updates.
- Existing world models / SAEs / compiler paths untouched.
- New examples follow naming `cosmos-mot-*.n.orca.md` (like `econ-temporal-*.n.orca.md`); added to any "all examples" sweeps.
- After impl + green + PR: update IMPROVEMENT_GOAL Recent Activity; future scheduler cycles or cross-project can consume.
- For world-sae scaffold (separate): once n-orca examples exist, the .n.orca.md becomes the "host spec" imported in world-sae's world_model.py (like econ-sae does).
- No rollback; if MoT modeling needs refinement (e.g. after real hooks), new OpenSpec or follow-up tasks.

## Open Questions

- Exact joint attention op surface: new DualStreamJointAttention(d_model, n_heads, ar_len, dm_len?) with internal concat/mask vs. builder emitting two MHA + manual K/V concat tensors? (Prefer op for cleanliness and special-case emission.)
- How much of mRoPE / 3D pos to model vs. document? (Start with Positional + rich docstring + example comment; extend PositionalEmbedding later if needed.)
- Should there be a "full diffusion unroll helper" or stay per-step only? (Per design: per-step; external in consumer like temporal.)
- Timestep embed exact (sinusoid like original DiT, or learned Linear)? Make configurable or fixed for now.
- GT labeling strategy for new -sae (reasoner CoT as weak labels? synthetic dataset metadata? probes on motion/collision?). Defer to world-sae creation (subagent report has initial vocab).
- When to scaffold world-sae vs. first land n-orca support + one probe? (Subagent: n-orca OpenSpec first, then minimal repo.)

This design enables safe, incremental, test-first addition of MoT support modeled directly on the subagent's concrete recommendations and the successful temporal precedent. It positions n-orca to support the next wave of world-model interpretability work.
