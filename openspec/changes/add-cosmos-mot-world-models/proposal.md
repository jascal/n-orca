# add-cosmos-mot-world-models

## Why

NVIDIA Cosmos (esp. Cosmos 3, launched ~2026-05/06) is a major open world foundation model family for Physical AI: unified omnimodal (language + vision + audio + action) generative world models for simulation, prediction, video gen, policy, and reasoning. Its core architectural innovation is the **Mixture of Transformers (MoT)** pattern: a dual-tower (AR reasoner tower + diffusion/flow-matching generator tower) per transformer block with joint attention (AR attends causally; DM attends bidirectionally over concatenated AR+DM keys/values). This unifies prior separate AR+DiT models into one reason-then-generate system with strong physics awareness, multimodal fusion, and controllable generation (text2world, image2video, action-conditioned dynamics, transfer).

Current n-orca world model builders (baseline/deep/attn/temporal) cover MLP/attn + explicit per-step state (inspired by econ-sae). Cosmos/MoT represents the next evolution: diffusion timestep conditioning, dual-stream layers + specialized joint attention for cross-tower (cross-"modal") conditioning, 3D multimodal positional embeddings (mRoPE + FPS modulation), latent VAE-style video/action tokens, and external iteration over denoising steps (analogous to temporal's external hidden carry).

Adding MoT support keeps n-orca as the single source of truth / verifiable spec layer for neural architectures used across the SAE ecosystem (econ-sae, sm-sae, sae-forge, polygram). It also enables a new sibling `-sae` repo (world-sae or cosmos-sae) with rich, controllable, physics-grounded synthetic GT activations (objects, collisions, trajectories, diffusion-stage factors, multimodal conjunctive features, regime-like scene context) — far richer substrate than toy econ/sm for testing SAE families (topk, gated, attn-prefixed, supervised) on real-world-scale interpretability challenges.

Subagent investigation (spawned per explicit request) confirmed high feasibility for both the new `-sae` repo and direct n-orca NN design extensions (builders, ops, examples). See subagent report for full research, citations, GT feature vocab sketch, and detailed integration plan.

This aligns with IMPROVEMENT_GOAL focus: expand world model/SAE library, agent tooling (MCP/skills for new builders), ecosystem alignment, quality (tests/examples for new patterns like dual-tower + timestep).

## What Changes

- New world model builders in `n_orca/world_models.py`: e.g. `mot_denoise_step(...)`, `mot_reasoner_only(...)`, `mot_multimodal_omni(...)` (high-level DAGs modeling dual-tower MoT + joint attn + timestep embed; toy dims for verification).
- Corresponding new ops or extensions in `n_orca/ops/spec.py` (TimestepEmbed / DiffusionTimestep; DualStreamJointAttention or MHA extension with AR-causal vs DM-bidirectional concat logic; possible mRoPE helper).
- MCP exposure in `n_orca/mcp_server.py` (extend `build_world_model` with `variant="mot_denoise" | "mot_reasoner" | ...` or dedicated tool; pass diffusion/timestep params).
- Skill updates in `.claude/skills/n-orca-build-world-model/SKILL.md` (add variants to list/desc/examples).
- Verifiable examples: `examples/cosmos-mot-denoise-step.n.orca.md` (+ .mmd), `cosmos-mot-reasoner.n.orca.md`, optional omni full; generated + `n-orca verify` clean + PyTorch compile runnable (stub).
- Tests in `tests/test_sae_and_world_models.py` (topology, state tensors for timestep/dual, flow correctness, compile roundtrip).
- Docs: README (builders section), `docs/proposed-sae-extensions.md` (mark implemented with date), AGENTS.md / grammar if needed, cross-refs in OpenSpec.
- OpenSpec artifacts (this change) + later archive.
- (Future, low priority in this change: HF adapter stubs if Cosmos config parseable; full 3D video tensor threading already supported.)

(Actual topology / op details designed in design.md. Prefer reusing/extending existing (Linear, MHA, Patch/TubeletEmbed, Positional, Add) + minimal new for MoT specifics. Diffusion iteration and full encoders external like temporal hidden carry.)

## Capabilities

Affects the "world model" capability / examples surface and "advanced architecture patterns" (dual-stream, diffusion conditioning, multimodal joint attention). Directly enables n-orca as host spec for Cosmos-style substrates in new SAE interpretability work.

## Scope

In (n-orca only):
- Builders + ops + MCP + skills + 1-3 examples + tests + docs/OpenSpec updates.
- Verifiable high-level MoT pattern specs (per-denoise-step DAG + external loop for diffusion schedule; reasoner tower; joint conditioning).

Out:
- Actual Cosmos weights / full training / inference (use HF diffusers / cosmos-framework in consumer repos).
- Production-scale video/action tokenization or mRoPE impl (stubs + notes).
- Creating the world-sae repo itself (separate follow-up scaffold after n-orca support; or parallel).
- Changes to econ-sae/sm-sae/sae-forge/polygram (read-only research + later coordinated PRs once n-orca green; they will consume the .n.orca.md specs).
- New SAE variant specific to diffusion (reuse existing families first).

## Related

- Subagent investigation report (2026-06-02, id 019e8a6d-f3b7-77f1-baf1-8786d6306436): primary source for Cosmos 3 / MoT details, GT potential, n-orca plan, risks.
- econ-sae (TemporalWorldModel + regime SAEs; uses n-orca temporal + attn world models for h1 substrate).
- sm-sae (exact GT factorization for particle physics; contrast/complement for multimodal physics).
- sae-forge (WorldModel protocol + n-orca .n.orca consumption for host archs; adapters).
- polygram (dictionary learning target on activations from n-orca hosts).
- Prior n-orca: add-temporal-world-model (explicit state carry pattern; per-step + external recurrence), advanced SAE examples (attn_topk, supervised_topk, gated).
- NVIDIA sources: Cosmos technical report (research.nvidia.com/.../cosmos3/technical-report.pdf), github.com/NVIDIA/cosmos , HF nvidia/Cosmos3-Nano etc.

This change makes n-orca the canonical way to define (and generate/verify/compile) the NN architectures for the next generation of world-model SAE experiments.

## Initial Acceptance Criteria (for builders + examples + integration)

- New builder(s) e.g. `mot_denoise_step(d_model=256, n_heads=4, timestep_dim=128, ...)` must produce a verifiable Architecture:
  - Explicit tensors for AR stream, DM/noisy stream, timestep t, outputs (denoised or reasoner logits).
  - Dual-tower structure via layers (separate LN/MLP per tower) + joint attention flow (AR causal; DM attends AR+DM).
  - Timestep conditioning (embed + add/inject to DM path).
  - Compatible with existing compiler (PyTorch emits runnable nn.Module stub; Mermaid).
- Example(s) `examples/cosmos-mot-*.n.orca.md` + .mmd must:
  - `n-orca verify` → VALID.
  - `n-orca compile pytorch` produces importable/forwardable module.
  - Demonstrate MoT patterns (dual streams, joint cross-conditioning, timestep).
- MCP `build_world_model(variant="mot_denoise" | "mot_reasoner", ...)` (or equivalent) exposes key hypers (d_model, n_heads, timestep_dim, etc.).
- Tests cover new builders (has expected layers/flows/tensors for dual + timestep; compiles; invariants hold for causal integrity where documented).
- Docs updated (README lists new variants under world models; proposed-sae-extensions marked "Implemented"; skill descs list cosmos/MoT).
- OpenSpec tasks completed for research/design/impl/examples; change ready to archive after green.
- No breakage to existing 4 world models / 6 SAEs / 140+ tests / all examples VALID.
- (Stretch) Note in AGENTS.md / IMPROVEMENT_GOAL about new Physical AI / multimodal world model use case for future -sae siblings.

Focus on patterns that allow the SAE substrate to capture physics concepts, diffusion progression, and cross-modal (reasoner↔generator) conjunctive features. Keep scopes small (toy sizes); real Cosmos (16B/64B) is for downstream use via HF + hooks.
