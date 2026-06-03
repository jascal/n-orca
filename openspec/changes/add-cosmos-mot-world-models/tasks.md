## 1. Research & Design
- [x] 1.1 Subagent investigation of Cosmos 3 (omnimodal WFM), MoT dual-tower + joint attn details, diffusion integration, multimodal posembs, world model specifics, SAE GT potential (physics concepts, diffusion stages, conjunctive cross-modal, tiers), new -sae repo feasibility (world-sae/cosmos-sae skeleton), and n-orca integration plan (builders, ops, examples, MCP). Report delivered with citations, sketches, risks (2026-06-02, subagent 019e8a6d-f3b7-77f1-baf1-8786d6306436). See subagent output + linked NVIDIA report.
- [x] 1.2 Review n-orca current state (world_models.py temporal/attn, ops/spec.py vision-ready ops like Tubelet/Patch/Positional, compiler MHA special case + multi-output/3D, mcp_server, skills, econ-sae TemporalWorldModel patterns for state carry analogy, sae-forge WorldModel protocol). Confirm fit for MoT (per-step DAG + external denoise loop; dual streams via tensors/flows).
- [x] 1.3 Create design.md for the chosen approach (per-step explicit AR/DM + timestep tensors; dual-tower via layers + joint attn op/flow; reuse + minimal new ops; external iteration like temporal; toy verifiable specs). Reference proposal, subagent report §6, and temporal design precedent.

## 2. Implementation (n-orca core)
- [x] 2.1 Add `mot_denoise_step(...)` (primary) + `mot_reasoner_only(...)` (and optional fuller) builders in `n_orca/world_models.py`. Return Architecture with dual-stream tensors, timestep, appropriate layers/flows for MoT (AR causal path, DM joint path, timestep embed + inject). Docstring references subagent/Cosmos report + "external diffusion schedule like temporal hidden carry". Handle dual-output or main DM out. (Done in scheduler cycle: builder using existing ops per design "smallest slice", + test + ex + verify green; mcp wire 2.3 later.)
- [ ] 2.2 Add minimal new ops (or extend) in `n_orca/ops/spec.py`: `TimestepEmbed` (or DiffusionTimestep; sinusoidal/learned, pytorch_call), `DualStreamJointAttention` (params for d_model/n_heads; special _torch_call handling concat AR/DM K/V + causal_AR vs bidirectional_DM masks; register). Update any shape inference if needed. (Can start with flows using existing ops + document joint in example if op too heavy for first slice.)
- [x] 2.3 Wire new variant(s) into `n_orca/mcp_server.py::build_world_model` (add cases for "mot_denoise", "mot_reasoner", etc. + relevant kwargs like timestep_dim; update docstring and error msgs). Ensure CLI/MCP exposure works.
- [ ] 2.4 (Optional in slice) Minor verifier tweaks if new invariants (e.g. causal note) or shape rules for dual streams.

## 3. Examples & Tests
- [x] 3.1 Add `examples/cosmos-mot-denoise-step.n.orca.md` + `.mmd` (generated via builder + render + compile_mermaid; or manual from design sketch). Must `n-orca verify` → VALID and compile to PyTorch runnable stub. (Generated + VALID in 2.1 cycle.)
- [ ] 3.2 Add at least one more (e.g. `cosmos-mot-reasoner.n.orca.md`) demonstrating AR-only causal tower.
- [x] 3.3 Add tests in `tests/test_sae_and_world_models.py` (e.g. `test_mot_world_model_has_dual_streams_and_timestep()`, checks tensors/layers/flows include joint/timestep, MHA or custom, compiles; similar to temporal test). Total tests increase; full suite green. (Added + 142p in cycle.)
- [x] 3.4 Full `n-orca verify --all-examples` (or equivalent sweep) + `python -m pytest -q` (must stay 100% pass + all VALID, including prior econ/temporal/sae examples). (Full sweep green post change.)

## 4. Docs & Sync
- [ ] 4.1 Update `n-orca/.claude/skills/n-orca-build-world-model/SKILL.md` (add MoT/Cosmos variants to description, list, example usage; note for Physical AI / diffusion world models + future -sae).
- [x] 4.2 Update README.md (SAE & World Model builders section; list new variants; Current OpenSpec Changes callout; ecosystem note linking to subagent report / new possibilities for world-sae). (Done in scheduler cycle 2026-06-03 via pure docs PR#5; skill 4.1 was pre-updated in 2.1 slice.)
- [ ] 4.3 Mark status in `docs/proposed-sae-extensions.md` (or add MoT section) with date; cross-ref "use n-orca as shared source of truth".
- [ ] 4.4 Update AGENTS.md if process notes needed; optionally IMPROVEMENT_GOAL focus areas mention Physical AI world models.
- [ ] 4.5 (Post-impl) Consider scripts/generate_* or sibling doc regen if they cover world models.

## 5. OpenSpec close + follow-ups
- [ ] 5.1 Mark all tasks [x]; ensure acceptance criteria met (verifiable builders/examples, MCP, tests green, docs).
- [ ] 5.2 Create feature branch (e.g. grok/add-cosmos-mot-...), commit (referencing this OpenSpec + improve-10/11), push, gh pr create (include summary of subagent findings + motivation + test evidence).
- [ ] 5.3 Address any PR feedback in follow-up cycle(s); merge when ready.
- [ ] 5.4 Archive the change (openspec archive ... or manual to archive/YYYY-MM-DD-add-cosmos-mot-world-models/).
- [ ] 5.5 (Separate but related) Scaffold minimal world-sae (or cosmos-sae) repo at `/Users/allans/code/world-sae` (pyproject, worldsae/generator.py stub using diffusers + HF note for Cosmos3-Nano, worldsae/world_model.py with n-orca import + .n.orca.md reference, ground_truth.py with initial vocab from subagent §4, basic scripts/tests). Mirror econ-sae layout. Add README tying back to n-orca as spec source. (Can be its own OpenSpec in the new repo or simple bootstrap.)
- [ ] 5.6 Later cross-project: sae-forge adapter for cosmos host, polygram demo on sample acts, econ-sae/sm-sae docs mentions (read-only first, then PRs).

(Research complete via dedicated subagent. Impl follows safety: read before edit, test+verify after groups, only n-orca for this change, log to goal/todos, SDLC/PR for the significant addition. Subagent report is the authoritative reference for details.)
