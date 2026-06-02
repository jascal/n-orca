# n-orca Long-Term Improvement Goal

**Goal**: Carefully but indefinitely improve the n-orca repository (Neural-network Orchestrated Architecture Language).

## Core Principles (always follow)
- **Safety and Care first**: All changes must be small, incremental, well-tested, and non-breaking unless explicitly justified. Never merge or consider work "done" until:
  - Full test suite passes (`python -m pytest -q`).
  - All shipped examples verify successfully via the CLI.
  - New features have corresponding tests and examples.
- **Agentic and Spec-Driven**: Prefer and extend OpenSpec for non-trivial changes. Enhance the MCP server and `.claude/skills/` so that Grok, Claude, and other agents can drive improvements effectively.
- **Ecosystem Alignment**: n-orca should be the canonical, verified source of truth for neural architectures used across the workspace (especially econ-sae, sm-sae, polygram, sae-forge world models and SAEs). Changes should improve cross-project consistency.
- **Quality over Speed**: Prioritize:
  1. Documentation, examples, and discoverability.
  2. Test coverage and robustness (edge cases for multi-output, 3D tensors, complex flows, temporal state).
  3. Error messages and LLM-friendly suggestions.
  4. New useful builders (temporal/recurrent, more HF adapters, additional ops).
- **Indefinite but Sustainable**: Work in small sessions. Use todos to track. Log progress. If blocked, clearly document the reason and next steps for the user.
- **No Unnecessary Breaking Changes**: Maintain backward compatibility for builders, MCP tools, and the language surface.

## Focus Areas (prioritized)
- Maintain 140+ passing tests and full example verification at all times.
- Expand world model and SAE library (temporal variants are high priority per econ-sae research).
- Improve agent tooling (skills, MCP, example generators).
- Documentation and onboarding (README, docs/, skills descriptions, AGENTS.md if appropriate).
- Cross-project integration and usage in sibling repos.
- Internal quality (compiler robustness, op coverage, performance of verify/compile).
- Release hygiene (version bumps, changelogs, CI robustness).

## How to Work on This Goal
1. Start each improvement cycle by reading this file + current `openspec/changes/` + `todo` state.
2. Run audits: full tests, example verification, `n-orca info` / build on key examples.
3. Pick the next smallest safe task from the active todo list.
4. Implement, test thoroughly, update docs/examples as needed.
5. Mark todos, update this goal file or use tools to log progress.
6. If a cycle ends with no safe work possible, clearly state blockers.

This goal is intended to be driven by recurring schedulers, background sub-agents, and interactive sessions with Grok. It should run "carefully but indefinitely" until the user decides the project has reached a stable mature state.

Last updated: 2026-06-03 (added OpenSpec for Cosmos 3 / MoT after subagent investigation; temporal PR #3 pending review).
## Recent Activity

**2026-06-02 (initial setup by Grok during n-orca lead handoff)**:
- Created this IMPROVEMENT_GOAL.md file with core principles, focus areas, and working instructions.
- Created structured todo list covering infrastructure, audit, docs, builders (esp. temporal), agent features, integration, quality, release.
- Set up recurring scheduler (ID 019e8a647036) that fires every 3 hours with a detailed self-driving improvement prompt. The prompt will audit state, pick safe next todo, implement/test/update, log here.
- Updated todo 01 to completed and 02 (audit) to in_progress.
- Performed initial audit: 140 tests passing (Python 3.11 venv), example verification healthy, OpenSpec state good (temporal stub active + archive), n-orca skills present (4 dedicated), goal file and scheduler confirmed active.
- Scheduler is now the primary mechanism for "carefully but indefinitely" running improvement cycles.

Next scheduler fire expected around 2026-06-03T01:11 UTC. Cycles will be conservative and test-first.

**2026-06-02 (continued setup)**:
- Created AGENTS.md (modeled on polygram style) tailored to n-orca: covers what the repo is, OpenSpec+agentic workflow, the new skills/MCP, dependency contracts with econ-sae ecosystem, file layout, dev commands, conventions, and specific guidance for agents on how to use the IMPROVEMENT_GOAL + scheduler + todos.
- This is a safe, high-value documentation improvement that makes future agent work (including the recurring scheduler) much more effective.
- Marked relevant todo progress.

The indefinite careful improvement system is now bootstrapped:
- Persistent goal definition (this file)
- Structured todos
- Recurring 3h scheduler with self-driving prompt
- AGENTS.md for orientation

Scheduler will continue driving cycles. Human can also trigger by asking me to "run the next improvement cycle" or similar.

**2026-06-02 (scheduled cycle execution)**:
- Completed full Audit Phase per scheduler prompt:
  - Env: .venv with Python 3.11.15 active.
  - Tests: 140 passed, 1 harmless torch/numpy warning. (Repeated after changes: still 140/140.)
  - Example verification: All examples (conv-classifier, econ-*, sae- variants including new attn/gated/supervised, hf-generated bert/gpt2/etc., residual, simple-mlp, tiny-vit, transformer, unet) returned "VALID". No failures.
  - Inspected key areas via list_dir, read_file, grep: n_orca/world_models.py (only baseline/deep/attn implemented), sae.py (advanced SAEs present), mcp_server.py (build_world_model supports 3 variants, no temporal), .claude/skills/ (n-orca-build-world-model mentions "temporal substrates" in desc but lists only 3), openspec/changes/ (add-temporal-world-model active with proposal+tasks; previous archived), README has OpenSpec callout, AGENTS.md exists (recent), test_sae_and_world_models.py covers the 3 world models + SAEs, no "temporal" in core code (only docs/OpenSpec).
  - Reviewed active OpenSpec: proposal details econ-sae motivation, acceptance criteria for state carry/attention composition; tasks start with research 1.1-1.3 (design.md pending).
- Current health summary: Excellent (tests/examples green, 0.2.0, skills/MCP/OpenSpec in place). Gaps: temporal world model not implemented (high priority per goal/OpenSpec/econ-sae); skill desc slightly ahead of impl; no design.md for temporal OpenSpec yet; potential for more temporal/robustness in future.
- Planning: Reviewed IMPROVEMENT_GOAL principles (safety/tests first, OpenSpec, econ-sae alignment, incremental). From todo list (improve-04 in progress for builders/temporal), picked smallest/safest/highest-value aligned task: create initial design.md for add-temporal-world-model (maps to OpenSpec task 1.3). Justified: Pure docs addition (no code edit, zero risk to tests/examples), directly advances active OpenSpec and goal's top priority (temporal builders), breaks down large research into actionable design without needing sibling code changes yet (read-only inspection sufficient for now). Conservative choice over jumping to impl.
- Execution: Created openspec/changes/add-temporal-world-model/design.md with full sections (Context from proposal, Goals/Non-Goals, Decisions on explicit hidden state per-step + attn composition, Risks, Migration, Open Questions). Used write tool after reading proposal. Re-ran tests + relevant example verify post-edit: 140 pass, example VALID, file confirmed. Updated OpenSpec tasks.md to mark 1.3 [x]. No new work discovered; no blockers.
- Progress: Updated todo_write (04a completed, 04 noted). This cycle focused on design scaffolding for temporal – safe first step toward builder.
- No blockers. Next logical would be research/1.1 or impl, but per safety, this was ideal for session.

Next scheduled run ~3 hours from cycle start. All rules followed (tests/examples green at start/end, only n-orca changes, small/doc-focused, verified after edit).

## SDLC & Collaboration Process (Git, Commits, PRs, Feedback)

The recurring improvement cycles focus on **safe development work**. The full software development lifecycle (including source control and collaboration) is explicitly part of the process when appropriate:

### When to perform Git / PR steps (at the end of Execution, before final logging)
- After a task is implemented + all tests pass + relevant examples verify:
  - If the change is **significant, user-facing, or touches public API** (new builders, MCP changes, behavior changes, new skills, etc.):
    - Create or switch to a feature branch (e.g. `grok/improve-<short-name>` or follow existing convention).
    - Commit the changes with a clear, descriptive commit message (reference the todo id and goal where relevant).
    - Push the branch.
    - Create a Pull Request using the `gh` CLI (`gh pr create`).
    - Include a good PR description summarizing the change, motivation (from goal/OpenSpec), and testing performed.
  - If the change is **small, internal, documentation-only, or low-risk**:
    - It may be committed directly (or via a small PR) at the agent's discretion, but still prefer a PR for auditability unless the change is trivial.
- The cycle **does not block or wait** for human review/feedback. 
  - The cycle ends after the PR is created (or after committing if no PR was warranted).
  - "Waiting for feedback" happens outside the automated timer. 
  - The human can paste review comments in a future session and ask the agent to "address PR #X feedback" or "run the next improvement cycle to incorporate review comments".
  - A future cycle can then check out the PR branch (if needed), make fixes, push, and update the PR.

### Tools available for SDLC steps
- `run_terminal_command` with `git` (branch, add, commit, push, checkout, etc.).
- `run_terminal_command` with `gh` (gh pr create, gh pr comment, gh pr merge, etc.).
- All changes must still follow the Safety Rules (tests + example verification must be green before considering the task done).

### Why this model?
- Keeps the 3-hour cycles moving and productive ("carefully but indefinitely").
- Ensures changes are properly reviewed via GitHub PRs when they matter.
- Avoids the scheduler stalling forever waiting for humans.
- Matches how real agent-driven development works in practice (as demonstrated in prior direct requests for branch/PR work on this repo).

This section was added to make the full SDLC (commit → PR → review loop) an explicit, first-class part of the scheduled improvement process.

**2026-06-02 (SDLC clarification)**:
- Added a new dedicated section "SDLC & Collaboration Process (Git, Commits, PRs, Feedback)" to this file. It clearly defines:
  - When to create branches, commit, push, and open PRs (significant/user-facing changes get PRs; prefer PRs for visibility).
  - That cycles do **not** wait inside the timer for human feedback.
  - Feedback is handled by future cycles triggered by the human pasting review comments.
- Deleted the previous scheduler (ID 019e8a647036).
- Created a new scheduler (ID 019e8a68c057) with an updated prompt that now has an explicit "**4. SDLC Phase**" step. The prompt instructs the agent to consult the new SDLC section in this goal file and perform git + gh pr create when appropriate at the end of Execution.
- This makes committing code, creating PRs, and the overall collaboration loop a first-class, documented part of the recurring 3-hour improvement schedule.

Future scheduled cycles will now automatically include the full SDLC steps (where they make sense) instead of treating them as separate user requests.

**2026-06-03 (scheduled cycle)**:
- Completed full Audit Phase: .venv/Python 3.11.15, 140 tests pass (1 warning), all ~21 examples verify VALID (including previous new sae-* and econ-*). Inspected code (world_models has 3 builders; mcp/skill list 3; no temporal impl yet; design.md exists from prior; OpenSpec active has proposal+tasks+design; README has callout; AGENTS exists). Reviewed active OpenSpec: 1.3 done, others pending for temporal.
- Planning: From todos (04 in progress for temporal builders; OpenSpec tasks 1.1 research/2.1 builder pending), picked next: implement basic temporal_world_model per design + review of econ-sae TemporalWorldModel (which uses attn + GRU for cross-period state on (B,T,N) trajectories, h1 after GRU). Justified: Directly advances highest priority (temporal per goal/OpenSpec/econ-sae), smallest safe step post-design (basic impl with explicit hidden state using existing ops + linear for state; no new ops needed yet; will verify/compile). Conservative: no sibling edits, tests/examples will be re-run, state not mixed into main path yet (per design "starting point").
- Execution: 
  - Added temporal_world_model() to world_models.py (attn path + separate hidden_in -> hidden_update linear -> hidden_out; tensors include hidden; docstring references econ-sae/design; no output_shape invariant due to dual outputs). (Note: layer later renamed from state_update per PR #3 review nit.)
  - Wired "temporal" to mcp_server.py (added gru_hidden param, case in build, updated docstring).
  - Updated n-orca-build-world-model skill (desc + variants + example).
  - Added test_temporal_world_model_has_state_tensors (checks tensors and ops).
  - Generated examples/econ-temporal-world-model.n.orca.md + .mmd via builder/render (verifies VALID).
  - Updated world_models docstring, OpenSpec tasks.md (marked 2.1/2.2/3.1).
  - Re-ran tests (141 passed, new test included) + new example verify after edits/groups.
- SDLC Phase: Significant (new public builder, MCP/skill update, new example/test, OpenSpec advance). Created branch grok/add-temporal-world-model, committed (7 files), pushed, gh pr create -> https://github.com/jascal/n-orca/pull/3 . Per rules: no wait for feedback in cycle.
- Progress: todo_write updated (04/04b completed). Appended here.
- No blockers. Full green.

Next scheduled ~3h. All safety/SDLC followed (tests first, only n-orca, PR created for the change, logged).

**2026-06-03 (Cosmos 3 + MoT subagent investigation + OpenSpec bootstrap)**:
- Per explicit user request ("spin off a separate agent to investigate ... Cosmos 3 world model and in particular the Mixture of Transformers pattern ... whether that could perhaps be another -sae repo and at the same time another set of NN designs to be able to include examples for and support generating from within n-orca"), used todo_write to record the task, then spawn_subagent (general-purpose, background, detailed standalone prompt including: read n-orca IMPROVEMENT_GOAL + source (world_models with temporal, mcp, compiler/verifier/ops, skills, tests, examples, econ-sae TemporalWorldModel for patterns), research Cosmos 3 (NVIDIA WFM for physical AI/video/world prediction), focus on MoT for diffusion+multimodal (dual-tower AR+DM, joint attn, timestep, mRoPE), assess -sae fit (GT factors like econ/sm: physics, conjunctive, diffusion stages, regime; controllable via prompts/actions/synthetic data), assess n-orca fit (new layers/flows/builders e.g. mot_world_model, timestep embed, fusion paths; invariants for sparsity/noise; example + verify + MCP/skill; possible OpenSpec or new cosmos-sae skeleton). Instructed to use todo_write internally, follow safety (n-orca only for now), produce findings report + concrete recs + draft artifacts.
- Subagent completed autonomously (ID 019e8a6d-f3b7-77f1-baf1-8786d6306436; ~214s, 52 tool calls incl. web_search/web_fetch for NVIDIA report/HF/GitHub + full workspace list_dir/read/grep + runpy on n-orca + MCP search; 1 turn). Status: completed, exit 0. Delivered **Comprehensive Research Report** (grounded, with sections 1-8 + citations plan + absolute paths + code sketches).
  - Cosmos 3: open omnimodal WFM platform (reasoner + generator + policy + sim in unified MoT); Nano/Super (16B/64B dual ~8+8 / 32+32); HF nvidia/Cosmos3-*; github.com/NVIDIA/cosmos + cosmos-framework. Physics-aware video/world/action gen from text/image/video/action. Curator/Evaluator/etc.
  - MoT (key innovation): not classic MoE; dual-tower per block (AR reasoner causal + DM generator bidirectional) sharing joint self-attn (AR: causal over AR only; DM: full over concat AR+DM). AR prefix (discrete) + DM (continuous noisy latents). Flow-matching (not pure diffusion). Co-init from VLM. "Reasons first then generates."
  - Arch details: encoders (ViT AR, VAE DM for video/audio, geometric action), 3D mRoPE + FPS modulation + modality offsets + fixed temporal gap, latent space denoising, flexible modes (t2i/t2v/i2v/v2v + sound/action/policy), embodiments.
  - SAE GT potential: **High**. Controllable rollouts via open models + prompts/actions + hooks (mid-block reasoner hiddens + DM at specific t/noise). Derivable GT: physics (collision/gravity/occlusion/inertia), objects/poses, spatio-temporal motion, multimodal alignment (text-video-action), diffusion factors (coarse→detail), action/policy (dynamics/grasp). Tiers like econ (single-modality, cross-modal conjunctive, regime/scene, higher-order physics+embodiment). Complements sm-sae (exact particle) + econ-sae (macro sim + temporal). Enables new world-sae/cosmos-sae for polysemanticity tests on realistic Physical AI data.
  - New -sae repo: **Highly feasible + recommended** ("world-sae" for generality to other WFMs). Skeleton: worldsae/generator.py (diffusers/vLLM-Omni wrapper + return_acts hooks), world_model.py (WorldModel protocol + n-orca .n.orca passthrough), ground_truth.py (vocab + tiers + scoring), sae/ (data/models/train/eval/forge/polygram bridges), scripts, tests, docs (with n-orca-generated .n.orca.md). Mirrors econ/sm exactly.
  - n-orca integration: **Perfect fit**. Concrete plan (§6): new builders mot_denoise_step / mot_reasoner / mot_omni in world_models.py (tensors ar_x/dm_x/t , dual LN/MLP paths, joint attn flow, timestep); minimal new ops (TimestepEmbed, DualStreamJointAttention with concat/mask special-case like MHA); MCP/skill updates; examples cosmos-mot-*.n.orca.md (verifiable, compile to stub); tests; docs/README/AGENTS; OpenSpec "add-cosmos-mot-world-models" first (then impl slice); tie to -sae via n-orca as "verifiable spec layer".
  - Risks (detailed): scale/activation volume (start Nano + short + subsets), GT is derived not exact (use reasoner + synthetic meta + probes), diffusion not pure DAG (per-step + external), evolving API (pin to report), scope (keep toy verifiable; don't reimpl Cosmos), etc. Overall: high upside, low-risk if scoped.
  - Recs: 1. Immediate n-orca OpenSpec-propose + impl (smallest safe: mot_denoise_step + ops + 1-2 ex + test + MCP). 2. Bootstrap world-sae skeleton. 3. Cross (sae-forge etc.). Prioritize n-orca first so specs exist for the -sae.
- Action taken: Created todo items (improve-09 completed for investigation; improve-10 in_progress for OpenSpec; 11-15 pending for impl/examples/docs/world-sae scaffold/cross). 
- Created n-orca/openspec/changes/add-cosmos-mot-world-models/ with full proposal.md (Why from Cosmos value + subagent + econ alignment; What/scope/acceptance mirroring temporal + report sketches; Initial Acceptance Criteria for verifiable dual-tower+timestep builders/examples/MCP), design.md (Context with MoT breakdown + subagent tie-in; Goals/NonGoals; 5 key Decisions on per-step DAG + explicit streams/timestep + external loop + minimal new ops + hypers; Risks from report; Migration; Open Questions), tasks.md (1. Research [x] via subagent+review+design; 2. Impl builders/ops/MCP; 3. Examples+tests with verify; 4. Docs/skills/README; 5. SDLC + archive + world-sae scaffold + cross-project). Followed temporal OpenSpec structure exactly; no unnecessary files (subagent created none).
- Updated this IMPROVEMENT_GOAL.md (Recent Activity) + README? (pending; will note active OpenSpec).
- No code changes or tests yet (pure scaffolding like early temporal design phase). No sibling edits. All per n-orca safety (tests/examples will be green before any impl).
- Current OpenSpec state: add-temporal-world-model (some tasks remain), add-cosmos-mot-world-models (new, research/design done).
- Next: Scheduler or user-triggered cycle can pick improve-10 (or 2.3 etc from temporal); implement smallest (e.g. stub builder + example that verifies, per "if builder then example+verify" rule). Subagent results feed directly into design/plan. High-value expansion of n-orca's NN surface for diffusion/multimodal/world models while enabling potential new -sae direction.
- Blockers: None. PR #3 (temporal) review addressed + merged in follow-up step (see next entry).

Next scheduled ~3h (or manual "run next improvement cycle"). All rules followed (OpenSpec first for non-trivial, subagent for parallel research without stalling main, todos+goal logging, n-orca only, safety first).

**2026-06-03 (address PR #3 review nits + merge temporal; restore + commit cosmos OpenSpec work)**:
- Received pasted ✅ PR Review #3 (Approved, "Merge as-is (or after the tiny naming/doc nits)").
- Checked out the PR branch (grok/add-temporal-world-model), stashed unrelated post-PR work (cosmos OpenSpec + goal/AGENTS/README updates), made targeted fixes:
  1. Renamed internal layer `state_update` -> `hidden_update` in temporal_world_model (world_models.py:287) + flows + comments. Regenerated examples to match (nit 1).
  2. Enhanced `test_temporal_world_model_has_state_tensors` to assert `hidden_out` shape + call-time gru_hidden dim via hyperparameters (nit 2; names were already both asserted).
  3. Expanded docstring + comments in temporal_world_model to explicitly call out "Full recurrent logic (e.g. actual GRU cell) is planned for a follow-up" (nit 3).
  4. Nit 4 (usage comment in example for threading hidden) noted as future per review suggestion; example already shows the tensors/flows clearly.
- Re-ran: specific test green, full sae_and_world_models module, full pytest (141 passed), `n-orca verify` on the temporal example (VALID).
- Committed the 4 files (world_models, test, 2 examples) on the PR branch with detailed message; `git push`.
- `gh pr merge 3 --merge` (with body summarizing nits addressed). PR now MERGED (confirmed via gh pr view json: state=MERGED, mergeCommit, mergedAt).
- Switched to main, `git fetch && git pull --ff-only` (brought in temporal + nit fix commit).
- `git stash pop` (restored cosmos OpenSpec dir + IMPROVEMENT_GOAL + AGENTS + README mod + temporal design.md).
- Synced docs for consistency: updated references in temporal design.md and the new cosmos-mot design.md from "state_update" to "hidden_update".
- Updated this goal file: fixed historical bullet, appended this entry; also updated the cosmos entry's "still open" note.
- Updated temporal OpenSpec tasks.md (added note on nit polish under 3.2).
- Staged + will commit the net new work (cosmos OpenSpec as primary deliverable from subagent processing + goal logging + doc syncs + feedback address) as follow-up on main. This keeps the SDLC visible.
- No blockers. Full green. PR #3 complete. The temporal builder is now merged, nits addressed, and the parallel cosmos investigation OpenSpec is ready for next cycles to implement (improve-11 etc.).
- Per review: "Solid continuation of the handoff — looking forward to the next refinements (full GRU, shape inference..., integration examples)."

All per safety (tests/examples re-verified after edits), SDLC (branch/push/merge via gh for the PR), and goal process.

**2026-06-03 (q-orca-kb expansion: n-orca-lang room)**:
- Coordinated with user to expand the shared q-orca-kb (now beyond pure quantum, with orca-lang / q-orca-lang rooms) to include an `n-orca-lang` room under `q-orca-implementations` wing.
- Added `n-orca-lang-wiki` CrawlConfig in q-orca-kb/q_orca_kb/crawlers/site_configs.py (deepwiki.com/jascal/n-orca seed, same patterns as siblings, room="n-orca-lang").
- Updated tests (test_mcp_crawl_site.py) and tool descriptions in q-orca-kb MCP.
- Seeded the room immediately with comprehensive local content via direct indexer: all main .py modules (world_models with temporal, sae, ops, verifier, compiler, mcp_server, hf adapters, parser, render, ast), docs (README, AGENTS, IMPROVEMENT_GOAL, grammar, proposed-sae-extensions, verification), and all examples (.n.orca.md + .mmd).
- Ran the new crawl_site-equivalent for n-orca-lang-wiki (partial due to DeepWiki JS "Loading..." pages; main page indexed).
- Result: 62 sources, 533 drawers in the room. Semantic search works well for n-orca concepts (temporal_world_model, build_world_model, hidden_update, etc.).
- Updated n-orca/AGENTS.md to document the new KB integration for agents.
- Note: MCP stdio transport for q-orca-kb was closed in this session after process management (doctor can start fresh instances; Claude agent informed and will drive crawls via its MCP connection). Direct Python access to the palace works. Full crawl via MCP "n-orca-lang-wiki" can be triggered by Claude or future cycles.
- This makes n-orca first-class in the shared Orca-family knowledge base for papers, docs, and cross-project grounding.

Next scheduled ~3h (or manual "run next improvement cycle"). All rules followed.
