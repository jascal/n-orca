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
3. **Check open PRs first (critical SDLC rule)**: Always run `gh pr list --state open --json number,title,headRefName,files` (using run_terminal_command or gh tool) to list all open PRs and the files they touch. Only pick a task if:
   - Its planned changes (files in OpenSpec, code in n_orca/, tests, examples, docs touched by the task) do **not** overlap with any files changed in an open PR.
   - The task is not a re-implementation of work already submitted in an open PR (e.g., duplicate builder or same OpenSpec slice).
   - If the task's work is already in an open PR, skip it. Instead, wait for human to paste review feedback for a future cycle to address, or pick a completely unrelated safe task (e.g., pure docs on untouched files, or KB indexing).
   This rule prevents duplicate PRs, re-work, and scheduler cycles fighting open changes. It takes precedence over "smallest task".
4. Pick the next smallest safe task from the active todo list (subject to the open-PRs check above).
5. Implement, test thoroughly, update docs/examples as needed.
6. Mark todos, update this goal file or use tools to log progress.
7. If a cycle ends with no safe work possible, clearly state blockers.

This goal is intended to be driven by recurring schedulers, background sub-agents, and interactive sessions with Grok. It should run "carefully but indefinitely" until the user decides the project has reached a stable mature state.

Last updated: 2026-06-03 (scheduler cycle: mcp timestep_dim for mot + PR#8; open PRs rule followed (none); 142 tests; cosmos 2.2/3.2/4.x + temporal remain pending).
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

### Avoiding Duplicate Work on Open PRs (new rule to prevent re-execution of in-flight changes)
- **Before every Planning phase** (step 3 in "How to Work", including inside the embedded scheduler prompt): Explicitly query open PRs.
- Never select a task, create a branch, or open a new PR for work that touches files or OpenSpec changes already present in an open PR.
- The scheduler must treat "work already in an open PR" as "blocked for new implementation". The correct action for in-flight work is to let the PR be reviewed/merged, then (if feedback is provided by pasting the review) use a future cycle to address nits on that PR's branch.
- This rule was added after a scheduler cycle duplicated the `mot_denoise_step` work (creating PR#7 while PR#6 was open/being merged), leading to wasted effort and review overhead.
- In practice: Run the gh list command early in the cycle. If all high-priority tasks (e.g. next OpenSpec slice) are covered by open PRs, fall back to safe unrelated work (e.g. KB paper indexing into n-orca-lang room, pure docs on untouched areas, or low-risk test additions elsewhere) or document "no safe non-overlapping work; waiting on PRs X,Y".

### Why this model?
- Keeps the 3-hour cycles moving and productive ("carefully but indefinitely").
- Ensures changes are properly reviewed via GitHub PRs when they matter.
- Avoids the scheduler stalling forever waiting for humans.
- Matches how real agent-driven development works in practice (as demonstrated in prior direct requests for branch/PR work on this repo).
- Prevents the exact problem of re-audits "forgetting" that a PR is already in flight and re-doing the same slice.

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

**2026-06-03 (scheduled cycle - start cosmos MoT impl slice 2.1 mot_denoise_step + full audit + KB + SDLC PR#6)**:
- Started per scheduler prompt exactly: "Start now with full audit + read goal + openspec + todos." + use search_tool for q-orca-kb/n-orca then use_tool q-orca-kb__* (room="n-orca-lang", queries "temporal world model", "mixture of transformers", "diffusion", "language design", "neural architecture", "world model", "SAE", "formal methods for DAGs" etc) + "pick the NEXT SMALLEST, SAFEST, HIGHEST-VALUE task aligned with priorities (temporal remaining or start cosmos impl slice (smallest: builder for mot_denoise_step using existing ops where possible + test + example + verify))" + "Prefer pure docs/tests first if possible. Use q-orca-kb for research before impl" + "Always: read_file before edit, search_replace, re-audit green, 100% green small steps, n-orca only, SDLC if sig, log detailed, use KB n-orca-lang".
- Read full IMPROVEMENT_GOAL.md (194 lines, Recent ends at q-orca-kb expansion; Last updated old; focus temporal remain + cosmos), list_dir openspec/changes (active temporal + cosmos), read full proposal/design/tasks.md for both (temporal: 2.3/3.3/4.x/5.x remain; cosmos: all 2-5 pending, 2.1 smallest per design "start with one primary builder+ex using exist ops").
- todo_write (merge) to load/manage (legacy + new current-cycle, improve-11, cosmos-2.1, temporal-*, q-orca-kb-use etc).
- Full audits:
  - Tests: .venv/bin/python -m pytest -q --tb=no → 141 passed (1 harmless; confirms no mot yet, temporal present).
  - Examples verify: all top-level + sample hf `n-orca verify` → all VALID (econ-temporal present/VALID, no cosmos-mot ex yet; 16+ others).
  - n-orca cli + builds ok.
  - Code inspect: list_dir n_orca/ examples/; read_file world_models.py (ends at temporal, no mot, docstring old), mcp_server.py (temporal only, error msg no mot), test_sae... (has temporal test, ~25 tests, no mot), skill (temporal listed, no mot), README (OpenSpec callout has both, but builders list only to attn, badge 141, 4.x pending), AGENTS (KB noted); grep no "mot_denoise".
  - OpenSpec: cosmos 2.1 builder pending (design says use exist ops for first slice, note joint for later; external like temporal).
- KB (per prompt, before planning):
  - search_tool "q-orca-kb" (got list_seeds, server_status, kb_status, list_crawl_sites, index_seeds, search_papers etc), "n-orca", "q-orca-kb__list_sources".
  - use_tool q-orca-kb__list_sources (room=), q-orca-kb__search_papers (queries+room) → "Transport closed" (MCP known issue).
  - Direct fallback (mp_list_sources/mp_search + DEFAULT_PALACE): 67 sources in n-orca-lang (q-orca-implementations); samples the 5 lang-design arXiv PDFs; searches for "temporal world model hidden_update", "mixture of transformers OR MoT diffusion", "diffusion transformer", "formal methods DAG neural language design", "world model SAE" → hits n-orca self docs (goal/OpenSpec chunks, world_models descs, README) + language papers. Confirmed SITE_CONFIGS n-orca-lang-wiki present.
  - Grounding: n-orca self is strong source (temporal/MoT patterns in OpenSpec/goal); lang design papers useful for DSL; no external MoT/diffusion papers yet (self + design.md sufficient; noted for possible future q-orca-kb index of arXiv "diffusion transformer" etc, but not blocking smallest slice).
- Planning: Reviewed principles (safety/green gate, OpenSpec, n-orca source truth for econ + now cosmos per subagent, docs quality, small steps, KB first, no wait). From todos/OpenSpec (cosmos 2.1 pending smallest per prompt explicit "builder for mot_denoise_step using existing ops where possible + test + example + verify"; temporal 4.x/2.3 also but docs/ops heavier), KB (self grounded, no paper forcing new op now), picked: cosmos 2.1 mot_denoise_step (builder using exist ops + test + ex + verify; mcp/skill later). Justif: smallest/safest per design ("start with primary builder+ex", "reuse/extend existing... or document joint in ex if op too heavy for first slice"), directly priority "starting cosmos impl slice", advances OpenSpec acceptance (verifiable builder/ex/MCP later, no break), 0 risk to existing (additive like temporal), tests/ex will gate, prefer not pure docs since impl pending and prompt specifies the builder slice. (Could have done more README but cosmos 2.1 higher per "prioritize ... cosmos impl slice").
- Execution (read-first, only n-orca, search_replace, re-audit):
  - read_file (goal parts/tail, openspec tasks x2, world_models end+top, test end, skill, README sections, mcp, design for exact decisions).
  - search_replace: 1. world_models.py top docstring (add mot desc). 2. append full mot_denoise_step func (tensors AR/DM/t, layers AR causal + DM+ts, flows residual, docstring refs, hypers, no invariant, using exist ops + notes per design).
  - Then python: build+verify(VALID, 62k params)+render+mmd.
  - Generate ex: write examples/cosmos-mot-denoise-step.n.orca.md + .mmd via builder/render.
  - Verify new ex: VALID.
  - search_replace test file: add test_mot... after temporal test (assert dual tensors, MHA, ts_embed, valid).
  - Run: specific test pass; full pytest 142 pass; full ex sweep all VALID (new + prior).
  - Re-reads/greps for inspect.
- Re-audits after groups + final: 142p, all ex VALID (incl new 62,528p/Depth9), no errors.
- SDLC (sig: new public builder per goal): on grok/add-cosmos-mot-denoise-step (new name, prior existed), selective git add (4 files: world_models, test, 2 ex), commit (msg refs OpenSpec 2.1 + improve-11 + goal + subagent), push -u, gh pr create #6 (title/body with motiv, tests 142+VALID, KB details, OpenSpec refs, no wait). PR: https://github.com/jascal/n-orca/pull/6
- Progress: todo_write (improve-11, cosmos-2.1/3.1/3.3, q-orca-kb-use, current complete; noted mcp 2.3 pending). No new blockers surfaced. OpenSpec tasks not edited (builder is code; docs later per scope).
- Used KB n-orca-lang first (as required).
- No blockers. 100% green. Followed every rule (reads first, KB, smallest per explicit, verify before/after, SDLC, n-orca only, log).
- Next expected: future cycle (after PR#6 review/feedback if pasted) can do 2.3 mcp wire + update skill/error, or 3.2 more ex (reasoner), or 4.x docs (README builders list, since current lags even temporal), or temporal 2.3/3.3/4.x, or index more papers (e.g. arXiv on MoT/diffusion) to n-orca-lang room via q-orca-kb, or world-sae stub. PR#6 for review.

Next scheduled ~3h. All safety/SDLC/KB/prompt followed (tests+ex 100% green at start+end+re-audits, read/search_replace, conservative slice using exist ops, full log, n-orca source advanced for MoT).

**2026-06-03 (address PR #7 review nits post #6 merge)**:
- PR #7 (duplicate re-execution of mot_denoise_step from scheduler re-audit showing "no mot" despite #6) received ✅ Approved review.
- Addressed nits on main (post #6 merge):
  1. Added comment in `mot_denoise_step()` docstring area: "# AR/DM towers use separate MHA for now; true DualStreamJointAttention op ... planned for 2.2 per design."
  2. Enhanced `test_mot_denoise_step_has_dual_streams_and_timestep()` with `dm_denoised.shape == dm_noisy.shape` assert (and verified test still passes).
  3. MCP/skill defer noted (2.3 pending per OpenSpec).
  4. README already current (142 badge, mot listed in world models bullets + cosmos ex refs, test breakdown mentions mot) — no edit needed.
- Committed as 660c340 "address PR #7 review nits (post #6 merge)".
- Added review response comment to PR #7.
- Closed PR #7 as duplicate of merged #6 (per review recommendation).
- No code/tests breakage; 142p + VALID ex confirmed.
- Updated this log.

Next scheduled ~3h. All rules followed.

**2026-06-03 (address PR #8 review nits + merge)**:
- PR #8 (MCP timestep_dim for mot) received ✅ Approved review with 3 minor nits.
- Addressed on PR branch:
  - Nit 1: Enhanced docstring example for `timestep_dim` (added full call example with gru_hidden + timestep_dim).
  - Nit 2: Added `test_build_world_model_mot_with_timestep_dim()` in `test_mcp_server.py` (MCP tests now 6 passing).
  - Nit 3: Deferred per review (README/skill mentions already present from prior work).
- Committed nits: b074b49.
- Pushed, added review response comment, then `gh pr merge 8 --merge`.
- Merged cleanly into main (fast-forward on local).
- No further OpenSpec/tasks changes needed (2.3 already [x]; this was polish).
- Full re-audit: 142p + MCP tests green + mcp mot call with custom ts dim VALID.

Next scheduled ~3h. All rules followed.

**2026-06-03 (scheduled cycle - mcp timestep_dim for mot + full audit + KB + SDLC PR#8)**:
- Started per scheduler prompt: full read prompt + goal (last has workflow fix + PR#7 nits + previous 2.1 logs) + openspec/changes (list_dir + read tasks for temporal: 2.3/3.3/4.2/4.3/5.x remain; cosmos: 2.2/3.2/4.1/4.3+ pending, 2.1/2.3/3.1/3.3/4.2 done per file) + todo_write (merge to see pending).
- Open PRs check (critical, first in planning): gh pr list --state open = [] (none). "Open PRs checked via gh; no overlap with none". Safe to pick any.
- Full audits:
  - Tests: .venv/bin/python -m pytest -q --tb=no → 142 passed (1 harmless).
  - Examples verify: all *.n.orca.md + hf samples → all VALID (incl cosmos-mot, temporal, sae etc).
  - n-orca cli + builds ok.
  - Code inspect: list_dir n_orca/; read_file mcp_server.py (mot case present but no timestep_dim param in def or call, **common would leak input_dim causing error; skill has mot desc; README current; world_models has mot builder with timestep support; temporal has hidden shapes; no Timestep/Dual ops in spec.py yet); grep; test_mcp has no mot test yet.
  - OpenSpec: confirmed 2.2 ops pending (per design), 4.1 skill desc already in file so can mark, mcp wire partial.
- KB (per prompt): search_tool "q-orca-kb"; use_tool attempts (closed); direct mp_list/mp_search on n-orca-lang: 67 sources (self n-orca + 5 lang arXiv); searches for temporal/MoT/diffusion/language design/SAE hit self docs (goal, world_models, OpenSpec) + papers. Confirmed n-orca-lang-wiki config. Grounded before pick: self is source, limited external on timestep ops, no need to index now.
- Planning: Reviewed principles + open PRs rule (none open). From todos/OpenSpec + KB (self grounded), picked: enhance mcp build_world_model to accept/pass timestep_dim for "mot" variant (add param+doc+fix call site). Justif: smallest/safest (5-line backward-compat addition like gru_hidden for temporal; completes MCP exposure per design "Expose ... timestep_dim ... in ... MCP" and 2.3; fixes latent bug where **common would error; no new ops (2.2 heavier); high value for agents/MCP/skill usability; tests/mcp green; "prefer pure docs" but this is minimal code polish to make existing work; over 3.2 ex (would need new builder) or 2.2 (new ops in spec.py + possible verifier changes). "Open PRs checked via gh; no overlap with none". Conservative.
- Execution (read-first):
  - read_file (goal, openspec tasks x2, mcp_server.py build fn, world_models mot sig for ref, skill).
  - search_replace x2 on mcp_server.py: add timestep_dim=128 param after gru_hidden + docstring; update mot case to pass it and use mot_kw instead of **common (to avoid input_dim error).
  - Test: python direct build_world_model(variant="mot", ..., timestep_dim=32) → VALID arch.
  - Re-audits: mcp tests 5p, full pytest 142p; no ex change needed.
- SDLC (MCP public API change): branch grok/add-cosmos-mot-mcp-timestep-support, git add only mcp_server.py, commit (ref OpenSpec 2.3/design/goal), push, gh pr create #8. No wait.
- Progress: todo_write (audit done, cosmos-mcp-timestep complete, noted 2.2 next). Appended this to goal. OpenSpec tasks not edited (mcp wire was [x], this is polish).
- Used KB first.
- No blockers. 100% green. Followed all (reads, PR check first, smallest per rule, search_replace, re-audit, SDLC, n-orca only, log).
- Next expected: future cycle can do cosmos 2.2 (new ops per design), 3.2 reasoner ex, 4.1 mark skill in tasks (already desc), temporal 2.3/3.3, or KB index. PR#8 for review.

Next scheduled ~3h. All safety/SDLC/KB/prompt followed (tests+ex 100% green, open PRs checked none, read/search_replace, full log).

**2026-06-03 (fix scheduler SDLC workflow for open PRs)**:
- User feedback: scheduler should not select/work on tasks that touch or duplicate open PRs (the PR#7 duplicate re-impl of mot_denoise_step after #6 was the symptom).
- Updated "How to Work on This Goal" (new step 3: always gh pr list open PRs first; only pick non-overlapping tasks; skip if already in open PR).
- Added dedicated subsection "Avoiding Duplicate Work on Open PRs" under SDLC section with concrete rules and the reason (PR#7 example).
- Updated the embedded scheduler cycle instructions (in the recurring task prompt) to include the open-PRs check explicitly in the Planning phase.
- Updated Last updated and added this Recent Activity entry.
- Also updated AGENTS.md (see separate edit) to reflect the rule for agents.
- This ensures future cycles (including 019e8a9e5f8e) will check `gh pr list` before picking from todos/OpenSpec and will prefer unrelated safe work if the next logical slice is already in a PR.
- No code changes; pure docs/process fix. Tests/examples untouched.
