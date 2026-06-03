# n-orca — Agent Orientation

## What this repo is

n-orca is a Markdown DSL for declaring, verifying, visualizing, and executing neural network architectures as typed DAGs. It is a domain-specific dialect of the Orca state machine language family.

The core value: give LLMs (and humans) **one artifact** that covers authoring, reading, visualizing, explaining (via structured errors), and executing (compiles to runnable PyTorch `nn.Module`).

It ships with:
- Strong verifier (naming, DAG structure, shape inference, resource budgets, op coverage).
- Compilers to Mermaid and PyTorch.
- First-class support for **world models** (econ-sae style substrates) and **SAE architectures** (topk, l1, jumprelu, attn_topk, supervised_topk, gated).
- Hugging Face conversion for many model families.
- MCP server and dedicated agent skills for direct use by Grok, Claude, etc.

## Workflow: OpenSpec-driven + Agentic

All non-trivial work should land through OpenSpec changes in `openspec/changes/`.

- Use `openspec new change <name>` (or the `/openspec-propose` skill).
- Each change gets `proposal.md`, `design.md` (when useful), `tasks.md`, and `specs/<capability>/spec.md`.
- Validate with `openspec validate <name>`.
- Archive with `openspec archive <name>` when complete (or use the skill).

There is a persistent top-level improvement goal defined in `IMPROVEMENT_GOAL.md`. A recurring scheduler (every 3 hours, currently ID 019e8d4f094e) drives careful, test-first improvement cycles against the todos and this goal.

**Scheduler status note:** Only 1 active 3h loop at any time. Historical IDs elsewhere in this repo (e.g. older ones in goal logs) are records of deleted/replaced instances (refreshes for prompt updates like the open-PRs rule). Always query live state with the scheduler_list tool to confirm. No duplicates left running.

**Important workflow rule (added to prevent duplicate work):** Before picking any task in a cycle, run `gh pr list --state open` (or equivalent) and ensure the chosen OpenSpec/todo item does not modify files or scope already present in an open PR. Do not re-implement work that's already in an open PR (e.g. a builder or slice). Address review feedback on existing PRs when pasted by the human instead. See the "Avoiding Duplicate Work on Open PRs" subsection in IMPROVEMENT_GOAL.md for full details.

## n-orca Agent Skills & MCP

Dedicated skills (in `.claude/skills/`):
- `n-orca-verify`
- `n-orca-compile`
- `n-orca-build-sae` (supports all variants including advanced attn_topk / supervised_topk / gated)
- `n-orca-build-world-model`

The MCP server (`n-orca-mcp` or `python -m n_orca.mcp_server`) exposes tools including `verify_markdown`, `compile_mermaid`, `compile_pytorch`, `build_sae`, `build_world_model`, HF operations, etc.

When working as an agent, prefer using the skills and MCP tools over raw shell where possible. The skills know how to read files and call the right MCP functions.

## Key Dependency Contracts & Alignment

- Python >= 3.10.
- Optional extras: `[torch]`, `[hf]`, `[mcp]`.
- Strong alignment with **econ-sae** (world models as SAE substrates, advanced SAE variants for conjunctive/regime recovery).
- Intended to be the canonical source for architecture definitions used by econ-sae, sm-sae, polygram, sae-forge.
- q-orca-kb now has a dedicated `n-orca-lang` room (in the `q-orca-implementations` wing) for indexing n-orca docs, source, examples, and related papers (via deepwiki crawl + local files). Agents can use the q-orca-kb MCP (search_papers with room filter, crawl_site "n-orca-lang-wiki", etc.) for grounded knowledge.
- When adding builders or changing the surface, update examples, tests, the MCP tool, and the skills.

## File Layout (key parts)

```
n-orca/
  n_orca/
    ast.py, parser/, verifier/, compiler/ (mermaid + pytorch)
    sae.py, world_models.py          # primary builders for interpretability work
    ops/spec.py                      # op registry + shape rules
    mcp_server.py, cli/
    hf/ (adapters + convert)
  .claude/skills/                    # agent skills (n-orca-* + openspec-*)
  openspec/                          # spec-driven changes
  examples/ (many .n.orca.md + .mmd, including econ-* and sae-*)
  tests/ (esp. test_sae_and_world_models.py)
  docs/ (grammar.md, verification.md, proposed-sae-extensions.md)
  IMPROVEMENT_GOAL.md                # persistent long-term improvement goal
  AGENTS.md (this file)
```

## Local Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"
python -m pytest -q
n-orca verify examples/transformer-block.n.orca.md
```

Always re-verify examples and run the full test suite after changes.

## Conventions

- One Markdown file is the source of truth.
- Tables for layers and flow (LLMs are reliable with this format).
- Verify before execute/compile.
- Builders should be simple functions that return a verified `Architecture`.
- New ops go in `ops/spec.py` with shape inference and PyTorch emission.
- Advanced SAE/world model work should tie back to findings in econ-sae (and siblings).
- Use the scheduler-driven improvement loop (see IMPROVEMENT_GOAL.md) for ongoing careful progress.

## How Agents Should Work Here

1. Read `IMPROVEMENT_GOAL.md` and the current todo list at the start of any session.
2. Run audits (tests + example verification) before proposing changes.
3. Prefer tasks from the active OpenSpec change or the top-level improvement todos.
4. Make small, tested changes. Update the goal file's Recent Activity section.
5. When the recurring scheduler prompt fires (every 3h), follow its full cycle instructions exactly.

This setup allows Grok (and other agents) to improve n-orca carefully and indefinitely with minimal human intervention between cycles.