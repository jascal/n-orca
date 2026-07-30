# add-topology-properties

## Why

n-orca's verifier answers *"is this DAG well-formed and shape-consistent?"*. It
cannot answer *"is subgraph A structurally incapable of influencing subgraph
B?"*

That second question is the one that matters for architectures where an
information boundary **is the design claim** rather than an implementation
detail: gated/modulated architectures, frozen value cores, privileged
sub-networks, adapters that must not write back into a base model — any case
where the topology itself is the guarantee.

Today those claims live in `## verification rules` as prose. The parser stores
them in `Architecture.verification_rules` and **nothing ever checks them**. A
rule that reads

> NO path exists from any cortical layer to any core layer

is decidable by BFS in microseconds and is currently a comment. Worse, it is a
comment that *looks* checked, because it sits in a section named "verification
rules" inside a file the verifier reports `VALID` for.

`orca-lang` already solved the analogous problem for state machines
(`packages/orca-lang/docs/property-checking-design.md`): six property types,
BFS-based checking, counterexample traces, integrated into the verify pipeline.
This change ports that capability to layer DAGs, reusing its vocabulary and
error-code naming so the two languages stay one family.

## What Changes

- **New `## groups` section** (table, like `## tensors`) naming sets of layers,
  so properties can be stated at subgraph granularity. New AST type `Group`;
  new field `Architecture.groups`.
- **New `## properties` section** (bullet list, like `orca-lang`'s
  `properties` block) with six property kinds. New AST type `Property`; new
  field `Architecture.properties`.
  - Ported from `orca-lang`: `reachable`, `unreachable`, `passes_through`,
    `within` (the DAG-native rename of `responds`).
  - **DAG-native, no `orca-lang` equivalent:** `param_fraction` (a group's
    share of total parameters) and `out_dim` (a layer's inferred last
    dimension).
  - Deliberately **not** ported: `live` (Stage 2's
    every-layer-reaches-an-output check already is the DAG version) and
    `invariant` (`## invariants` already covers resource bounds — a second
    mechanism would be a fork).
- **New verifier Stage 6 — Property**, in a new module
  `n_orca/verifier/properties.py` exposing `check_properties(arch, report)`.
  Runs last, only when Naming and Structural passed, because it needs a valid
  acyclic graph with resolvable names, and because `param_fraction` / `out_dim`
  read Stage 3/4 outputs off the report.
- **Counterexample paths are part of the contract.** Exclusion and
  pass-through failures name the concrete violating path as an arrow chain
  (`cortex_ln -> critic_fc -> feedback_proj -> core_add`). An error that says
  "isolation violated" without the path is not actionable for the LLM refiner
  that is the primary consumer.
- **Ten new error codes**, mirroring `orca-lang`'s naming:
  `PROPERTY_REACHABILITY_FAIL`, `PROPERTY_EXCLUSION_FAIL`,
  `PROPERTY_PATH_FAIL`, `PROPERTY_DEPTH_FAIL`,
  `PROPERTY_PARAM_FRACTION_FAIL`, `PROPERTY_WIDTH_FAIL`,
  `PROPERTY_UNKNOWN_REFERENCE`, `PROPERTY_MALFORMED` (all `E`),
  `GROUP_NAME_COLLISION` (`E`), `GROUP_OVERLAP` (`W`).
- **Surface propagation** (per the `AGENTS.md` rule): parser, `render`,
  Mermaid (`## groups` becomes `subgraph` blocks — the isolation boundary
  becomes visible), CLI (`Properties: N checked, M failed` plus `--json` plus
  `--strict`), MCP `verify_markdown`, the `n-orca-verify` skill,
  `docs/grammar.md`, `docs/verification.md`.
- **Two examples**: `examples/subcortical-value-cortex.n.orca.md` (positive —
  its old prose `## verification rules` become executable properties) and
  `examples/subcortical-value-cortex-leak.n.orca.md` (negative — a critic head
  that feeds back into the value core, which must fail
  `PROPERTY_EXCLUSION_FAIL` with the bypass path in the message).

## Capabilities

Affects the **verification** capability. Touches the parser/AST, render, and
Mermaid surfaces as a consequence.

## Scope

**In:** the two sections, the AST types, Stage 6, the ten codes,
counterexample formatting, the surface wiring listed above, one positive and
one negative example, `tests/test_properties.py`.

**Out:**

- Guard-style pruning (`orca-lang` needs it because transitions carry guards;
  DAG edges do not — every flow edge is always live, so the DAG check is
  *exact*, not an overapproximation).
- A `max_layers` size limit à la `MACHINE_TOO_LARGE`. Layer DAGs are already
  bounded by the file, the checks are O(V+E) per property, and the shipped
  `llama-7b` example has hundreds of layers — a 64-node cap would be actively
  wrong here. Revisit only if a real file gets slow.
- Any new op, any change to shape inference, any change to existing error
  codes or existing verify output.

## Backwards compatibility

`## groups` and `## properties` are optional sections. Unknown `##` sections
are currently ignored silently by the parser, so **no existing file changes
meaning**. Every file in `examples/` must produce byte-identical `verify` and
`compile mermaid` output after this change; that is asserted by a regression
test, not assumed.

## Related

- `orca-lang`: `packages/orca-lang/docs/property-checking-design.md`,
  `packages/orca-lang/examples/payment-with-properties.orca.md`.
- Bonus fix bundled here: `docs/grammar.md` lists the pipeline as
  "1 — Structural / 2 — Naming", while `verifier.py` and
  `docs/verification.md` both have Naming first. The code is right; the table
  in `grammar.md` is wrong, and it has to be rewritten anyway to add Stage 6.

## Acceptance Criteria

1. `openspec validate add-topology-properties` passes.
2. `python -m pytest -q` is clean; the 143 pre-existing tests still pass.
3. Every file in `examples/` and `examples/hf-generated/` verifies, and
   produces output identical to the pre-change baseline — except the two new
   files.
4. `examples/subcortical-value-cortex.n.orca.md` verifies `VALID` with all six
   property kinds exercised and `Properties: 6 checked, 0 failed`.
5. `examples/subcortical-value-cortex-leak.n.orca.md` verifies `INVALID` with
   `PROPERTY_EXCLUSION_FAIL` whose message contains the literal path
   `cortex_ln -> critic_fc -> feedback_proj -> core_add`.
6. Group-to-group quantification is asymmetric in the right direction:
   `unreachable: core from cortex` passes on the positive example while
   `unreachable: cortex from core` fails, because the neuromodulator bus edge
   exists.
