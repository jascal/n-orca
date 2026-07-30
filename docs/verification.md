# Verification — Error Catalog

Every code below is a stable identifier. Tools (LLM refiners, CI checks,
editor integrations) can pattern-match on the code.

Severity column: `E` = error (architecture is invalid); `W` = warning
(architecture is still considered valid unless `--strict` is set).

| Code | Stage | Severity | Meaning | Typical fix |
|------|-------|----------|---------|-------------|
| `NO_LAYERS` | 1 | E | architecture declares no `## layer` headings | add at least one layer |
| `NO_INPUT_LAYER` | 1 | E | no layer marked `[input]` | mark the entry layer `## layer <name> [input]` |
| `NO_OUTPUT_LAYER` | 1 | E | no layer marked `[output]` | mark the exit layer `## layer <name> [output]` |
| `DUPLICATE_LAYER` | 1 | E | a layer name appears twice | rename or merge the duplicate |
| `UNKNOWN_LAYER_REFERENCE` | 1 | E | a `## flow` source or target isn't declared | declare `## layer <name>` or fix the typo |
| `INPUT_NOT_IN_TENSORS` | 1 | W | input layer has no row in `## tensors` | add a row declaring shape and dtype |
| `OUTPUT_NOT_IN_TENSORS` | 1 | W | output layer has no row in `## tensors` | add a row declaring shape and dtype |
| `CYCLE_DETECTED` | 2 | E | the layer graph contains a cycle | redirect an edge to break the cycle |
| `UNREACHABLE_LAYER` | 2 | E | a layer has no path from any `[input]` | connect it via `## flow`, or remove it |
| `LAYER_NOT_REACHING_OUTPUT` | 2 | E | a layer has no path to any `[output]` | add an outgoing edge, or remove the layer |
| `INPUT_HAS_PREDECESSOR` | 2 | W | input layer has incoming edges | remove those edges or unmark `[input]` |
| `OUTPUT_HAS_SUCCESSOR` | 2 | W | output layer has outgoing edges | remove those edges or unmark `[output]` |
| `INPUT_SHAPE_UNDECLARED` | 3 | W | input layer's shape can't be inferred | add a row in `## tensors` |
| `MISSING_OP` | 3 | E | non-IO layer has no `op:` | add `- op: <Op>(args)` |
| `SHAPE_MISMATCH` | 3 | E | op's shape rule rejects its inputs | fix op args or upstream shape chain |
| `INPUT_ARITY_MISMATCH` | 3 | E | op expects N inputs, got M | add/remove `## flow` edges |
| `OUTPUT_ARITY_MISMATCH` | 3 | E | output layer has != 1 incoming edge | merge with `Add`/`Concat` first |
| `OUTPUT_SHAPE_MISMATCH` | 3 | E | inferred output shape differs from `## tensors` | fix the chain or the tensor row |
| `DECLARED_SHAPE_MISMATCH` | 3 | E | layer's explicit `shape:` differs from inferred | remove the bullet or fix the chain |
| `PARAM_BUDGET_EXCEEDED` | 4 | E | total params exceeds `param_count <=` | shrink widths/depths, or relax invariant |
| `DEPTH_BUDGET_EXCEEDED` | 4 | E | longest path exceeds `depth <=` | shorten chain, or relax invariant |
| `OUTPUT_SHAPE_INVARIANT` | 4 | E | inferred output shape differs from `## invariants` | adjust architecture or invariant |
| `FLOPS_NOT_IMPLEMENTED` | 4 | W | `flops` invariants are accepted but not enforced | remove the invariant, or use an external profiler |
| `UNKNOWN_OP` | 5 | W | layer uses an op not in the standard library | pick a standard op, or accept the placeholder and supply a custom module in the host |
| `PROPERTY_REACHABILITY_FAIL` | 6 | E | `reachable` — no path exists | add a `## flow` path, or drop the property |
| `PROPERTY_EXCLUSION_FAIL` | 6 | E | `unreachable` — a path DOES exist | remove an edge on the reported path, or drop the property |
| `PROPERTY_PATH_FAIL` | 6 | E | `passes_through` — a bypassing path exists | route it through the cut, remove the bypass, or widen the cut |
| `PROPERTY_DEPTH_FAIL` | 6 | E | `within` — shortest path exceeds N (or no path) | shorten the chain, or raise the bound |
| `PROPERTY_PARAM_FRACTION_FAIL` | 6 | E | a group exceeds its parameter budget | shrink the group's layers, or relax the budget |
| `PROPERTY_WIDTH_FAIL` | 6 | E | `out_dim` — layer's last dim exceeds the bound | reduce the layer's width, or raise the bound |
| `PROPERTY_UNKNOWN_REFERENCE` | 6 | E | a property or group names a nonexistent layer or group | declare it, or fix the typo |
| `PROPERTY_MALFORMED` | 6 | E | a `## properties` bullet doesn't parse | rewrite it in the kind's grammar |
| `GROUP_NAME_COLLISION` | 6 | E | a group name shadows a layer name | rename the group |
| `GROUP_OVERLAP` | 6 | W | a layer belongs to two groups | make the groups disjoint |

## Stage 6 — property checking

Stage 6 decides the claims in `## properties` against the layer DAG, with
`## groups` naming subgraphs. See [`grammar.md`](./grammar.md) for the syntax
and the group-quantification rules.

**Ordering.** Stage 6 runs last. It needs Naming and Structural to have passed
(operand resolution and BFS are meaningless on a graph with an unresolvable
name or a cycle), and it reads `inferred_shapes` and `param_count`, which
Stages 3–4 put on the report. **When an earlier stage failed, Stage 6 is
skipped silently** — no property or group diagnostics at all. A cycle would
otherwise make every `unreachable` claim "fail" and bury the real error.

**Counterexamples.** `PROPERTY_EXCLUSION_FAIL` and `PROPERTY_PATH_FAIL` always
name the concrete violating path as an arrow chain:

```
[ERR]  PROPERTY_EXCLUSION_FAIL: property `unreachable: core from cortex` fails:
       a path exists — cortex_ln -> critic_fc -> feedback_proj -> core_add
```

The path is a **shortest** witness and is a deterministic function of the
document (adjacency is built in `## flow` row order, the search is BFS), so it
is stable enough to diff in CI. It is also *exact*: a `## flow` edge carries no
guard, so unlike `orca-lang`'s state-machine checks these results are not
overapproximations — the reported path is a real activation and gradient path.

The path is available structurally too, as `path` on each entry of the
`properties.results` array in `--json` output.

## How the LLM should consume errors

Each error carries:

```json
{
  "code": "SHAPE_MISMATCH",
  "message": "layer 'add': Add inputs have mismatched shapes: [...]",
  "suggestion": "check the layer's op args and incoming shapes",
  "severity": "error"
}
```

The `code` is the stable handle; the `message` describes the specific
instance; the `suggestion` is written in the imperative voice so an LLM
can act on it directly when refining the architecture.

`n-orca verify --json` produces a JSON array of these reports, one per
architecture in the file. Each report also carries the Stage 6 outcome:

```json
{
  "groups": {"core": ["core_sense", "core_fc"], "cortex": ["embed"]},
  "properties": {
    "checked": 6,
    "passed": 5,
    "failed": 1,
    "results": [
      {
        "kind": "unreachable",
        "raw": "unreachable: core from cortex",
        "passed": false,
        "code": "PROPERTY_EXCLUSION_FAIL",
        "path": ["cortex_ln", "critic_fc", "feedback_proj", "core_add"],
        "skipped": false
      }
    ]
  }
}
```

`skipped` marks a claim that could not be evaluated for a benign reason — an
unresolvable symbolic bound, or a layer whose shape Stage 3 gave up on. Skipped
claims count toward neither `passed` nor `failed`, so `checked` never overstates
what was actually proved.
