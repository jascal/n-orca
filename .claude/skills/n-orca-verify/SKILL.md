---
name: n-orca-verify
description: Verify a neural architecture defined in .n.orca.md for naming, structural (DAG), shape, resource, op, and topology-property correctness. Use when the user wants to check a neural net spec before compiling or implementing.
argument-hint: [file]
allowed-tools: Read, mcp__n_orca__verify_markdown
---

Verify the N-Orca architecture definition.

If $ARGUMENTS is a file path, read the file first, then call `verify_markdown` with its contents (or use --file if supported).

If $ARGUMENTS is raw `.n.orca.md` source, pass it directly.

After verification:

- If valid: confirm and note any warnings.
- If invalid: list errors with codes, messages, and suggestions. Group by severity. Offer to refine or fix common issues.

Show the full structured output from the tool. Do not hide details.

This is the neural equivalent of /orca-verify — one Markdown spec, verified before any PyTorch code is emitted.

## Stage 6 — topology properties

The payload carries a `properties` block (`checked` / `passed` / `failed` /
`results`) and a `groups` map whenever the document declares `## properties`
or `## groups`. Report the `Properties: N checked, M failed` summary alongside
the usual result, and **always surface the counterexample path** — it is the
actionable part.

Stage 6 is skipped silently when Naming or Structural failed. If a document
declares properties but the payload reports `checked: 0`, say so: the claims
were **not** checked, and fixing the earlier error is what unblocks them. Never
present that as "properties passed".

A `result` with `skipped: true` was not evaluated (unresolvable symbolic bound,
or a layer whose shape Stage 3 could not infer). Do not report it as proved.

### What to do about each code

| Code | Meaning | Fix |
|------|---------|-----|
| `PROPERTY_EXCLUSION_FAIL` | an `unreachable` boundary is violated; the message names the exact path | Remove one `## flow` edge on that path. Pick the edge that is the leak, not the first one — usually the edge entering the protected subgraph. If the influence is intentional, the property is what's wrong: delete it and say so explicitly rather than silently. |
| `PROPERTY_PATH_FAIL` | a `passes_through` cut is bypassed | Either route the reported path through the named cut, delete the bypassing edge, or add the bypass layers to the cut group if they are legitimately part of the boundary. |
| `PROPERTY_REACHABILITY_FAIL` | a `reachable` claim has no path | Add the missing `## flow` edges. If the two really are independent, drop the property. |
| `PROPERTY_DEPTH_FAIL` | a `within` bound is exceeded, or no path exists at all | Read the message: "shortest path is N edges" means shorten the chain or raise the bound; "no path" means raising the bound will never help — connect them or drop the property. |
| `PROPERTY_PARAM_FRACTION_FAIL` | a group exceeds its share of parameters | The message gives the actual percentage and the raw counts. Shrink the group's layer widths; do **not** pad the rest of the model to move the ratio. |
| `PROPERTY_WIDTH_FAIL` | a layer's inferred last dim exceeds the bound | Change that layer's op args, or raise the bound. |
| `PROPERTY_UNKNOWN_REFERENCE` | a property or a `## groups` row names something undeclared | Almost always a typo against `## layer` headings — check spelling before adding a layer. |
| `PROPERTY_MALFORMED` | a bullet doesn't parse | The suggestion carries the exact grammar for that kind. Rewrite the bullet; don't delete the claim. |
| `GROUP_NAME_COLLISION` | a group is named like a layer | Rename the group. |
| `GROUP_OVERLAP` (W) | a layer is in two groups | Check whether the asserted boundary is the intended one — overlap usually means it isn't. Legal, so only fix it if the claim is actually wrong. |

When you fix a property failure by editing the architecture, re-verify. When you
fix it by deleting or weakening the property, say plainly that you removed a
check rather than satisfied it.