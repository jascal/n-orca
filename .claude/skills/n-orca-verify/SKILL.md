---
name: n-orca-verify
description: Verify a neural architecture defined in .n.orca.md for naming, structural (DAG), shape, resource, and op correctness. Use when the user wants to check a neural net spec before compiling or implementing.
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