---
name: n-orca-compile
description: Compile a .n.orca.md architecture to Mermaid (visual) or PyTorch (runnable nn.Module). Use when the user wants to visualize or get executable code from a verified neural spec.
argument-hint: [mermaid|pytorch] [file]
allowed-tools: Read, mcp__n_orca__compile_mermaid, mcp__n_orca__compile_pytorch
---

Compile an N-Orca architecture.

Usage: /n-orca-compile mermaid path/to/arch.n.orca.md
       /n-orca-compile pytorch path/to/arch.n.orca.md [--out model.py]

- Read the source if a file path is given.
- Call the appropriate MCP compile tool (compile_mermaid or compile_pytorch).
- For pytorch, the output is runnable Python code that can be exec'd or written to file.
- Always verify first if the source looks unverified (suggest running n-orca-verify).

Output the result. For PyTorch, include a small usage example if possible (instantiate + forward with dummy input).

Supports the full set of shipped ops and the advanced SAE/world-model builders.