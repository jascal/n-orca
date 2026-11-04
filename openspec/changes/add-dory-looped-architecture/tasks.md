## 1. Core conversion support

- [x] 1.1 Register `GroupedQueryAttention` with explicit-head_dim params and
  topology-only PyTorch emission.
- [x] 1.2 Implement and register a `dory` HF adapter that unrolls the
  recurrent block, wires input re-injection, and declares block/loop groups.

## 2. Evidence and documentation

- [x] 2.1 Add conversion tests: block/loop groups, op counts, re-injection
  wiring, configurable loop count, inconsistent-split rejection, op params,
  PyTorch shape smoke test.
- [x] 2.2 Generate and verify `examples/hf-generated/dory1-4b.{n.orca.md,mmd}`
  from the published config.
- [x] 2.3 Document the adapter and its approximations in the README.

## 3. Validation

- [x] 3.1 Run OpenSpec validation and the full test suite.
