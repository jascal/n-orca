## 1. Core conversion support

- [x] 1.1 Register semantic LFM3 gated-convolution and head-gated GQA ops with
  shape, parameter, and executable approximation behavior.
- [x] 1.2 Implement and register an `lfm3` Hugging Face adapter that emits the
  config-defined hybrid decoder schedule and sandwich-norm residual blocks.

## 2. Evidence and documentation

- [x] 2.1 Add conversion/verification regression tests using the published
  LFM3-1.7B configuration, including exact layer-type counts and order.
- [x] 2.2 Generate and verify the LFM3-1.7B Markdown and Mermaid examples
  from the published config (the command is documented rather than committing
  a 197-node generated artifact).
- [x] 2.3 Document the supported family and compilation approximations in the
  README.

## 3. Validation

- [x] 3.1 Run OpenSpec validation, the full test suite, and the shipped-example
  verification sweep.
