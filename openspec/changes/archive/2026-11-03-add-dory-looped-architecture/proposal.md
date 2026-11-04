## Why

NVIDIA's Nemotron-Labs Dory1.0-4B (`model_type: "dory"`) is a looped
(latent-recurrent) normalized-GPT decoder: a shared Recurrent block is applied
`n_recurrent_loops` times between an Input and an Output block, and the Input
block's output is re-injected before every loop. n-orca cannot currently
convert it, and the loop and its re-injection edge do not fit any existing
adapter's straight-line decoder skeleton.

## What Changes

- Add a `GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, window)`
  op whose parameter count uses an explicit `head_dim` (Dory has
  `16 x 256 = 4096 != hidden_size 2560`). PyTorch emission is topology-only.
- Add an HF adapter for `model_type: "dory"` that splits `layer_types` into
  Input / Recurrent / Output by `n_input_layers` / `n_recurrent_layers` /
  `n_output_layers`, **unrolls** the Recurrent block `n_recurrent_loops` times
  (n-orca graphs are DAGs), wires an `Add` re-injection node at each loop
  entry, and declares one `## groups` entry per block/loop so Mermaid draws
  them as subgraphs.
- Ship `examples/hf-generated/dory1-4b.{n.orca.md,mmd}` from the published
  config, with tests and README rows.

## Capabilities

### New Capabilities

- `dory-looped-architecture`: Convert and verify the published Dory looped
  decoder as an inspectable, unrolled n-orca architecture.

### Modified Capabilities

- None.

## Impact

Additive: one op, one adapter, one example pair, tests, README. The frozen
example baseline excludes the new example by name, as it does for other
post-baseline examples that declare groups.

Known approximations, stated in the generated description: unrolled loops
share weights in the real model but n-orca counts each loop's parameters;
SwiGLU is rendered as FeedForward; the nGPT update
`h <- Norm(h + alpha*(Norm(f(h)) - h))` is rendered as `Add` + `LayerNorm`;
the exact scaling/normalization of the re-injection sum is not stated in the
public config and is drawn as a plain `Add`.
