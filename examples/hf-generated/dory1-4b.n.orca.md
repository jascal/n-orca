# architecture Dory1_4B

> NVIDIA Nemotron-Labs Dory: a looped (latent-recurrent) normalized-GPT decoder. Input block (16 sublayers) -> shared Recurrent block (40 sublayers) applied 2x with the input-block output h_in re-injected (added) before every loop -> Output block (16 sublayers) -> LM head. Sublayers alternate GQA attention (1024-token sliding window, RoPE theta 10000; or global, RoPE theta 10000000) and SwiGLU MLPs (rendered as FeedForward). Each sublayer uses the nGPT hypersphere update h <- Norm(h + alpha*(Norm(f(h)) - h)), drawn as f -> Add (alpha-lerp) -> renorm (rendered as LayerNorm). The recurrent loops are UNROLLED and share weights in the real model; n-orca counts each loop's parameters separately. The token embedding and LM head are untied. Compiler output is topology-only for GQA, RoPE, sliding-window masks and nGPT scaling.

## hyperparameters

| Name               | Type | Default |
|--------------------|------|---------|
| vocab_size         | int  | 131072  |
| d_model            | int  | 2560    |
| n_heads            | int  | 16      |
| n_kv_heads         | int  | 4       |
| head_dim           | int  | 256     |
| d_ff               | int  | 7168    |
| sliding_window     | int  | 1024    |
| max_pos            | int  | 262144  |
| n_input_layers     | int  | 16      |
| n_recurrent_layers | int  | 40      |
| n_output_layers    | int  | 16      |
| n_recurrent_loops  | int  | 2       |

## tensors

| Name      | Shape              | Dtype   |
|-----------|--------------------|---------|
| input_ids | (B, S)             | int64   |
| logits    | (B, S, vocab_size) | float32 |

## layer input_ids [input]
> Token IDs

## layer tok_embed
> Token embedding (nGPT: rows normalized)
- op: Embedding(vocab_size, d_model)

## layer in_swa_0
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer in_swa_0_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_swa_0_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_mlp_1
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer in_mlp_1_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_mlp_1_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_swa_2
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer in_swa_2_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_swa_2_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_mlp_3
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer in_mlp_3_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_mlp_3_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_swa_4
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer in_swa_4_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_swa_4_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_mlp_5
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer in_mlp_5_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_mlp_5_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_attn_6
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer in_attn_6_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_attn_6_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_mlp_7
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer in_mlp_7_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_mlp_7_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_swa_8
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer in_swa_8_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_swa_8_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_mlp_9
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer in_mlp_9_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_mlp_9_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_swa_10
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer in_swa_10_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_swa_10_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_mlp_11
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer in_mlp_11_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_mlp_11_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_swa_12
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer in_swa_12_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_swa_12_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_mlp_13
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer in_mlp_13_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_mlp_13_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_attn_14
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer in_attn_14_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_attn_14_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer in_mlp_15
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer in_mlp_15_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer in_mlp_15_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_inject
> Loop 1: s~ = s_0 + h_in (input re-injection)
- op: Add()

## layer loop1_swa_16
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_16_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_16_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_17
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_17_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_17_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_18
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_18_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_18_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_19
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_19_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_19_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_20
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_20_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_20_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_21
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_21_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_21_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_attn_22
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer loop1_attn_22_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_attn_22_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_23
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_23_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_23_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_24
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_24_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_24_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_25
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_25_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_25_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_26
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_26_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_26_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_27
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_27_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_27_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_28
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_28_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_28_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_29
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_29_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_29_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_attn_30
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer loop1_attn_30_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_attn_30_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_31
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_31_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_31_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_32
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_32_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_32_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_33
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_33_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_33_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_34
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_34_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_34_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_35
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_35_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_35_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_36
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_36_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_36_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_37
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_37_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_37_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_attn_38
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer loop1_attn_38_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_attn_38_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_39
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_39_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_39_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_40
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_40_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_40_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_41
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_41_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_41_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_42
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_42_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_42_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_43
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_43_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_43_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_44
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_44_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_44_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_45
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_45_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_45_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_attn_46
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer loop1_attn_46_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_attn_46_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_47
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_47_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_47_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_48
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_48_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_48_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_49
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_49_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_49_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_50
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_50_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_50_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_51
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_51_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_51_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_swa_52
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop1_swa_52_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_swa_52_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_53
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_53_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_53_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_attn_54
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer loop1_attn_54_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_attn_54_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop1_mlp_55
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop1_mlp_55_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop1_mlp_55_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_inject
> Loop 2: s~ = s_1 + h_in (input re-injection)
- op: Add()

## layer loop2_swa_16
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_16_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_16_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_17
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_17_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_17_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_18
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_18_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_18_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_19
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_19_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_19_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_20
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_20_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_20_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_21
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_21_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_21_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_attn_22
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer loop2_attn_22_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_attn_22_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_23
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_23_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_23_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_24
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_24_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_24_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_25
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_25_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_25_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_26
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_26_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_26_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_27
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_27_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_27_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_28
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_28_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_28_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_29
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_29_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_29_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_attn_30
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer loop2_attn_30_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_attn_30_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_31
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_31_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_31_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_32
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_32_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_32_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_33
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_33_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_33_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_34
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_34_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_34_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_35
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_35_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_35_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_36
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_36_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_36_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_37
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_37_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_37_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_attn_38
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer loop2_attn_38_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_attn_38_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_39
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_39_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_39_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_40
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_40_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_40_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_41
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_41_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_41_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_42
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_42_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_42_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_43
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_43_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_43_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_44
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_44_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_44_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_45
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_45_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_45_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_attn_46
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer loop2_attn_46_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_attn_46_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_47
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_47_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_47_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_48
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_48_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_48_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_49
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_49_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_49_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_50
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_50_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_50_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_51
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_51_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_51_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_swa_52
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer loop2_swa_52_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_swa_52_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_53
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_53_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_53_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_attn_54
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer loop2_attn_54_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_attn_54_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer loop2_mlp_55
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer loop2_mlp_55_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer loop2_mlp_55_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_swa_56
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer out_swa_56_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_swa_56_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_mlp_57
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer out_mlp_57_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_mlp_57_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_swa_58
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer out_swa_58_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_swa_58_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_mlp_59
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer out_mlp_59_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_mlp_59_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_swa_60
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer out_swa_60_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_swa_60_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_mlp_61
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer out_mlp_61_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_mlp_61_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_attn_62
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer out_attn_62_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_attn_62_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_mlp_63
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer out_mlp_63_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_mlp_63_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_swa_64
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer out_swa_64_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_swa_64_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_mlp_65
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer out_mlp_65_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_mlp_65_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_swa_66
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer out_swa_66_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_swa_66_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_mlp_67
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer out_mlp_67_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_mlp_67_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_swa_68
> Causal GQA, sliding window, RoPE (local theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, sliding_window)

## layer out_swa_68_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_swa_68_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_mlp_69
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer out_mlp_69_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_mlp_69_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_attn_70
> Causal GQA, global context, RoPE (long theta); nGPT-normalized q/k
- op: GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, 0)

## layer out_attn_70_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_attn_70_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer out_mlp_71
> SwiGLU MLP, bias-free (rendered as FeedForward)
- op: FeedForward(d_model, d_ff, 0.0)

## layer out_mlp_71_lerp
> nGPT update h + alpha*(Norm(f(h)) - h) (rendered as Add)
- op: Add()

## layer out_mlp_71_norm
> Renormalize to the hypersphere (rendered as LayerNorm)
- op: LayerNorm(d_model)

## layer lm_head
> LM projection (nGPT logit scaling not drawn)
- op: Linear(d_model, vocab_size)

## layer logits [output]

## flow

| Source             | Target             | Tensor             |
|--------------------|--------------------|--------------------|
| input_ids          | tok_embed          | tokens             |
| tok_embed          | in_swa_0           | in_swa_0_in        |
| in_swa_0           | in_swa_0_lerp      | in_swa_0_out       |
| tok_embed          | in_swa_0_lerp      | in_swa_0_skip      |
| in_swa_0_lerp      | in_swa_0_norm      | in_swa_0_mix       |
| in_swa_0_norm      | in_mlp_1           | in_mlp_1_in        |
| in_mlp_1           | in_mlp_1_lerp      | in_mlp_1_out       |
| in_swa_0_norm      | in_mlp_1_lerp      | in_mlp_1_skip      |
| in_mlp_1_lerp      | in_mlp_1_norm      | in_mlp_1_mix       |
| in_mlp_1_norm      | in_swa_2           | in_swa_2_in        |
| in_swa_2           | in_swa_2_lerp      | in_swa_2_out       |
| in_mlp_1_norm      | in_swa_2_lerp      | in_swa_2_skip      |
| in_swa_2_lerp      | in_swa_2_norm      | in_swa_2_mix       |
| in_swa_2_norm      | in_mlp_3           | in_mlp_3_in        |
| in_mlp_3           | in_mlp_3_lerp      | in_mlp_3_out       |
| in_swa_2_norm      | in_mlp_3_lerp      | in_mlp_3_skip      |
| in_mlp_3_lerp      | in_mlp_3_norm      | in_mlp_3_mix       |
| in_mlp_3_norm      | in_swa_4           | in_swa_4_in        |
| in_swa_4           | in_swa_4_lerp      | in_swa_4_out       |
| in_mlp_3_norm      | in_swa_4_lerp      | in_swa_4_skip      |
| in_swa_4_lerp      | in_swa_4_norm      | in_swa_4_mix       |
| in_swa_4_norm      | in_mlp_5           | in_mlp_5_in        |
| in_mlp_5           | in_mlp_5_lerp      | in_mlp_5_out       |
| in_swa_4_norm      | in_mlp_5_lerp      | in_mlp_5_skip      |
| in_mlp_5_lerp      | in_mlp_5_norm      | in_mlp_5_mix       |
| in_mlp_5_norm      | in_attn_6          | in_attn_6_in       |
| in_attn_6          | in_attn_6_lerp     | in_attn_6_out      |
| in_mlp_5_norm      | in_attn_6_lerp     | in_attn_6_skip     |
| in_attn_6_lerp     | in_attn_6_norm     | in_attn_6_mix      |
| in_attn_6_norm     | in_mlp_7           | in_mlp_7_in        |
| in_mlp_7           | in_mlp_7_lerp      | in_mlp_7_out       |
| in_attn_6_norm     | in_mlp_7_lerp      | in_mlp_7_skip      |
| in_mlp_7_lerp      | in_mlp_7_norm      | in_mlp_7_mix       |
| in_mlp_7_norm      | in_swa_8           | in_swa_8_in        |
| in_swa_8           | in_swa_8_lerp      | in_swa_8_out       |
| in_mlp_7_norm      | in_swa_8_lerp      | in_swa_8_skip      |
| in_swa_8_lerp      | in_swa_8_norm      | in_swa_8_mix       |
| in_swa_8_norm      | in_mlp_9           | in_mlp_9_in        |
| in_mlp_9           | in_mlp_9_lerp      | in_mlp_9_out       |
| in_swa_8_norm      | in_mlp_9_lerp      | in_mlp_9_skip      |
| in_mlp_9_lerp      | in_mlp_9_norm      | in_mlp_9_mix       |
| in_mlp_9_norm      | in_swa_10          | in_swa_10_in       |
| in_swa_10          | in_swa_10_lerp     | in_swa_10_out      |
| in_mlp_9_norm      | in_swa_10_lerp     | in_swa_10_skip     |
| in_swa_10_lerp     | in_swa_10_norm     | in_swa_10_mix      |
| in_swa_10_norm     | in_mlp_11          | in_mlp_11_in       |
| in_mlp_11          | in_mlp_11_lerp     | in_mlp_11_out      |
| in_swa_10_norm     | in_mlp_11_lerp     | in_mlp_11_skip     |
| in_mlp_11_lerp     | in_mlp_11_norm     | in_mlp_11_mix      |
| in_mlp_11_norm     | in_swa_12          | in_swa_12_in       |
| in_swa_12          | in_swa_12_lerp     | in_swa_12_out      |
| in_mlp_11_norm     | in_swa_12_lerp     | in_swa_12_skip     |
| in_swa_12_lerp     | in_swa_12_norm     | in_swa_12_mix      |
| in_swa_12_norm     | in_mlp_13          | in_mlp_13_in       |
| in_mlp_13          | in_mlp_13_lerp     | in_mlp_13_out      |
| in_swa_12_norm     | in_mlp_13_lerp     | in_mlp_13_skip     |
| in_mlp_13_lerp     | in_mlp_13_norm     | in_mlp_13_mix      |
| in_mlp_13_norm     | in_attn_14         | in_attn_14_in      |
| in_attn_14         | in_attn_14_lerp    | in_attn_14_out     |
| in_mlp_13_norm     | in_attn_14_lerp    | in_attn_14_skip    |
| in_attn_14_lerp    | in_attn_14_norm    | in_attn_14_mix     |
| in_attn_14_norm    | in_mlp_15          | in_mlp_15_in       |
| in_mlp_15          | in_mlp_15_lerp     | in_mlp_15_out      |
| in_attn_14_norm    | in_mlp_15_lerp     | in_mlp_15_skip     |
| in_mlp_15_lerp     | in_mlp_15_norm     | in_mlp_15_mix      |
| in_mlp_15_norm     | loop1_inject       | s_0                |
| in_mlp_15_norm     | loop1_inject       | h_in_1             |
| loop1_inject       | loop1_swa_16       | loop1_swa_16_in    |
| loop1_swa_16       | loop1_swa_16_lerp  | loop1_swa_16_out   |
| loop1_inject       | loop1_swa_16_lerp  | loop1_swa_16_skip  |
| loop1_swa_16_lerp  | loop1_swa_16_norm  | loop1_swa_16_mix   |
| loop1_swa_16_norm  | loop1_mlp_17       | loop1_mlp_17_in    |
| loop1_mlp_17       | loop1_mlp_17_lerp  | loop1_mlp_17_out   |
| loop1_swa_16_norm  | loop1_mlp_17_lerp  | loop1_mlp_17_skip  |
| loop1_mlp_17_lerp  | loop1_mlp_17_norm  | loop1_mlp_17_mix   |
| loop1_mlp_17_norm  | loop1_swa_18       | loop1_swa_18_in    |
| loop1_swa_18       | loop1_swa_18_lerp  | loop1_swa_18_out   |
| loop1_mlp_17_norm  | loop1_swa_18_lerp  | loop1_swa_18_skip  |
| loop1_swa_18_lerp  | loop1_swa_18_norm  | loop1_swa_18_mix   |
| loop1_swa_18_norm  | loop1_mlp_19       | loop1_mlp_19_in    |
| loop1_mlp_19       | loop1_mlp_19_lerp  | loop1_mlp_19_out   |
| loop1_swa_18_norm  | loop1_mlp_19_lerp  | loop1_mlp_19_skip  |
| loop1_mlp_19_lerp  | loop1_mlp_19_norm  | loop1_mlp_19_mix   |
| loop1_mlp_19_norm  | loop1_swa_20       | loop1_swa_20_in    |
| loop1_swa_20       | loop1_swa_20_lerp  | loop1_swa_20_out   |
| loop1_mlp_19_norm  | loop1_swa_20_lerp  | loop1_swa_20_skip  |
| loop1_swa_20_lerp  | loop1_swa_20_norm  | loop1_swa_20_mix   |
| loop1_swa_20_norm  | loop1_mlp_21       | loop1_mlp_21_in    |
| loop1_mlp_21       | loop1_mlp_21_lerp  | loop1_mlp_21_out   |
| loop1_swa_20_norm  | loop1_mlp_21_lerp  | loop1_mlp_21_skip  |
| loop1_mlp_21_lerp  | loop1_mlp_21_norm  | loop1_mlp_21_mix   |
| loop1_mlp_21_norm  | loop1_attn_22      | loop1_attn_22_in   |
| loop1_attn_22      | loop1_attn_22_lerp | loop1_attn_22_out  |
| loop1_mlp_21_norm  | loop1_attn_22_lerp | loop1_attn_22_skip |
| loop1_attn_22_lerp | loop1_attn_22_norm | loop1_attn_22_mix  |
| loop1_attn_22_norm | loop1_mlp_23       | loop1_mlp_23_in    |
| loop1_mlp_23       | loop1_mlp_23_lerp  | loop1_mlp_23_out   |
| loop1_attn_22_norm | loop1_mlp_23_lerp  | loop1_mlp_23_skip  |
| loop1_mlp_23_lerp  | loop1_mlp_23_norm  | loop1_mlp_23_mix   |
| loop1_mlp_23_norm  | loop1_swa_24       | loop1_swa_24_in    |
| loop1_swa_24       | loop1_swa_24_lerp  | loop1_swa_24_out   |
| loop1_mlp_23_norm  | loop1_swa_24_lerp  | loop1_swa_24_skip  |
| loop1_swa_24_lerp  | loop1_swa_24_norm  | loop1_swa_24_mix   |
| loop1_swa_24_norm  | loop1_mlp_25       | loop1_mlp_25_in    |
| loop1_mlp_25       | loop1_mlp_25_lerp  | loop1_mlp_25_out   |
| loop1_swa_24_norm  | loop1_mlp_25_lerp  | loop1_mlp_25_skip  |
| loop1_mlp_25_lerp  | loop1_mlp_25_norm  | loop1_mlp_25_mix   |
| loop1_mlp_25_norm  | loop1_swa_26       | loop1_swa_26_in    |
| loop1_swa_26       | loop1_swa_26_lerp  | loop1_swa_26_out   |
| loop1_mlp_25_norm  | loop1_swa_26_lerp  | loop1_swa_26_skip  |
| loop1_swa_26_lerp  | loop1_swa_26_norm  | loop1_swa_26_mix   |
| loop1_swa_26_norm  | loop1_mlp_27       | loop1_mlp_27_in    |
| loop1_mlp_27       | loop1_mlp_27_lerp  | loop1_mlp_27_out   |
| loop1_swa_26_norm  | loop1_mlp_27_lerp  | loop1_mlp_27_skip  |
| loop1_mlp_27_lerp  | loop1_mlp_27_norm  | loop1_mlp_27_mix   |
| loop1_mlp_27_norm  | loop1_swa_28       | loop1_swa_28_in    |
| loop1_swa_28       | loop1_swa_28_lerp  | loop1_swa_28_out   |
| loop1_mlp_27_norm  | loop1_swa_28_lerp  | loop1_swa_28_skip  |
| loop1_swa_28_lerp  | loop1_swa_28_norm  | loop1_swa_28_mix   |
| loop1_swa_28_norm  | loop1_mlp_29       | loop1_mlp_29_in    |
| loop1_mlp_29       | loop1_mlp_29_lerp  | loop1_mlp_29_out   |
| loop1_swa_28_norm  | loop1_mlp_29_lerp  | loop1_mlp_29_skip  |
| loop1_mlp_29_lerp  | loop1_mlp_29_norm  | loop1_mlp_29_mix   |
| loop1_mlp_29_norm  | loop1_attn_30      | loop1_attn_30_in   |
| loop1_attn_30      | loop1_attn_30_lerp | loop1_attn_30_out  |
| loop1_mlp_29_norm  | loop1_attn_30_lerp | loop1_attn_30_skip |
| loop1_attn_30_lerp | loop1_attn_30_norm | loop1_attn_30_mix  |
| loop1_attn_30_norm | loop1_mlp_31       | loop1_mlp_31_in    |
| loop1_mlp_31       | loop1_mlp_31_lerp  | loop1_mlp_31_out   |
| loop1_attn_30_norm | loop1_mlp_31_lerp  | loop1_mlp_31_skip  |
| loop1_mlp_31_lerp  | loop1_mlp_31_norm  | loop1_mlp_31_mix   |
| loop1_mlp_31_norm  | loop1_swa_32       | loop1_swa_32_in    |
| loop1_swa_32       | loop1_swa_32_lerp  | loop1_swa_32_out   |
| loop1_mlp_31_norm  | loop1_swa_32_lerp  | loop1_swa_32_skip  |
| loop1_swa_32_lerp  | loop1_swa_32_norm  | loop1_swa_32_mix   |
| loop1_swa_32_norm  | loop1_mlp_33       | loop1_mlp_33_in    |
| loop1_mlp_33       | loop1_mlp_33_lerp  | loop1_mlp_33_out   |
| loop1_swa_32_norm  | loop1_mlp_33_lerp  | loop1_mlp_33_skip  |
| loop1_mlp_33_lerp  | loop1_mlp_33_norm  | loop1_mlp_33_mix   |
| loop1_mlp_33_norm  | loop1_swa_34       | loop1_swa_34_in    |
| loop1_swa_34       | loop1_swa_34_lerp  | loop1_swa_34_out   |
| loop1_mlp_33_norm  | loop1_swa_34_lerp  | loop1_swa_34_skip  |
| loop1_swa_34_lerp  | loop1_swa_34_norm  | loop1_swa_34_mix   |
| loop1_swa_34_norm  | loop1_mlp_35       | loop1_mlp_35_in    |
| loop1_mlp_35       | loop1_mlp_35_lerp  | loop1_mlp_35_out   |
| loop1_swa_34_norm  | loop1_mlp_35_lerp  | loop1_mlp_35_skip  |
| loop1_mlp_35_lerp  | loop1_mlp_35_norm  | loop1_mlp_35_mix   |
| loop1_mlp_35_norm  | loop1_swa_36       | loop1_swa_36_in    |
| loop1_swa_36       | loop1_swa_36_lerp  | loop1_swa_36_out   |
| loop1_mlp_35_norm  | loop1_swa_36_lerp  | loop1_swa_36_skip  |
| loop1_swa_36_lerp  | loop1_swa_36_norm  | loop1_swa_36_mix   |
| loop1_swa_36_norm  | loop1_mlp_37       | loop1_mlp_37_in    |
| loop1_mlp_37       | loop1_mlp_37_lerp  | loop1_mlp_37_out   |
| loop1_swa_36_norm  | loop1_mlp_37_lerp  | loop1_mlp_37_skip  |
| loop1_mlp_37_lerp  | loop1_mlp_37_norm  | loop1_mlp_37_mix   |
| loop1_mlp_37_norm  | loop1_attn_38      | loop1_attn_38_in   |
| loop1_attn_38      | loop1_attn_38_lerp | loop1_attn_38_out  |
| loop1_mlp_37_norm  | loop1_attn_38_lerp | loop1_attn_38_skip |
| loop1_attn_38_lerp | loop1_attn_38_norm | loop1_attn_38_mix  |
| loop1_attn_38_norm | loop1_mlp_39       | loop1_mlp_39_in    |
| loop1_mlp_39       | loop1_mlp_39_lerp  | loop1_mlp_39_out   |
| loop1_attn_38_norm | loop1_mlp_39_lerp  | loop1_mlp_39_skip  |
| loop1_mlp_39_lerp  | loop1_mlp_39_norm  | loop1_mlp_39_mix   |
| loop1_mlp_39_norm  | loop1_swa_40       | loop1_swa_40_in    |
| loop1_swa_40       | loop1_swa_40_lerp  | loop1_swa_40_out   |
| loop1_mlp_39_norm  | loop1_swa_40_lerp  | loop1_swa_40_skip  |
| loop1_swa_40_lerp  | loop1_swa_40_norm  | loop1_swa_40_mix   |
| loop1_swa_40_norm  | loop1_mlp_41       | loop1_mlp_41_in    |
| loop1_mlp_41       | loop1_mlp_41_lerp  | loop1_mlp_41_out   |
| loop1_swa_40_norm  | loop1_mlp_41_lerp  | loop1_mlp_41_skip  |
| loop1_mlp_41_lerp  | loop1_mlp_41_norm  | loop1_mlp_41_mix   |
| loop1_mlp_41_norm  | loop1_swa_42       | loop1_swa_42_in    |
| loop1_swa_42       | loop1_swa_42_lerp  | loop1_swa_42_out   |
| loop1_mlp_41_norm  | loop1_swa_42_lerp  | loop1_swa_42_skip  |
| loop1_swa_42_lerp  | loop1_swa_42_norm  | loop1_swa_42_mix   |
| loop1_swa_42_norm  | loop1_mlp_43       | loop1_mlp_43_in    |
| loop1_mlp_43       | loop1_mlp_43_lerp  | loop1_mlp_43_out   |
| loop1_swa_42_norm  | loop1_mlp_43_lerp  | loop1_mlp_43_skip  |
| loop1_mlp_43_lerp  | loop1_mlp_43_norm  | loop1_mlp_43_mix   |
| loop1_mlp_43_norm  | loop1_swa_44       | loop1_swa_44_in    |
| loop1_swa_44       | loop1_swa_44_lerp  | loop1_swa_44_out   |
| loop1_mlp_43_norm  | loop1_swa_44_lerp  | loop1_swa_44_skip  |
| loop1_swa_44_lerp  | loop1_swa_44_norm  | loop1_swa_44_mix   |
| loop1_swa_44_norm  | loop1_mlp_45       | loop1_mlp_45_in    |
| loop1_mlp_45       | loop1_mlp_45_lerp  | loop1_mlp_45_out   |
| loop1_swa_44_norm  | loop1_mlp_45_lerp  | loop1_mlp_45_skip  |
| loop1_mlp_45_lerp  | loop1_mlp_45_norm  | loop1_mlp_45_mix   |
| loop1_mlp_45_norm  | loop1_attn_46      | loop1_attn_46_in   |
| loop1_attn_46      | loop1_attn_46_lerp | loop1_attn_46_out  |
| loop1_mlp_45_norm  | loop1_attn_46_lerp | loop1_attn_46_skip |
| loop1_attn_46_lerp | loop1_attn_46_norm | loop1_attn_46_mix  |
| loop1_attn_46_norm | loop1_mlp_47       | loop1_mlp_47_in    |
| loop1_mlp_47       | loop1_mlp_47_lerp  | loop1_mlp_47_out   |
| loop1_attn_46_norm | loop1_mlp_47_lerp  | loop1_mlp_47_skip  |
| loop1_mlp_47_lerp  | loop1_mlp_47_norm  | loop1_mlp_47_mix   |
| loop1_mlp_47_norm  | loop1_swa_48       | loop1_swa_48_in    |
| loop1_swa_48       | loop1_swa_48_lerp  | loop1_swa_48_out   |
| loop1_mlp_47_norm  | loop1_swa_48_lerp  | loop1_swa_48_skip  |
| loop1_swa_48_lerp  | loop1_swa_48_norm  | loop1_swa_48_mix   |
| loop1_swa_48_norm  | loop1_mlp_49       | loop1_mlp_49_in    |
| loop1_mlp_49       | loop1_mlp_49_lerp  | loop1_mlp_49_out   |
| loop1_swa_48_norm  | loop1_mlp_49_lerp  | loop1_mlp_49_skip  |
| loop1_mlp_49_lerp  | loop1_mlp_49_norm  | loop1_mlp_49_mix   |
| loop1_mlp_49_norm  | loop1_swa_50       | loop1_swa_50_in    |
| loop1_swa_50       | loop1_swa_50_lerp  | loop1_swa_50_out   |
| loop1_mlp_49_norm  | loop1_swa_50_lerp  | loop1_swa_50_skip  |
| loop1_swa_50_lerp  | loop1_swa_50_norm  | loop1_swa_50_mix   |
| loop1_swa_50_norm  | loop1_mlp_51       | loop1_mlp_51_in    |
| loop1_mlp_51       | loop1_mlp_51_lerp  | loop1_mlp_51_out   |
| loop1_swa_50_norm  | loop1_mlp_51_lerp  | loop1_mlp_51_skip  |
| loop1_mlp_51_lerp  | loop1_mlp_51_norm  | loop1_mlp_51_mix   |
| loop1_mlp_51_norm  | loop1_swa_52       | loop1_swa_52_in    |
| loop1_swa_52       | loop1_swa_52_lerp  | loop1_swa_52_out   |
| loop1_mlp_51_norm  | loop1_swa_52_lerp  | loop1_swa_52_skip  |
| loop1_swa_52_lerp  | loop1_swa_52_norm  | loop1_swa_52_mix   |
| loop1_swa_52_norm  | loop1_mlp_53       | loop1_mlp_53_in    |
| loop1_mlp_53       | loop1_mlp_53_lerp  | loop1_mlp_53_out   |
| loop1_swa_52_norm  | loop1_mlp_53_lerp  | loop1_mlp_53_skip  |
| loop1_mlp_53_lerp  | loop1_mlp_53_norm  | loop1_mlp_53_mix   |
| loop1_mlp_53_norm  | loop1_attn_54      | loop1_attn_54_in   |
| loop1_attn_54      | loop1_attn_54_lerp | loop1_attn_54_out  |
| loop1_mlp_53_norm  | loop1_attn_54_lerp | loop1_attn_54_skip |
| loop1_attn_54_lerp | loop1_attn_54_norm | loop1_attn_54_mix  |
| loop1_attn_54_norm | loop1_mlp_55       | loop1_mlp_55_in    |
| loop1_mlp_55       | loop1_mlp_55_lerp  | loop1_mlp_55_out   |
| loop1_attn_54_norm | loop1_mlp_55_lerp  | loop1_mlp_55_skip  |
| loop1_mlp_55_lerp  | loop1_mlp_55_norm  | loop1_mlp_55_mix   |
| loop1_mlp_55_norm  | loop2_inject       | s_1                |
| in_mlp_15_norm     | loop2_inject       | h_in_2             |
| loop2_inject       | loop2_swa_16       | loop2_swa_16_in    |
| loop2_swa_16       | loop2_swa_16_lerp  | loop2_swa_16_out   |
| loop2_inject       | loop2_swa_16_lerp  | loop2_swa_16_skip  |
| loop2_swa_16_lerp  | loop2_swa_16_norm  | loop2_swa_16_mix   |
| loop2_swa_16_norm  | loop2_mlp_17       | loop2_mlp_17_in    |
| loop2_mlp_17       | loop2_mlp_17_lerp  | loop2_mlp_17_out   |
| loop2_swa_16_norm  | loop2_mlp_17_lerp  | loop2_mlp_17_skip  |
| loop2_mlp_17_lerp  | loop2_mlp_17_norm  | loop2_mlp_17_mix   |
| loop2_mlp_17_norm  | loop2_swa_18       | loop2_swa_18_in    |
| loop2_swa_18       | loop2_swa_18_lerp  | loop2_swa_18_out   |
| loop2_mlp_17_norm  | loop2_swa_18_lerp  | loop2_swa_18_skip  |
| loop2_swa_18_lerp  | loop2_swa_18_norm  | loop2_swa_18_mix   |
| loop2_swa_18_norm  | loop2_mlp_19       | loop2_mlp_19_in    |
| loop2_mlp_19       | loop2_mlp_19_lerp  | loop2_mlp_19_out   |
| loop2_swa_18_norm  | loop2_mlp_19_lerp  | loop2_mlp_19_skip  |
| loop2_mlp_19_lerp  | loop2_mlp_19_norm  | loop2_mlp_19_mix   |
| loop2_mlp_19_norm  | loop2_swa_20       | loop2_swa_20_in    |
| loop2_swa_20       | loop2_swa_20_lerp  | loop2_swa_20_out   |
| loop2_mlp_19_norm  | loop2_swa_20_lerp  | loop2_swa_20_skip  |
| loop2_swa_20_lerp  | loop2_swa_20_norm  | loop2_swa_20_mix   |
| loop2_swa_20_norm  | loop2_mlp_21       | loop2_mlp_21_in    |
| loop2_mlp_21       | loop2_mlp_21_lerp  | loop2_mlp_21_out   |
| loop2_swa_20_norm  | loop2_mlp_21_lerp  | loop2_mlp_21_skip  |
| loop2_mlp_21_lerp  | loop2_mlp_21_norm  | loop2_mlp_21_mix   |
| loop2_mlp_21_norm  | loop2_attn_22      | loop2_attn_22_in   |
| loop2_attn_22      | loop2_attn_22_lerp | loop2_attn_22_out  |
| loop2_mlp_21_norm  | loop2_attn_22_lerp | loop2_attn_22_skip |
| loop2_attn_22_lerp | loop2_attn_22_norm | loop2_attn_22_mix  |
| loop2_attn_22_norm | loop2_mlp_23       | loop2_mlp_23_in    |
| loop2_mlp_23       | loop2_mlp_23_lerp  | loop2_mlp_23_out   |
| loop2_attn_22_norm | loop2_mlp_23_lerp  | loop2_mlp_23_skip  |
| loop2_mlp_23_lerp  | loop2_mlp_23_norm  | loop2_mlp_23_mix   |
| loop2_mlp_23_norm  | loop2_swa_24       | loop2_swa_24_in    |
| loop2_swa_24       | loop2_swa_24_lerp  | loop2_swa_24_out   |
| loop2_mlp_23_norm  | loop2_swa_24_lerp  | loop2_swa_24_skip  |
| loop2_swa_24_lerp  | loop2_swa_24_norm  | loop2_swa_24_mix   |
| loop2_swa_24_norm  | loop2_mlp_25       | loop2_mlp_25_in    |
| loop2_mlp_25       | loop2_mlp_25_lerp  | loop2_mlp_25_out   |
| loop2_swa_24_norm  | loop2_mlp_25_lerp  | loop2_mlp_25_skip  |
| loop2_mlp_25_lerp  | loop2_mlp_25_norm  | loop2_mlp_25_mix   |
| loop2_mlp_25_norm  | loop2_swa_26       | loop2_swa_26_in    |
| loop2_swa_26       | loop2_swa_26_lerp  | loop2_swa_26_out   |
| loop2_mlp_25_norm  | loop2_swa_26_lerp  | loop2_swa_26_skip  |
| loop2_swa_26_lerp  | loop2_swa_26_norm  | loop2_swa_26_mix   |
| loop2_swa_26_norm  | loop2_mlp_27       | loop2_mlp_27_in    |
| loop2_mlp_27       | loop2_mlp_27_lerp  | loop2_mlp_27_out   |
| loop2_swa_26_norm  | loop2_mlp_27_lerp  | loop2_mlp_27_skip  |
| loop2_mlp_27_lerp  | loop2_mlp_27_norm  | loop2_mlp_27_mix   |
| loop2_mlp_27_norm  | loop2_swa_28       | loop2_swa_28_in    |
| loop2_swa_28       | loop2_swa_28_lerp  | loop2_swa_28_out   |
| loop2_mlp_27_norm  | loop2_swa_28_lerp  | loop2_swa_28_skip  |
| loop2_swa_28_lerp  | loop2_swa_28_norm  | loop2_swa_28_mix   |
| loop2_swa_28_norm  | loop2_mlp_29       | loop2_mlp_29_in    |
| loop2_mlp_29       | loop2_mlp_29_lerp  | loop2_mlp_29_out   |
| loop2_swa_28_norm  | loop2_mlp_29_lerp  | loop2_mlp_29_skip  |
| loop2_mlp_29_lerp  | loop2_mlp_29_norm  | loop2_mlp_29_mix   |
| loop2_mlp_29_norm  | loop2_attn_30      | loop2_attn_30_in   |
| loop2_attn_30      | loop2_attn_30_lerp | loop2_attn_30_out  |
| loop2_mlp_29_norm  | loop2_attn_30_lerp | loop2_attn_30_skip |
| loop2_attn_30_lerp | loop2_attn_30_norm | loop2_attn_30_mix  |
| loop2_attn_30_norm | loop2_mlp_31       | loop2_mlp_31_in    |
| loop2_mlp_31       | loop2_mlp_31_lerp  | loop2_mlp_31_out   |
| loop2_attn_30_norm | loop2_mlp_31_lerp  | loop2_mlp_31_skip  |
| loop2_mlp_31_lerp  | loop2_mlp_31_norm  | loop2_mlp_31_mix   |
| loop2_mlp_31_norm  | loop2_swa_32       | loop2_swa_32_in    |
| loop2_swa_32       | loop2_swa_32_lerp  | loop2_swa_32_out   |
| loop2_mlp_31_norm  | loop2_swa_32_lerp  | loop2_swa_32_skip  |
| loop2_swa_32_lerp  | loop2_swa_32_norm  | loop2_swa_32_mix   |
| loop2_swa_32_norm  | loop2_mlp_33       | loop2_mlp_33_in    |
| loop2_mlp_33       | loop2_mlp_33_lerp  | loop2_mlp_33_out   |
| loop2_swa_32_norm  | loop2_mlp_33_lerp  | loop2_mlp_33_skip  |
| loop2_mlp_33_lerp  | loop2_mlp_33_norm  | loop2_mlp_33_mix   |
| loop2_mlp_33_norm  | loop2_swa_34       | loop2_swa_34_in    |
| loop2_swa_34       | loop2_swa_34_lerp  | loop2_swa_34_out   |
| loop2_mlp_33_norm  | loop2_swa_34_lerp  | loop2_swa_34_skip  |
| loop2_swa_34_lerp  | loop2_swa_34_norm  | loop2_swa_34_mix   |
| loop2_swa_34_norm  | loop2_mlp_35       | loop2_mlp_35_in    |
| loop2_mlp_35       | loop2_mlp_35_lerp  | loop2_mlp_35_out   |
| loop2_swa_34_norm  | loop2_mlp_35_lerp  | loop2_mlp_35_skip  |
| loop2_mlp_35_lerp  | loop2_mlp_35_norm  | loop2_mlp_35_mix   |
| loop2_mlp_35_norm  | loop2_swa_36       | loop2_swa_36_in    |
| loop2_swa_36       | loop2_swa_36_lerp  | loop2_swa_36_out   |
| loop2_mlp_35_norm  | loop2_swa_36_lerp  | loop2_swa_36_skip  |
| loop2_swa_36_lerp  | loop2_swa_36_norm  | loop2_swa_36_mix   |
| loop2_swa_36_norm  | loop2_mlp_37       | loop2_mlp_37_in    |
| loop2_mlp_37       | loop2_mlp_37_lerp  | loop2_mlp_37_out   |
| loop2_swa_36_norm  | loop2_mlp_37_lerp  | loop2_mlp_37_skip  |
| loop2_mlp_37_lerp  | loop2_mlp_37_norm  | loop2_mlp_37_mix   |
| loop2_mlp_37_norm  | loop2_attn_38      | loop2_attn_38_in   |
| loop2_attn_38      | loop2_attn_38_lerp | loop2_attn_38_out  |
| loop2_mlp_37_norm  | loop2_attn_38_lerp | loop2_attn_38_skip |
| loop2_attn_38_lerp | loop2_attn_38_norm | loop2_attn_38_mix  |
| loop2_attn_38_norm | loop2_mlp_39       | loop2_mlp_39_in    |
| loop2_mlp_39       | loop2_mlp_39_lerp  | loop2_mlp_39_out   |
| loop2_attn_38_norm | loop2_mlp_39_lerp  | loop2_mlp_39_skip  |
| loop2_mlp_39_lerp  | loop2_mlp_39_norm  | loop2_mlp_39_mix   |
| loop2_mlp_39_norm  | loop2_swa_40       | loop2_swa_40_in    |
| loop2_swa_40       | loop2_swa_40_lerp  | loop2_swa_40_out   |
| loop2_mlp_39_norm  | loop2_swa_40_lerp  | loop2_swa_40_skip  |
| loop2_swa_40_lerp  | loop2_swa_40_norm  | loop2_swa_40_mix   |
| loop2_swa_40_norm  | loop2_mlp_41       | loop2_mlp_41_in    |
| loop2_mlp_41       | loop2_mlp_41_lerp  | loop2_mlp_41_out   |
| loop2_swa_40_norm  | loop2_mlp_41_lerp  | loop2_mlp_41_skip  |
| loop2_mlp_41_lerp  | loop2_mlp_41_norm  | loop2_mlp_41_mix   |
| loop2_mlp_41_norm  | loop2_swa_42       | loop2_swa_42_in    |
| loop2_swa_42       | loop2_swa_42_lerp  | loop2_swa_42_out   |
| loop2_mlp_41_norm  | loop2_swa_42_lerp  | loop2_swa_42_skip  |
| loop2_swa_42_lerp  | loop2_swa_42_norm  | loop2_swa_42_mix   |
| loop2_swa_42_norm  | loop2_mlp_43       | loop2_mlp_43_in    |
| loop2_mlp_43       | loop2_mlp_43_lerp  | loop2_mlp_43_out   |
| loop2_swa_42_norm  | loop2_mlp_43_lerp  | loop2_mlp_43_skip  |
| loop2_mlp_43_lerp  | loop2_mlp_43_norm  | loop2_mlp_43_mix   |
| loop2_mlp_43_norm  | loop2_swa_44       | loop2_swa_44_in    |
| loop2_swa_44       | loop2_swa_44_lerp  | loop2_swa_44_out   |
| loop2_mlp_43_norm  | loop2_swa_44_lerp  | loop2_swa_44_skip  |
| loop2_swa_44_lerp  | loop2_swa_44_norm  | loop2_swa_44_mix   |
| loop2_swa_44_norm  | loop2_mlp_45       | loop2_mlp_45_in    |
| loop2_mlp_45       | loop2_mlp_45_lerp  | loop2_mlp_45_out   |
| loop2_swa_44_norm  | loop2_mlp_45_lerp  | loop2_mlp_45_skip  |
| loop2_mlp_45_lerp  | loop2_mlp_45_norm  | loop2_mlp_45_mix   |
| loop2_mlp_45_norm  | loop2_attn_46      | loop2_attn_46_in   |
| loop2_attn_46      | loop2_attn_46_lerp | loop2_attn_46_out  |
| loop2_mlp_45_norm  | loop2_attn_46_lerp | loop2_attn_46_skip |
| loop2_attn_46_lerp | loop2_attn_46_norm | loop2_attn_46_mix  |
| loop2_attn_46_norm | loop2_mlp_47       | loop2_mlp_47_in    |
| loop2_mlp_47       | loop2_mlp_47_lerp  | loop2_mlp_47_out   |
| loop2_attn_46_norm | loop2_mlp_47_lerp  | loop2_mlp_47_skip  |
| loop2_mlp_47_lerp  | loop2_mlp_47_norm  | loop2_mlp_47_mix   |
| loop2_mlp_47_norm  | loop2_swa_48       | loop2_swa_48_in    |
| loop2_swa_48       | loop2_swa_48_lerp  | loop2_swa_48_out   |
| loop2_mlp_47_norm  | loop2_swa_48_lerp  | loop2_swa_48_skip  |
| loop2_swa_48_lerp  | loop2_swa_48_norm  | loop2_swa_48_mix   |
| loop2_swa_48_norm  | loop2_mlp_49       | loop2_mlp_49_in    |
| loop2_mlp_49       | loop2_mlp_49_lerp  | loop2_mlp_49_out   |
| loop2_swa_48_norm  | loop2_mlp_49_lerp  | loop2_mlp_49_skip  |
| loop2_mlp_49_lerp  | loop2_mlp_49_norm  | loop2_mlp_49_mix   |
| loop2_mlp_49_norm  | loop2_swa_50       | loop2_swa_50_in    |
| loop2_swa_50       | loop2_swa_50_lerp  | loop2_swa_50_out   |
| loop2_mlp_49_norm  | loop2_swa_50_lerp  | loop2_swa_50_skip  |
| loop2_swa_50_lerp  | loop2_swa_50_norm  | loop2_swa_50_mix   |
| loop2_swa_50_norm  | loop2_mlp_51       | loop2_mlp_51_in    |
| loop2_mlp_51       | loop2_mlp_51_lerp  | loop2_mlp_51_out   |
| loop2_swa_50_norm  | loop2_mlp_51_lerp  | loop2_mlp_51_skip  |
| loop2_mlp_51_lerp  | loop2_mlp_51_norm  | loop2_mlp_51_mix   |
| loop2_mlp_51_norm  | loop2_swa_52       | loop2_swa_52_in    |
| loop2_swa_52       | loop2_swa_52_lerp  | loop2_swa_52_out   |
| loop2_mlp_51_norm  | loop2_swa_52_lerp  | loop2_swa_52_skip  |
| loop2_swa_52_lerp  | loop2_swa_52_norm  | loop2_swa_52_mix   |
| loop2_swa_52_norm  | loop2_mlp_53       | loop2_mlp_53_in    |
| loop2_mlp_53       | loop2_mlp_53_lerp  | loop2_mlp_53_out   |
| loop2_swa_52_norm  | loop2_mlp_53_lerp  | loop2_mlp_53_skip  |
| loop2_mlp_53_lerp  | loop2_mlp_53_norm  | loop2_mlp_53_mix   |
| loop2_mlp_53_norm  | loop2_attn_54      | loop2_attn_54_in   |
| loop2_attn_54      | loop2_attn_54_lerp | loop2_attn_54_out  |
| loop2_mlp_53_norm  | loop2_attn_54_lerp | loop2_attn_54_skip |
| loop2_attn_54_lerp | loop2_attn_54_norm | loop2_attn_54_mix  |
| loop2_attn_54_norm | loop2_mlp_55       | loop2_mlp_55_in    |
| loop2_mlp_55       | loop2_mlp_55_lerp  | loop2_mlp_55_out   |
| loop2_attn_54_norm | loop2_mlp_55_lerp  | loop2_mlp_55_skip  |
| loop2_mlp_55_lerp  | loop2_mlp_55_norm  | loop2_mlp_55_mix   |
| loop2_mlp_55_norm  | out_swa_56         | out_swa_56_in      |
| out_swa_56         | out_swa_56_lerp    | out_swa_56_out     |
| loop2_mlp_55_norm  | out_swa_56_lerp    | out_swa_56_skip    |
| out_swa_56_lerp    | out_swa_56_norm    | out_swa_56_mix     |
| out_swa_56_norm    | out_mlp_57         | out_mlp_57_in      |
| out_mlp_57         | out_mlp_57_lerp    | out_mlp_57_out     |
| out_swa_56_norm    | out_mlp_57_lerp    | out_mlp_57_skip    |
| out_mlp_57_lerp    | out_mlp_57_norm    | out_mlp_57_mix     |
| out_mlp_57_norm    | out_swa_58         | out_swa_58_in      |
| out_swa_58         | out_swa_58_lerp    | out_swa_58_out     |
| out_mlp_57_norm    | out_swa_58_lerp    | out_swa_58_skip    |
| out_swa_58_lerp    | out_swa_58_norm    | out_swa_58_mix     |
| out_swa_58_norm    | out_mlp_59         | out_mlp_59_in      |
| out_mlp_59         | out_mlp_59_lerp    | out_mlp_59_out     |
| out_swa_58_norm    | out_mlp_59_lerp    | out_mlp_59_skip    |
| out_mlp_59_lerp    | out_mlp_59_norm    | out_mlp_59_mix     |
| out_mlp_59_norm    | out_swa_60         | out_swa_60_in      |
| out_swa_60         | out_swa_60_lerp    | out_swa_60_out     |
| out_mlp_59_norm    | out_swa_60_lerp    | out_swa_60_skip    |
| out_swa_60_lerp    | out_swa_60_norm    | out_swa_60_mix     |
| out_swa_60_norm    | out_mlp_61         | out_mlp_61_in      |
| out_mlp_61         | out_mlp_61_lerp    | out_mlp_61_out     |
| out_swa_60_norm    | out_mlp_61_lerp    | out_mlp_61_skip    |
| out_mlp_61_lerp    | out_mlp_61_norm    | out_mlp_61_mix     |
| out_mlp_61_norm    | out_attn_62        | out_attn_62_in     |
| out_attn_62        | out_attn_62_lerp   | out_attn_62_out    |
| out_mlp_61_norm    | out_attn_62_lerp   | out_attn_62_skip   |
| out_attn_62_lerp   | out_attn_62_norm   | out_attn_62_mix    |
| out_attn_62_norm   | out_mlp_63         | out_mlp_63_in      |
| out_mlp_63         | out_mlp_63_lerp    | out_mlp_63_out     |
| out_attn_62_norm   | out_mlp_63_lerp    | out_mlp_63_skip    |
| out_mlp_63_lerp    | out_mlp_63_norm    | out_mlp_63_mix     |
| out_mlp_63_norm    | out_swa_64         | out_swa_64_in      |
| out_swa_64         | out_swa_64_lerp    | out_swa_64_out     |
| out_mlp_63_norm    | out_swa_64_lerp    | out_swa_64_skip    |
| out_swa_64_lerp    | out_swa_64_norm    | out_swa_64_mix     |
| out_swa_64_norm    | out_mlp_65         | out_mlp_65_in      |
| out_mlp_65         | out_mlp_65_lerp    | out_mlp_65_out     |
| out_swa_64_norm    | out_mlp_65_lerp    | out_mlp_65_skip    |
| out_mlp_65_lerp    | out_mlp_65_norm    | out_mlp_65_mix     |
| out_mlp_65_norm    | out_swa_66         | out_swa_66_in      |
| out_swa_66         | out_swa_66_lerp    | out_swa_66_out     |
| out_mlp_65_norm    | out_swa_66_lerp    | out_swa_66_skip    |
| out_swa_66_lerp    | out_swa_66_norm    | out_swa_66_mix     |
| out_swa_66_norm    | out_mlp_67         | out_mlp_67_in      |
| out_mlp_67         | out_mlp_67_lerp    | out_mlp_67_out     |
| out_swa_66_norm    | out_mlp_67_lerp    | out_mlp_67_skip    |
| out_mlp_67_lerp    | out_mlp_67_norm    | out_mlp_67_mix     |
| out_mlp_67_norm    | out_swa_68         | out_swa_68_in      |
| out_swa_68         | out_swa_68_lerp    | out_swa_68_out     |
| out_mlp_67_norm    | out_swa_68_lerp    | out_swa_68_skip    |
| out_swa_68_lerp    | out_swa_68_norm    | out_swa_68_mix     |
| out_swa_68_norm    | out_mlp_69         | out_mlp_69_in      |
| out_mlp_69         | out_mlp_69_lerp    | out_mlp_69_out     |
| out_swa_68_norm    | out_mlp_69_lerp    | out_mlp_69_skip    |
| out_mlp_69_lerp    | out_mlp_69_norm    | out_mlp_69_mix     |
| out_mlp_69_norm    | out_attn_70        | out_attn_70_in     |
| out_attn_70        | out_attn_70_lerp   | out_attn_70_out    |
| out_mlp_69_norm    | out_attn_70_lerp   | out_attn_70_skip   |
| out_attn_70_lerp   | out_attn_70_norm   | out_attn_70_mix    |
| out_attn_70_norm   | out_mlp_71         | out_mlp_71_in      |
| out_mlp_71         | out_mlp_71_lerp    | out_mlp_71_out     |
| out_attn_70_norm   | out_mlp_71_lerp    | out_mlp_71_skip    |
| out_mlp_71_lerp    | out_mlp_71_norm    | out_mlp_71_mix     |
| out_mlp_71_norm    | lm_head            | h_final            |
| lm_head            | logits             | logits             |

## groups

| Name             | Layers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_block      | in_swa_0, in_swa_0_lerp, in_swa_0_norm, in_mlp_1, in_mlp_1_lerp, in_mlp_1_norm, in_swa_2, in_swa_2_lerp, in_swa_2_norm, in_mlp_3, in_mlp_3_lerp, in_mlp_3_norm, in_swa_4, in_swa_4_lerp, in_swa_4_norm, in_mlp_5, in_mlp_5_lerp, in_mlp_5_norm, in_attn_6, in_attn_6_lerp, in_attn_6_norm, in_mlp_7, in_mlp_7_lerp, in_mlp_7_norm, in_swa_8, in_swa_8_lerp, in_swa_8_norm, in_mlp_9, in_mlp_9_lerp, in_mlp_9_norm, in_swa_10, in_swa_10_lerp, in_swa_10_norm, in_mlp_11, in_mlp_11_lerp, in_mlp_11_norm, in_swa_12, in_swa_12_lerp, in_swa_12_norm, in_mlp_13, in_mlp_13_lerp, in_mlp_13_norm, in_attn_14, in_attn_14_lerp, in_attn_14_norm, in_mlp_15, in_mlp_15_lerp, in_mlp_15_norm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| recurrent_loop_1 | loop1_inject, loop1_swa_16, loop1_swa_16_lerp, loop1_swa_16_norm, loop1_mlp_17, loop1_mlp_17_lerp, loop1_mlp_17_norm, loop1_swa_18, loop1_swa_18_lerp, loop1_swa_18_norm, loop1_mlp_19, loop1_mlp_19_lerp, loop1_mlp_19_norm, loop1_swa_20, loop1_swa_20_lerp, loop1_swa_20_norm, loop1_mlp_21, loop1_mlp_21_lerp, loop1_mlp_21_norm, loop1_attn_22, loop1_attn_22_lerp, loop1_attn_22_norm, loop1_mlp_23, loop1_mlp_23_lerp, loop1_mlp_23_norm, loop1_swa_24, loop1_swa_24_lerp, loop1_swa_24_norm, loop1_mlp_25, loop1_mlp_25_lerp, loop1_mlp_25_norm, loop1_swa_26, loop1_swa_26_lerp, loop1_swa_26_norm, loop1_mlp_27, loop1_mlp_27_lerp, loop1_mlp_27_norm, loop1_swa_28, loop1_swa_28_lerp, loop1_swa_28_norm, loop1_mlp_29, loop1_mlp_29_lerp, loop1_mlp_29_norm, loop1_attn_30, loop1_attn_30_lerp, loop1_attn_30_norm, loop1_mlp_31, loop1_mlp_31_lerp, loop1_mlp_31_norm, loop1_swa_32, loop1_swa_32_lerp, loop1_swa_32_norm, loop1_mlp_33, loop1_mlp_33_lerp, loop1_mlp_33_norm, loop1_swa_34, loop1_swa_34_lerp, loop1_swa_34_norm, loop1_mlp_35, loop1_mlp_35_lerp, loop1_mlp_35_norm, loop1_swa_36, loop1_swa_36_lerp, loop1_swa_36_norm, loop1_mlp_37, loop1_mlp_37_lerp, loop1_mlp_37_norm, loop1_attn_38, loop1_attn_38_lerp, loop1_attn_38_norm, loop1_mlp_39, loop1_mlp_39_lerp, loop1_mlp_39_norm, loop1_swa_40, loop1_swa_40_lerp, loop1_swa_40_norm, loop1_mlp_41, loop1_mlp_41_lerp, loop1_mlp_41_norm, loop1_swa_42, loop1_swa_42_lerp, loop1_swa_42_norm, loop1_mlp_43, loop1_mlp_43_lerp, loop1_mlp_43_norm, loop1_swa_44, loop1_swa_44_lerp, loop1_swa_44_norm, loop1_mlp_45, loop1_mlp_45_lerp, loop1_mlp_45_norm, loop1_attn_46, loop1_attn_46_lerp, loop1_attn_46_norm, loop1_mlp_47, loop1_mlp_47_lerp, loop1_mlp_47_norm, loop1_swa_48, loop1_swa_48_lerp, loop1_swa_48_norm, loop1_mlp_49, loop1_mlp_49_lerp, loop1_mlp_49_norm, loop1_swa_50, loop1_swa_50_lerp, loop1_swa_50_norm, loop1_mlp_51, loop1_mlp_51_lerp, loop1_mlp_51_norm, loop1_swa_52, loop1_swa_52_lerp, loop1_swa_52_norm, loop1_mlp_53, loop1_mlp_53_lerp, loop1_mlp_53_norm, loop1_attn_54, loop1_attn_54_lerp, loop1_attn_54_norm, loop1_mlp_55, loop1_mlp_55_lerp, loop1_mlp_55_norm |
| recurrent_loop_2 | loop2_inject, loop2_swa_16, loop2_swa_16_lerp, loop2_swa_16_norm, loop2_mlp_17, loop2_mlp_17_lerp, loop2_mlp_17_norm, loop2_swa_18, loop2_swa_18_lerp, loop2_swa_18_norm, loop2_mlp_19, loop2_mlp_19_lerp, loop2_mlp_19_norm, loop2_swa_20, loop2_swa_20_lerp, loop2_swa_20_norm, loop2_mlp_21, loop2_mlp_21_lerp, loop2_mlp_21_norm, loop2_attn_22, loop2_attn_22_lerp, loop2_attn_22_norm, loop2_mlp_23, loop2_mlp_23_lerp, loop2_mlp_23_norm, loop2_swa_24, loop2_swa_24_lerp, loop2_swa_24_norm, loop2_mlp_25, loop2_mlp_25_lerp, loop2_mlp_25_norm, loop2_swa_26, loop2_swa_26_lerp, loop2_swa_26_norm, loop2_mlp_27, loop2_mlp_27_lerp, loop2_mlp_27_norm, loop2_swa_28, loop2_swa_28_lerp, loop2_swa_28_norm, loop2_mlp_29, loop2_mlp_29_lerp, loop2_mlp_29_norm, loop2_attn_30, loop2_attn_30_lerp, loop2_attn_30_norm, loop2_mlp_31, loop2_mlp_31_lerp, loop2_mlp_31_norm, loop2_swa_32, loop2_swa_32_lerp, loop2_swa_32_norm, loop2_mlp_33, loop2_mlp_33_lerp, loop2_mlp_33_norm, loop2_swa_34, loop2_swa_34_lerp, loop2_swa_34_norm, loop2_mlp_35, loop2_mlp_35_lerp, loop2_mlp_35_norm, loop2_swa_36, loop2_swa_36_lerp, loop2_swa_36_norm, loop2_mlp_37, loop2_mlp_37_lerp, loop2_mlp_37_norm, loop2_attn_38, loop2_attn_38_lerp, loop2_attn_38_norm, loop2_mlp_39, loop2_mlp_39_lerp, loop2_mlp_39_norm, loop2_swa_40, loop2_swa_40_lerp, loop2_swa_40_norm, loop2_mlp_41, loop2_mlp_41_lerp, loop2_mlp_41_norm, loop2_swa_42, loop2_swa_42_lerp, loop2_swa_42_norm, loop2_mlp_43, loop2_mlp_43_lerp, loop2_mlp_43_norm, loop2_swa_44, loop2_swa_44_lerp, loop2_swa_44_norm, loop2_mlp_45, loop2_mlp_45_lerp, loop2_mlp_45_norm, loop2_attn_46, loop2_attn_46_lerp, loop2_attn_46_norm, loop2_mlp_47, loop2_mlp_47_lerp, loop2_mlp_47_norm, loop2_swa_48, loop2_swa_48_lerp, loop2_swa_48_norm, loop2_mlp_49, loop2_mlp_49_lerp, loop2_mlp_49_norm, loop2_swa_50, loop2_swa_50_lerp, loop2_swa_50_norm, loop2_mlp_51, loop2_mlp_51_lerp, loop2_mlp_51_norm, loop2_swa_52, loop2_swa_52_lerp, loop2_swa_52_norm, loop2_mlp_53, loop2_mlp_53_lerp, loop2_mlp_53_norm, loop2_attn_54, loop2_attn_54_lerp, loop2_attn_54_norm, loop2_mlp_55, loop2_mlp_55_lerp, loop2_mlp_55_norm |
| output_block     | out_swa_56, out_swa_56_lerp, out_swa_56_norm, out_mlp_57, out_mlp_57_lerp, out_mlp_57_norm, out_swa_58, out_swa_58_lerp, out_swa_58_norm, out_mlp_59, out_mlp_59_lerp, out_mlp_59_norm, out_swa_60, out_swa_60_lerp, out_swa_60_norm, out_mlp_61, out_mlp_61_lerp, out_mlp_61_norm, out_attn_62, out_attn_62_lerp, out_attn_62_norm, out_mlp_63, out_mlp_63_lerp, out_mlp_63_norm, out_swa_64, out_swa_64_lerp, out_swa_64_norm, out_mlp_65, out_mlp_65_lerp, out_mlp_65_norm, out_swa_66, out_swa_66_lerp, out_swa_66_norm, out_mlp_67, out_mlp_67_lerp, out_mlp_67_norm, out_swa_68, out_swa_68_lerp, out_swa_68_norm, out_mlp_69, out_mlp_69_lerp, out_mlp_69_norm, out_attn_70, out_attn_70_lerp, out_attn_70_norm, out_mlp_71, out_mlp_71_lerp, out_mlp_71_norm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

## invariants
- output_shape: (B, S, vocab_size)
