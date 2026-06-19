# CLSE Integration for Qwen2-VL

This document describes the modifications made to the original Qwen2-VL codebase to integrate **CLSE (Cross-Layer Spectral Evolution)** visual token pruning.

## Modified / Added Files

```
Qwen2VL/
├── modeling_qwen2_vl_clse.py   # modified — Qwen2VLTextModel with pruning logic
└── tools.py                    # new — spectral scoring utilities
```

---

## `tools.py`

Provides three scoring functions used by `Qwen2VLTextModel`.

### `spatial_spectral_score_2d(x, h, w, cutoff_ratio)`

Computes a per-token high-frequency energy score via 2D FFT.

- Reshapes `[B, N, C]` → `[B, C, H, W]` and applies 2D FFT.
- Builds a Gaussian high-pass filter centered at the DC component.
- Applies the filter in the frequency domain, inverts back to spatial domain, and averages over channels.
- Returns shape `[B, N]`.

### `get_evolution_factor(s_L, s_Lk, temp)`

Measures how much the spectral score of each token changed between layer `L` and layer `L+k`:

```
intensity = |s_Lk - s_L| / (mean(s_L) + s_L + ε)
factor    = sigmoid(zscore(intensity) / temp)
```

Returns a value in `(0, 1)` per token — higher means greater spectral evolution.

### `calculate_evolution_score(z_L, z_Lk, attention_weights, image_grid_thw, cutoff, temp, score_type)`

Combines spectral evolution and text-to-visual attention into a single importance score:

| `score_type` | Formula |
|---|---|
| `"clse"` | `evo_factor` |
| `"attn"` | `attn_score` |
| `"clse_attn"` | `evo_factor × attn_score` |

When `image_grid_thw=None` (post-pruning stages where the spatial grid is no longer intact), automatically falls back to attention-only scoring regardless of `score_type`.

---

## `modeling_qwen2_vl_clse.py`

### What changed

The original `Qwen2VLTextModel` `forward()` is replaced with a modified version that inserts visual token pruning during the prefill pass. The rest of the file (vision encoder, `Qwen2VLModel`, `Qwen2VLForConditionalGeneration`, etc.) is unchanged from the original.

The original class is preserved as a commented-out block immediately after the new implementation for reference.

### Key hyperparameters added to `__init__`

| Attribute | Default | Description |
|---|---|---|
| `prune` | `False` | Enable / disable visual token pruning |
| `keep_tokens` | — | *(not used; see `pruning_ratios`)* |
| `pruning_ratios` | `[0.334]` | Fraction of visual tokens to retain at each pruning stage |
| `L_list` | `[0]` | Layer indices at which reference features `Z_L` are recorded |
| `K_list` | `[1]` | Layer indices at which pruning is applied |
| `score_type` | `"clse"` | Token scoring method: `"attn"`, `"clse"`, or `"clse_attn"` |
| `cutoff` | `0.1` | Gaussian high-pass cutoff ratio for spectral filtering |
| `temp` | `0.1` | Temperature for sigmoid normalization of evolution intensity |
| `image_grid_thw` | `None` | Set externally before each forward call to the `[N_images, 3]` grid tensor `(T, H, W)` |

`L_list` and `K_list` are full lists supporting multi-stage pruning; the default single-element values configure a single pruning pass.

### Pruning procedure (executed once per stage at layer `K`)

1. **Record reference features** at layer `L` (`L_list[i]`): snapshot the visual token hidden states as `Z_L`.
2. **Pre-compute attention** at layer `K-1`: run a lightweight partial forward (Q/K projections + M-RoPE + softmax on the last query token only) via `_capture_attention_score`, and cache the result.
3. **Score tokens** at layer `K` (`K_list[i]`): call `calculate_evolution_score` with `Z_L`, the current visual features, and the cached attention map.
4. **Select top-k tokens**: target count = `round(original_visual_len × pruning_ratios[i])`. Keep the highest-scoring tokens, rebuild the sequence by concatenating prefix tokens, selected visual tokens, and text suffix in sorted order.
5. **Trim position ids and `cache_position`** to match the pruned sequence length.
6. **Trim causal masks** (`causal_mask_mapping`) to the pruned sequence via `index_select` on both sequence dimensions.
7. **Recompute position embeddings** (M-RoPE) for the pruned sequence.

> **Multi-stage note**: for stage 1, the spatial grid `(H//2, W//2)` is passed to the spectral scorer. For stage 2+, the spatial grid is no longer valid after pruning, so `image_grid_thw=None` is passed and scoring falls back to attention-only.

### `_capture_attention_score`

A helper that computes the attention weight of the **last query token** over all key tokens, without running the full decoder layer forward. Steps:

1. Apply `input_layernorm`.
2. Project to Q and K.
3. Apply Multimodal RoPE (`apply_multimodal_rotary_pos_emb`).
4. Broadcast K for GQA (`repeat_interleave`).
5. Compute `softmax(Q_last · Kᵀ / √d)` for the last query position only — shape `[B, num_heads, 1, seq_len]`.

This avoids the overhead of a full attention forward while still producing the attention map needed for token scoring.

### Differences from the LLaVA-1.5 CLSE implementation

| | LLaVA-1.5 | Qwen2-VL |
|---|---|---|
| Token count config | `keep_tokens` (absolute) | `pruning_ratios` (relative to original length) |
| KV cache alignment | Explicit slice of `DynamicCache` | Not applied (Qwen2-VL uses a different cache structure) |
| Attention pre-capture | Full layer forward with `output_attentions=True` | Lightweight partial forward (Q/K only) |
| Position embedding | Standard RoPE | Multimodal RoPE (M-RoPE) recomputed after pruning |
| Mask handling | 4D float causal mask sliced directly | `causal_mask_mapping` dict sliced via `index_select` |
| Visual token location | Fixed offsets `image_start=35`, `image_end=611` | Dynamic, detected via `visual_pos_masks` |
