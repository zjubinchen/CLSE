# CLSE Integration for Video-LLaVA

This document describes the modifications made to the original Video-LLaVA codebase to integrate **CLSE (Cross-Layer Spectral Evolution)** visual token pruning.

## Modified / Added Files

```
videollava/model/language_model/
├── clse_model.py   # new — CLSELlamaModel with pruning logic  (replaces gest_model.py)
├── tools.py        # modified — spectral scoring utilities
└── llava_llama.py  # modified — inherit CLSELlamaModel instead of LlamaModel
```

---

## `tools.py`

Provides three scoring functions used by `CLSELlamaModel`.

### `spatial_spectral_score_per_frame(x, t, h, w, cutoff_ratio)`

Computes a per-token high-frequency energy score via **per-frame 2D FFT**.

- Reshapes `[B, N, C]` → `[B*T, C, H, W]` by merging the temporal dimension into the batch dimension.
- Applies 2D FFT independently to each frame.
- Builds a Gaussian high-pass filter centered at the DC component.
- Applies the filter, inverts back to spatial domain, and averages over channels.
- Returns shape `[B, N]`.

> **Why per-frame 2D FFT instead of 3D FFT**: Video-LLaVA uses only 8 sparsely-sampled frames, which is insufficient for the temporal axis of a 3D FFT and causes aliasing artifacts. Per-frame 2D FFT captures spatial high-frequency structure reliably; temporal change is captured implicitly through cross-layer evolution.

`spatial_spectral_score_3d` is retained in the file as a reference alternative (3D Gaussian high-pass over `(T, H, W)`), but is not called by default.

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

## `clse_model.py`

Defines `CLSELlamaModel`, a subclass of `LlamaModel` that overrides `forward()` to insert visual token pruning during the prefill pass.

### Key hyperparameters

| Attribute | Default | Description |
|---|---|---|
| `prune` | `True` | Enable / disable visual token pruning |
| `keep_tokens` | `[136]` | Number of visual tokens to retain at each pruning stage |
| `L_list` | `[0]` | Layer indices at which reference features `Z_L` are recorded |
| `K_list` | `[1]` | Layer indices at which pruning is applied |
| `score_type` | `"clse"` | Token scoring method: `"attn"`, `"clse"`, or `"clse_attn"` |
| `cutoff` | `0.1` | Gaussian high-pass cutoff ratio for spectral filtering |
| `temp` | `0.1` | Temperature for sigmoid normalization of evolution intensity |

`L_list` and `K_list` are full lists supporting multi-stage pruning; the default single-element values configure a single pruning pass.

### Visual token layout

Video-LLaVA encodes 8 frames at 16×16 patch resolution, producing **2048** visual tokens (8 × 16 × 16). These occupy a fixed range in the sequence:

```
image_start = 35
image_end   = 2083   # 35 + 2048
```

### Pruning procedure (executed once per stage at layer `K`)

1. **Record reference features** at layer `L` (`L_list[i]`): snapshot the visual token hidden states as `Z_L`.
2. **Pre-compute attention** at layer `K-1`: run the full decoder layer with `output_attentions=True`, cache the result as `self.last_attention`.
3. **Score tokens** at layer `K` (`K_list[i]`): call `calculate_evolution_score` with `Z_L`, the current visual features, and the cached attention map.
4. **Select top-k tokens**: keep `keep_tokens[i]` highest-scoring tokens, rebuild the sequence by concatenating prefix tokens, selected visual tokens, and text suffix in sorted order.
5. **Trim attention mask and position ids** to match the pruned sequence length.
6. **Update visual boundary**: `current_image_end = image_start + keep_k`.

> **Multi-stage note**: for stage 1, the spatial grid `(8, 16, 16)` is passed to the spectral scorer. For stage 2+, the spatial grid is no longer valid after pruning, so `image_grid_thw=None` is passed and scoring falls back to attention-only.

---

## `llava_llama.py`

One-line change: `LlavaLlamaModel` now inherits from `CLSELlamaModel` instead of the original `LlamaModel`, injecting the CLSE pruning logic transparently into the existing Video-LLaVA inference pipeline.

```python
# before
from .gest_model import GestVLlamaModel
class LlavaLlamaModel(LlavaMetaModel, GestVLlamaModel): ...

# after
from .clse_model import CLSELlamaModel
class LlavaLlamaModel(LlavaMetaModel, CLSELlamaModel): ...
```

---

## Differences from the LLaVA-1.5 CLSE implementation

| | LLaVA-1.5 | Video-LLaVA |
|---|---|---|
| Visual token count | 576 (1 × 24 × 24) | 2048 (8 × 16 × 16) |
| Spectral scoring | 2D FFT over `[H, W]` | Per-frame 2D FFT over `[H, W]` with T merged into batch |
| `image_grid_thw` | 3-tuple `(T, H, W)`, `T=1` | 3-tuple `(T, H, W)`, `T=8` |
| Token count config | `keep_tokens` (absolute) | `keep_tokens` (absolute) |
| KV cache alignment | Explicit slice of `DynamicCache` | Not applied |
| Default `score_type` | `"clse"` | `"clse_attn"` |
| Default pruning stage | layer 0 → 1 | layer 2 → 3 |
