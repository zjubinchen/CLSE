# CLSE Integration for LLaVA-1.5

This document describes the modifications made to the original LLaVA-1.5 codebase to integrate **CLSE (Cross-Layer Spectral Evolution)** visual token pruning.

## Modified / Added Files

```
llava/model/language_model/
├── clse_model.py       # new — CLSELlamaModel with pruning logic
├── tools.py            # new — spectral scoring utilities
└── llava_llama.py      # modified — inherit CLSELlamaModel instead of LlamaModel
```

---

## `llava/model/language_model/clse_model.py`

Defines `CLSELlamaModel`, a subclass of `LlamaModel` that overrides `forward()` to insert visual token pruning during the prefill pass.

### Key hyperparameters

| Attribute | Default | Description |
|---|---|---|
| `prune` | `True` | Enable / disable visual token pruning |
| `keep_tokens` | `[192]` | Number of visual tokens to retain at each pruning stage |
| `L_list` | `[0]` | Layer indices at which reference features `Z_L` are recorded |
| `K_list` | `[1]` | Layer indices at which pruning is applied |
| `image_grid_thw` | `(1, 24, 24)` | Visual grid shape `(T, H, W)`; `T=1` for image, `T>1` for video |
| `score_type` | `"clse"` | Token scoring method: `"attn"`, `"clse"`, or `"clse_attn"` |
| `cutoff` | `0.1` | Gaussian high-pass cutoff ratio for spectral filtering |
| `temp` | `0.1` | Temperature for sigmoid normalization of evolution intensity |

### Pruning procedure (executed once per stage at layer `K`)

1. **Record reference features** at layer `L` (`L_list[i]`): snapshot the visual token hidden states as `Z_L`.
2. **Pre-compute attention** at layer `K-1` when `score_type != "clse"`: run the layer with `output_attentions=True` and cache the attention map for use in the next step.
3. **Score tokens** at layer `K` (`K_list[i]`): call `calculate_evolution_score` with `Z_L`, the current visual features, and the cached attention map.
4. **Select top-k tokens**: keep the highest-scoring `keep_tokens[i]` visual tokens, rebuild the sequence by concatenating prefix tokens, selected visual tokens, and text suffix, maintaining sorted order.
5. **Trim attention mask and position ids** to match the pruned sequence length.
6. **Align KV caches** of all preceding layers to the pruned sequence, ensuring consistent KV lengths during the decode phase.

---

## `llava/model/language_model/tools.py`

Provides three scoring functions used by `CLSELlamaModel`.

### `spatial_spectral_score_2d(x, t, h, w, cutoff_ratio)`

Computes a per-token high-frequency energy score via 2D FFT.

- Reshapes `[B, N, C]` → `[B*T, C, H, W]` to apply spatial 2D FFT independently per frame.
- Builds a Gaussian high-pass filter centered at the DC component.
- Applies the filter in the frequency domain, inverts back to spatial domain, and averages over channels.
- Returns shape `[B, T*H*W]`, compatible with both image (`T=1`) and video (`T>1`) inputs.

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

---

## `llava/model/language_model/llava_llama.py`

One-line change: `LlavaLlamaModel` now inherits from `CLSELlamaModel` instead of the original `LlamaModel`, injecting the CLSE pruning logic transparently into the existing LLaVA-1.5 inference pipeline.

```python
# before
class LlavaLlamaModel(LlavaMetaModel, LlamaModel): ...

# after
from .clse_model import CLSELlamaModel
class LlavaLlamaModel(LlavaMetaModel, CLSELlamaModel): ...
```
