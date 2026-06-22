Below are the modifications made to the original [LLaVA](https://github.com/haotian-liu/LLaVA.git) repo:

1. put `clse_model.py` and `tools.py` under `llava/model/language_model/`

2. replace `LlavaLlamaModel` from `llava/model/language_model/llava_llama.py` with following code (inheriting `CLSELlamaModel` instead of `FastVLlamaModel`):
```python
from .clse_model import CLSELlamaModel
class LlavaLlamaModel(LlavaMetaModel, CLSELlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
```

3. Go to your `transformers/src/transformers/models/llama/modeling_llama.py` in the conda envs, change **all three** 
```python
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
```
from different attention implementation, to

```python
cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)
```

4. After finishing the steps ahead, the updated LLaVA directly supports [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repo for more convinent evaluation of CLSE.


