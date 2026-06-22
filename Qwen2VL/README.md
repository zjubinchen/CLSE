Below are the modifications made to the original transformers `qwen2_vl` model:

1. put `tools.py` under `transformers/src/transformers/models/qwen2_vl/`

2. replace the `Qwen2VLTextModel` class in `transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py` with the content of `clse_qwen2vl_model.py`

3. in `transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`, insert the following at the beginning of `Qwen2VLForConditionalGeneration.forward()`:
```python
self.model.language_model.visual_pos_masks = (input_ids == 151655)
self.model.language_model.image_grid_thw = image_grid_thw
```

HINT: You could change the K and R hyper-parameters of CLSE in `clse_qwen2vl_model.py` (`retain_ratio`, `K_list`, `L_list`, `score_type`).

4. After finishing the steps ahead, the updated Qwen2-VL directly supports [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repo for more convenient evaluation of CLSE.
