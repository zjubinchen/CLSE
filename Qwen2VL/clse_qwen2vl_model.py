import os
from typing import Optional, Tuple, Union, List
from .tools import calculate_evolution_score

class Qwen2VLTextModel(Qwen2VLPreTrainedModel):
    config: Qwen2VLTextConfig

    def __init__(self, config: Qwen2VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)
        
        self.gradient_checkpointing = False

        self.prune = os.getenv("PRUNE", False) # whether to enable visual token pruning
        self.retain_ratio =  float(os.getenv("RETAIN_RATIO", 1.0)) 
        self.last_attention = None           # attention map cached from the layer before pruning
        self.image_grid_thw = None           # visual grid tensor [N_images, 3] with (T, H, W) per image; set externally before forward; H//2, W//2 used for spectral scoring
        self.Z_L = None                      # reference features cached at each L in L_list
        self.score_type = "clse_attn"             # scoring method: "attn", "clse", or "clse_attn"
        self.L_list = [0]                    # layer indices at which reference features are recorded
        self.K_list = [1,10,19]              # layer indices at which pruning is applied
 
        self.has_sliding_layers = "sliding_attention" in getattr(config, "layer_types", [])
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # 3D RoPE position ids: shape [3, B, S] for temporal/height/width
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
            text_position_ids = None
        else:
            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                text_position_ids = position_ids[0]
                position_ids = position_ids[1:]
            elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
                text_position_ids = None
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
                text_position_ids = None
            else:
                text_position_ids = None

        # convert raw attention mask to the 4D causal mask dict expected by decoder layers
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            from transformers.models.qwen2_vl.modeling_qwen2_vl import create_causal_mask, create_sliding_window_causal_mask
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # locate visual token range in the sequence via the visual position mask
        has_visual = False
        img_start, img_end = 0, 0
        visual_mask = kwargs.get("visual_pos_masks", getattr(self, "visual_pos_masks", None))
        if visual_mask is not None:
            current_mask = visual_mask[0] if visual_mask.dim() == 2 else visual_mask
            vis_indices = torch.nonzero(current_mask).squeeze()
            if vis_indices.numel() > 0:
                img_start, img_end = vis_indices[0].item(), vis_indices[-1].item() + 1
                has_visual = True

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # record original visual length for ratio-based pruning target computation
        if has_visual:
            original_visual_len = img_end - img_start

        keep_ratio_list_334 = [0.57,0.36,0.098]  # (0.57*9 + 0.36*9 +0.098*10)/28 = 0.334  
        keep_ratio_list_223 = [0.38,0.24,0.066]
        keep_ratio_list_112 = [0.19,0.12,0.034]
        ratio_dict = {
            0.334 : keep_ratio_list_334,
            0.223 : keep_ratio_list_223,
            0.112 : keep_ratio_list_112
        }
        self.keep_ratios = ratio_dict[self.retain_ratio] if self.retain_ratio in ratio_dict else [self.retain_ratio*r for r in [1.71,1.08,0.294]]


        for layer_idx, decoder_layer in enumerate(self.layers):

            # snapshot reference features at each L in L_list
            if self.prune and layer_idx in self.L_list and has_visual:
                self.Z_L = hidden_states[:, img_start:img_end, :].clone()

            # --- pruning at each K in K_list ---
            if self.prune and layer_idx in self.K_list and has_visual and self.last_attention is not None:
                
                k_idx = self.K_list.index(layer_idx)
                # target token count = original length * retention ratio
                target_keep = max(1, int(original_visual_len * self.keep_ratios[k_idx]))
                current_len = img_end - img_start
                seq_len_before = hidden_states.shape[1]

                # only prune when current visual length exceeds the target
                if current_len > target_keep:

                    # average over heads; take the last text token's attention over image tokens
                    image_attention_score = self.last_attention.mean(dim=1)[0, -1, img_start:img_end]

                    # spectral grid available for stage 1 only; subsequent stages use attention-only scoring
                    if layer_idx == self.K_list[0]:
                        h_new = self.image_grid_thw[0][1].item() // 2
                        w_new = self.image_grid_thw[0][2].item() // 2
                        image_grid = (h_new, w_new)
                    else:
                        image_grid = None
                    
                    # compute per-token importance score
                    evolution_score = calculate_evolution_score(
                        z_L=self.Z_L,
                        z_Lk=hidden_states[:, img_start:img_end, :],
                        attention_weights=image_attention_score.unsqueeze(0),
                        image_grid_thw=image_grid,
                        score_type=self.score_type
                    )

                    # select top-k visual tokens and rebuild the full sequence index
                    top_rank_indices = evolution_score.view(-1).topk(target_keep).indices.sort().values
                    keep_indexs = torch.cat((
                        torch.arange(img_start, device=hidden_states.device),
                        top_rank_indices + img_start,
                        torch.arange(img_end, hidden_states.shape[1], device=hidden_states.device)
                    )).long()

                    hidden_states = hidden_states[:, keep_indexs, :]
                    position_ids = position_ids[..., keep_indexs]
                    if text_position_ids is not None:
                        text_position_ids = text_position_ids[..., keep_indexs]
                    cache_position = cache_position[keep_indexs]

                    # trim causal masks to match the pruned sequence
                    if causal_mask_mapping is not None:
                        for k, v in causal_mask_mapping.items():
                            if v is not None and isinstance(v, torch.Tensor) and v.shape[-1] == seq_len_before:
                                causal_mask_mapping[k] = v.index_select(-2, keep_indexs).index_select(-1, keep_indexs)

                    position_embeddings = self.rotary_emb(hidden_states, position_ids)
                    # if past_key_values is not None:
                    #     for i in range(layer_idx):
                    #         if i < len(past_key_values.layers):
                    #             past_key_values.layers[i].keys = (
                    #                 past_key_values.layers[i].keys.index_select(2, keep_indexs)
                    #             )
                    #             past_key_values.layers[i].values = (
                    #                 past_key_values.layers[i].values.index_select(2, keep_indexs)
                    #         )
                    img_end = img_start + target_keep

            # select the appropriate mask for this layer's attention type
            mask_to_use = causal_mask_mapping[decoder_layer.attention_type] if causal_mask_mapping is not None else None

            # pre-compute attention one layer before each pruning point for attn-based scores
            next_layer_idx = layer_idx + 1
            if self.prune and next_layer_idx in self.K_list and has_visual and hidden_states.shape[1] > 1:
                self.last_attention = self._capture_attention_score(
                    decoder_layer, hidden_states, position_ids, mask_to_use, position_embeddings
                )

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=mask_to_use,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _capture_attention_score(
        self,
        layer,
        hidden_states,
        position_ids,
        attention_mask,
        position_embeddings
    ):
        """Compute attention weights for the last query token without running the full layer forward."""
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb

        with torch.no_grad():
            attn = layer.self_attn

            hidden_norm = layer.input_layernorm(hidden_states)

            query_states = attn.q_proj(hidden_norm)
            key_states = attn.k_proj(hidden_norm)

            bsz, q_len, _ = query_states.size()

            query_states = query_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)

            # apply multimodal rotary position embedding
            cos, sin = position_embeddings
            mrope_section = attn.rope_scaling["mrope_section"]
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, mrope_section
            )

            # broadcast key/value for GQA
            if attn.num_key_value_groups > 1:
                key_states = key_states.repeat_interleave(attn.num_key_value_groups, dim=1)

            # compute attention only for the last query token: [bsz, num_heads, 1, k_len]
            last_query_state = query_states[:, :, -1:, :]
            attn_weights = torch.matmul(last_query_state, key_states.transpose(2, 3)) * attn.scaling

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, -1:, :]

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            return attn_weights.detach()

    # def _capture_full_attention_score(
    #     self, 
    #     layer, 
    #     hidden_states, 
    #     position_ids, 
    #     attention_mask, 
    #     position_embeddings
    # ):
    #     from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
        
    #     with torch.no_grad():
    #         attn = layer.self_attn
            
    #         # 1. Norm (必须有)
    #         hidden_norm = layer.input_layernorm(hidden_states)
            
    #         # 2. Proj
    #         query_states = attn.q_proj(hidden_norm)
    #         key_states = attn.k_proj(hidden_norm)
            
    #         bsz, q_len, _ = query_states.size()
            
    #         # 3. Reshape [batch, seq_len, num_heads, head_dim] -> transpose
    #         query_states = query_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    #         key_states = key_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
            
    #         # 4. 应用 M-RoPE (直接使用传入的 position_embeddings)
    #         # position_embeddings 已经在主循环由 rotary_emb 计算好了 (cos, sin)
    #         cos, sin = position_embeddings
    #         mrope_section = attn.rope_scaling["mrope_section"]
            
    #         # 应用旋转 (注意 Qwen2VL 的特殊处理)
    #         query_states, key_states = apply_multimodal_rotary_pos_emb(
    #             query_states, key_states, cos, sin, mrope_section
    #         )
            
    #         # 5. GQA 广播 (如果是 GQA 模型)
    #         if attn.num_key_value_groups > 1:
    #             key_states = key_states.repeat_interleave(attn.num_key_value_groups, dim=1)
                
    #         # 6. 计算 Score
    #         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attn.scaling
            
    #         # 7. [关键修复] 应用 Mask
    #         # attention_mask 里的值为 0 (保留) 和 -inf (遮蔽)
    #         if attention_mask is not None:
    #             # 确保维度匹配，Qwen2VL mask 通常是 [bs, 1, seq, seq]
    #             attn_weights = attn_weights + attention_mask
                
    #         # 8. Softmax
    #         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
    #         return attn_weights.detach()