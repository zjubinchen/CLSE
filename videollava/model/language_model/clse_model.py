import torch
from typing import List, Optional, Union
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
    Cache,
    DynamicCache
)
from transformers.models.llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

from .tools import calculate_evolution_score


class CLSELlamaModel(LlamaModel):
    """
    LlamaModel with visual token pruning via CLSE (Cross-Layer Spectral Evolution).
    Supports score_type: "attn", "clse", "clse_attn".
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.last_attention = None       # attention map cached from the layer before pruning
        self.Z_L = None                  # reference features cached at each L in L_list
        self.cutoff = 0.1                # high-pass cutoff ratio for the Gaussian spectral filter
        self.temp = 0.1                  # temperature for sigmoid normalization of evolution intensity
        self.prune = os.getenv("PRUNE", False)    # whether to enable visual token pruning
        self.keep_token = [int(os.getenv("KEEP_TOKEN", 2048))]     # number of visual tokens to retain at each pruning stage
        self.L_list = [2]                # layer indices at which reference features are recorded
        self.K_list = [3]                # layer indices at which pruning is applied
        self.score_type = "clse_attn"    # scoring method: "attn", "clse", or "clse_attn"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPast, tuple]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # use actual sequence length (may differ from seq_length after multimodal embedding)
        actual_seq_length = inputs_embeds.shape[1]

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, (batch_size, actual_seq_length), inputs_embeds, past_key_values_length
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, actual_seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # visual token range in the sequence
        image_start = 35
        current_image_end = 2083  # image_start + 8 * 16 * 16
        has_visual = hidden_states.shape[1] >= current_image_end

        # normalize keep_tokens to a list aligned with K_list
        if isinstance(self.keep_tokens, int):
            keep_tokens_list = [self.keep_tokens] * len(self.K_list)
        else:
            keep_tokens_list = list(self.keep_tokens)
            if len(keep_tokens_list) < len(self.K_list):
                keep_tokens_list += [keep_tokens_list[-1]] * (len(self.K_list) - len(keep_tokens_list))

        for decoder_layer in self.layers:
            current_layer_idx = decoder_layer.self_attn.layer_idx

            # snapshot reference features at each L in L_list
            if self.prune and has_visual and current_layer_idx in self.L_list:
                self.Z_L = hidden_states[:, image_start:current_image_end, :]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # --- pruning at each K in K_list ---
            if self.prune and current_layer_idx in self.K_list and seq_length > 1 and has_visual:
                device = hidden_states.device
                k_idx = self.K_list.index(current_layer_idx)
                current_existing_len = current_image_end - image_start
                keep_k = min(keep_tokens_list[k_idx], current_existing_len)

                image_attention_score = self.last_attention.mean(dim=1)[0][-1][image_start:current_image_end]
                # spectral grid for stage 1 only; subsequent stages use attention-only
                image_grid_thw = (8, 16, 16) if current_layer_idx == self.K_list[0] else None

                evolution_score = calculate_evolution_score(
                    self.Z_L,
                    hidden_states[:, image_start:current_image_end, :],
                    image_attention_score.unsqueeze(0),
                    image_grid_thw,
                    self.cutoff,
                    self.temp,
                    self.score_type
                )

                # select top-k visual tokens and rebuild the full sequence index
                top_local = evolution_score.topk(keep_k, dim=-1).indices.squeeze(0).sort().values  # [keep_k]
                keep_indexs = torch.cat((
                    torch.arange(image_start, device=device),
                    top_local + image_start,
                    torch.arange(current_image_end, hidden_states.shape[1], device=device)
                )).long()

                hidden_states = hidden_states[:, keep_indexs, :]
                position_ids = position_ids[:, keep_indexs]

                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, :hidden_states.shape[1], :hidden_states.shape[1]]

                current_image_end = image_start + keep_k

            # --- pre-compute attention one layer before each pruning point ---
            last_attn = (self.prune and current_layer_idx in [k - 1 for k in self.K_list]
                         and seq_length > 1 and has_visual)
            if last_attn:
                seq_length_temp = hidden_states.size(1)
                attention_mask_temp = torch.ones(1, seq_length_temp, device=hidden_states.device)
                attention_mask_temp = _prepare_4d_causal_attention_mask(
                    attention_mask_temp, (1, seq_length_temp), hidden_states, past_key_values_length
                )
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_temp,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=True,
                    use_cache=use_cache,
                )
                self.last_attention = layer_outputs[1]
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions or last_attn else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
