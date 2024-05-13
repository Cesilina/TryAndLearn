import torch.nn as nn


class DeepseekV2Attention(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        ...
        # 表示向量中应用rope部分的维度
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # 表示向量中不应用rope部分的维度
        self.qk_nope_head_dim = config.qk_nope_head_dim
        # 每一个head的维度应该是两部分只和
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        
        # !下面省略layernorm的定义，实际使用RMSNorm
        # 对query进行压缩，即down-projection
        self.w_dq = nn.Linear(
            self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        # 对压缩后的query映射成高维，即up-projection
        self.w_uq = nn.Linear(
            config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        # 计算压缩后的latent kv以及需要缓存的应用RoPE的k的部分
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias)
        # 这里矩阵计算up-projection后的不应用RoPE的k的部分 和 up-projection后的v的结果
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        
        ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        
        bsz, q_len, _ = hidden_states.size()
        
        # 计算压缩后的Q，再还原成高维
        # [B, q_len, hidden_size]
        # 即[B, q_len, num_head * q_head_dim]
        q = self.w_uq(self.q_a_layernorm(self.w_dq(hidden_states)))
        # [B, num_head, q_len, q_head_dim]
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        # 划分成两部分
        # q_nope表示不需要应用RoPE的
        # q_pe表示需要应用RoPE的
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        
        # 得到当前压缩后的kv
        # [B, q_len, d_c + qk_rope_head_dim]
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.d_c, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        # 包含当前位置可用上下文的长度
        kv_seq_len = k_pe.size(-2)
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        
        # 将当前位置之前的压缩后的kv以及应用过rope的k的部分拼接到前面
        if past_key_value is not None:
            # 得到的应该是
            # compressed_kv: [B, kv_seq_len, d_c]
            # k_pe: [B, 1, kv_seq_len, qk_rope_head_dim]
            compressed_kv, k_pe = past_key_value.update(compressed_kv, k_pe)
        # 计算得到k^C和v^C
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        # k_nope的维度为[B, num_head, kv_seq_len, qk_nope_head_dim]
        # value_states的维度为[B, num_head, kv_seq_len, v_head_dim]
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
        
        key_states = k_pe.new_empty(bsz, self.num_heads, kv_seq_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        

        # [B, num_head, q_len, kv_seq_len]
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )
        ...
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        # [B, num_head, q_len, q_head_dim]
        attn_output = torch.matmul(attn_weights, v)