from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:
    def __init__(
            self,
            hidden_size = 768,
            intermediate_size = 12,
            num_hidden_layers = 12,
            num_attention_layers=12,
            num_channels = 3,
            image_size = 224,
            patch_size=16,
            layer_norm_eps = 1e-6,
            attention_dropout = 0.0,
            num_image_tokens: int=None,
            **kwargs
    ):
        super().init()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hiiden_layers = num_hidden_layers
        self.attention_layers = num_attention_layers
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        self.config = config
        self.embed_dims = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dims,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid'   # This indicates no padding is added
        )

        self.num_patches = (self.image_size//self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dims)
        self.register_buffer(
            'position_ids',
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor):
        _, _, height, width = pixel_values.shape  # [Batch_Size, Channels, Height, Width]
        patch_embeds = self.patch_embedding(pixel_values) # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] 
        embeddings = patch_embeds.flatten(2) # [Batch_Size, Embed_Dim, Num_Patches]  Num_patches = Num_Patches_H * Num_Patches_W
        embeddings = embeddings.transpose(1,2) # [Batch_Size, Num_Patches, Embed_Dim]
        embeddings += self.position_embedding(self.position_ids) # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().init()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states) # [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh') # [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc2(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        return hidden_states


class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().init()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = self.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()# hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3))*self.scale)# [Batch_Size, Num_Heads, Num_Patches, Num_Patches]

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention Weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f"{attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)# [Batch_Size, Num_Heads, Num_Patches, Head_Dim]  

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"'attn_output' should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f"{attn_weights.size()}"
            )
        
        attn_output = attn_output.transpose(1,2).contiguous() # Contiguous tensor is represented in memory in a continuous way. Avoids memory overhead in the upcoming reshape step
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class SiglipEncoder(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super.init()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hiiden_layers)]
        )

    def forward(self, inputs_embeds:torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        
        return hidden_states


class  SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().init()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states += residual # [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states += residual
        return hidden_states

