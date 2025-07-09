import torch
import torch.nn as nn

from typing import Type
from dna.modules import TransformerBlock, PatchEmbedding, MultiHeadSelfAttention, MLP


################################################################################
# Vision Transformer (ViT) - Modified with Block Token Encoder
################################################################################
class ViT(nn.Module):
    def __init__(
        self, 
        model_config, 
        attn_module: Type[MultiHeadSelfAttention],
        mlp_module: Type[MLP]
    ):
        super().__init__()
        self.model_config = model_config
        self.depth = model_config.max_path_len
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=model_config.in_channels,
            patch_size=model_config.patch_size,
            embed_dim=model_config.embed_dim,
            image_size=model_config.image_size
        )
        self.num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, model_config.embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(
                attn_module=attn_module,
                mlp_module=mlp_module,
                embed_dim=model_config.embed_dim, 
                num_heads=model_config.num_heads, 
                mlp_ratio=model_config.mlp_ratio
            ) for _ in range(self.depth)]
        )
        self.head = nn.Linear(model_config.embed_dim, model_config.num_classes, bias=False)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialization remains the same"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for id, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 1) Patch embedding and token setup
        x = self.patch_embed(x)        
        
        x = x + self.pos_embed
        
        # 2) Process through maximum path length in the graph
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, attn_mask=None)
            
        # 3) Final classification
        x = x.mean(dim=1)
        logits = self.head(x)
        
        return logits
