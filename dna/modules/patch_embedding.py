import torch.nn as nn
from .dense_blocks import TransformerBlock, MultiHeadSelfAttention, MLP


################################################################################
# Patch Embedding
################################################################################
class PatchEmbedding(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        patch_size=16, 
        embed_dim=768,
        image_size=224, 
        start_node_depth=0, 
        num_heads=1, 
        mlp_ratio=4.0
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.start_node_depth = start_node_depth

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        
        self.encoder = None
        if self.start_node_depth > 0:
            self.encoder = nn.ModuleList()
            for i in range(self.start_node_depth):
                self.encoder.add_module(
                    f'entf_{i}', 
                    TransformerBlock(MultiHeadSelfAttention, MLP, embed_dim, num_heads, mlp_ratio)
                )

    def forward(self, x):
        # x: (B, in_channels, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        if self.encoder:
            for block in self.encoder:
                x = block(x)
        return x
