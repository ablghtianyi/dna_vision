from typing import Type
import torch.nn as nn

from .mlp import MLP
from .attention import MultiHeadSelfAttention

################################################################################
# Pre-LN Attn only Block
################################################################################
class PreLNAttnBlock(nn.Module):
    def __init__(
        self, 
        attn_module: Type[MultiHeadSelfAttention], 
        embed_dim: int, 
        num_heads: int
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, bias=False)
        self.attn = attn_module(embed_dim, num_heads)
        
    def forward(self, x, attn_mask=None, selected_weights=None):
        """
        Args:
            x: Input tensor (B, N, E) - Batch, Sequence Length, Embedding Dimension
            attn_mask: Optional attention mask (B, N, N) or (N, N)

        Returns:
            Output tensor (B, N, E)
        """
        # Attention
        attn_in = self.norm1(x)
        attn_out = self.attn(attn_in, attn_mask=attn_mask)
        
        if selected_weights is not None:
            attn_out = selected_weights * attn_out
        return x + attn_out


################################################################################
# Pre-LN MLP only Block
################################################################################
class PreLNMLPBlock(nn.Module):
    def __init__(
        self, 
        mlp_module: Type[MLP], 
        embed_dim: int, 
        mlp_ratio: float = 4.0  # Processed by mlp module later while using LLaMAMLP
    ):
        super().__init__()
        self.norm2 = nn.LayerNorm(embed_dim, bias=False)
        self.mlp = mlp_module(embed_dim, mlp_ratio)
        
    def forward(self, x, selected_weights=None):
        """
        Args:
            x: Input tensor (B, N, E) - Batch, Sequence Length, Embedding Dimension
        
        Returns:
            Output tensor (B, N, E)
        """
        # MLP
        mlp_in = self.norm2(x)
        mlp_out = self.mlp(mlp_in)

        if selected_weights is not None:
            mlp_out = selected_weights * mlp_out
        return x + mlp_out


################################################################################
# Transformer Block
################################################################################
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        attn_module: Type[MultiHeadSelfAttention], 
        mlp_module: Type[MLP], 
        embed_dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, bias=False)
        self.norm2 = nn.LayerNorm(embed_dim, bias=False)
        self.attn = attn_module(embed_dim, num_heads)
        self.mlp = mlp_module(embed_dim, mlp_ratio=mlp_ratio)

    def forward(self, x, attn_mask=None, selected_weights=None):
        """
        Args:
            x: Input tensor (B, N, E) - Batch, Sequence Length, Embedding Dimension
            attn_mask: Optional attention mask (B, N, N) or (N, N)

        Returns:
            Output tensor (B, N, E)
        """
        # Attention
        attn_in = self.norm1(x)
        attn_out = self.attn(attn_in, attn_mask=attn_mask)
        x = x + attn_out

        # MLP
        mlp_in = self.norm2(x)
        mlp_out = self.mlp(mlp_in)
        
        if selected_weights is not None:
            mlp_out = selected_weights * mlp_out
        return x + mlp_out
    