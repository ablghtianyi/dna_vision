from typing import Union, Type
import torch.nn as nn

from .nested_attention import NestedMHA
from dna.modules import MLP

################################################################################
# Pre-LN Attn only Block
################################################################################
class NestedPreLNAttnBlock(nn.Module):
    def __init__(
        self, 
        attn_module: Type[Union[NestedMHA]], 
        embed_dim: int, 
        num_heads: int
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, bias=False)
        self.attn = attn_module(embed_dim, num_heads)

    def forward(self, x, attn_bias=None, selected_weights=None, is_causal=False):
        """
        Args:
            x: Input tensor (B, N, E) - Batch, Sequence Length, Embedding Dimension
            attn_bias: Optional attention bias (e.g., BlockDiagonalMask from xformers) for attention calculation.
            selected_weights: Optional weights for scaling the output.
            is_causal: Boolean indicating if the attention should be causal.

        Returns:
            Output tensor (B, N, E)
        """
        # Attention
        attn_in = self.norm1(x)
        # Pass attn_bias to the attention module
        attn_out = self.attn(attn_in, attn_bias=attn_bias, is_causal=is_causal)

        if selected_weights is not None:
            attn_out = selected_weights * attn_out
        return x + attn_out


################################################################################
# Transformer Block
################################################################################
class NestedTransformerBlock(nn.Module):
    def __init__(
        self, 
        attn_module: Type[NestedMHA], 
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

    def forward(self, x, attn_bias=None, selected_weights=None, is_causal=False):
        """
        Args:
            x: Input tensor (B, N, E) - Batch, Sequence Length, Embedding Dimension
            attn_bias: Optional attention bias (e.g., BlockDiagonalMask from xformers) for attention calculation.
            selected_weights: Optional weights for scaling the output.
            is_causal: Boolean indicating if the attention should be causal.

        Returns:
            Output tensor (B, N, E)
        """
        # Attention
        attn_in = self.norm1(x)
        # Pass attn_bias to the attention module
        attn_out = self.attn(attn_in, attn_bias=attn_bias, is_causal=is_causal)
        x = x + attn_out

        # MLP
        mlp_in = self.norm2(x)
        mlp_out = self.mlp(mlp_in)
        
        if selected_weights is not None:
            mlp_out = selected_weights * mlp_out
        return x + mlp_out
