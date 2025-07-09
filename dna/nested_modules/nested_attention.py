################################################################################
# **DO NOT USE THIS FILE**
# This file contains an experimental implementation of JaggedMultiHeadSelfAttention
# that is not used in the final implementation. The code is kept here for reference.
# Waiting for PyTorch 2.7 or later version to support nested tensors in SDPA properly
################################################################################
import os
import warnings
from typing import Optional

import torch

# Import the base class likely used
from dna.modules import MultiHeadSelfAttention # Assuming this is the intended base

# Try importing xformers
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind
        from xformers.ops.fmha.attn_bias import AttentionBias # Import base type for bias
        XFORMERS_AVAILABLE = True
    else:
        warnings.warn("xFormers is disabled (Attention Module)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention Module)")


################################################################################
# xFormers-based MultiHeadSelfAttention
################################################################################
class NestedMHA(MultiHeadSelfAttention): # Inherit from the original in dag.modules
    """
    Overrides the base MultiHeadSelfAttention to use xFormers memory_efficient_attention
    if available, and accepts an optional attention_bias parameter (e.g., BlockDiagonalMask).
    """
    def forward(self, x: torch.Tensor, attn_bias: Optional[AttentionBias] = None, is_causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = unbind(qkv, 2)

        # Apply xFormers memory efficient attention, not the convention is slightly different.
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, C) # (B, N, H*D) = (B, N, C)

        x = self.out(x)

        return x
