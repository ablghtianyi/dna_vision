from types import SimpleNamespace
import os
import warnings
from typing import Tuple, Dict

import torch
import torch.nn as nn

from dna.modules import (
    MLP,
    PreLNMLPBlock
)

from .nested_blocks import NestedPreLNAttnBlock, NestedTransformerBlock
from .nested_attention import NestedMHA

# Try importing xformers
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask
        XFORMERS_AVAILABLE = True
    else:
        warnings.warn("xFormers is disabled (Nested Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Nested Attention)")


class NestedModuleTokenizer(nn.Module):
    def __init__(self, model_config: SimpleNamespace):
        super().__init__()
        self.module_types = []
        self.module_names = []
        self.modules_list = []
        
        # Add identity module
        if hasattr(model_config, 'module_top_k') and model_config.module_top_k > 0 and model_config.use_id:
            for i in range(model_config.module_top_k):
                module_name = f"identity_{i}"
                module = nn.Identity()
                self.add_module(module_name, module)
                self.module_types.append("identity")
                self.module_names.append(module_name)
                self.modules_list.append(module)
        
        # Add Transformer module
        if hasattr(model_config, 'n_tf') and model_config.n_tf > 0:
            for i in range(model_config.n_tf):
                module_name = f"tf_{i}"
                module = NestedTransformerBlock(NestedMHA, MLP,
                                          model_config.embed_dim, model_config.num_heads, model_config.mlp_ratio)
                self.add_module(module_name, module)
                self.module_types.append("tf")
                self.module_names.append(module_name)
                self.modules_list.append(module)
            
        # Add attention module
        if hasattr(model_config, 'n_attn') and model_config.n_attn > 0:
            for i in range(model_config.n_attn):
                module_name = f"attn_{i}"
                module = NestedPreLNAttnBlock(NestedMHA, 
                                        model_config.embed_dim, model_config.num_heads)
                self.add_module(module_name, module)
                self.module_types.append("attn")
                self.module_names.append(module_name)
                self.modules_list.append(module)
            
        # Add MLP module
        if hasattr(model_config, 'n_mlp') and model_config.n_mlp > 0:
            for i in range(model_config.n_mlp):
                module_name = f"mlp_{i}"
                module = PreLNMLPBlock(MLP, 
                                       model_config.embed_dim, model_config.mlp_ratio)
                self.add_module(module_name, module)
                self.module_types.append("mlp")
                self.module_names.append(module_name)
                self.modules_list.append(module)
        
        self.modules_list = nn.ModuleList(self.modules_list)
        self.in_block_rescale = model_config.in_block_rescale
        self.module_top_k = model_config.module_top_k
        # Instance cache for attention biases
        self.attn_bias_cache: Dict[Tuple[int, ...], BlockDiagonalMask] = {} # Cache key is now tuple of seqlens

    def __len__(self):
        return len(self.modules_list)

    def __iter__(self):
        return zip(self.module_names, self.modules_list)
    
    @staticmethod
    def _nested_attn_like_forward(
        block: nn.Module,
        module_input: torch.Tensor,
        module_batch_indices: torch.Tensor,
        selected_weights: torch.Tensor = None,
    ):
        unique_batches, counts = torch.unique_consecutive(module_batch_indices, return_counts=True)
        seqlens = counts.tolist()

        attn_bias = BlockDiagonalMask.from_seqlens(seqlens)    
        module_output = block(module_input, selected_weights=selected_weights, attn_bias=attn_bias)

        return module_output.squeeze(0)

    @staticmethod
    def _mlp_like_forward(
        block: nn.Module, 
        module_input: torch.Tensor,
        selected_weights: torch.Tensor = None,
    ):
        """Process input through MLP like blocks without token exchange."""
        module_output = block(module_input, selected_weights=selected_weights)
        return module_output

    @staticmethod
    def _unknown_forward(module_input: torch.Tensor):
        """
        Process input through an unknown block (identity function).
        """
        return module_input

    def forward(
        self, 
        x: torch.Tensor, 
        selected_indices: torch.Tensor, 
        selected_weights: torch.Tensor,
    ):
        B, N, E = x.shape
        device = x.device

        # Flatten and expand inputs
        x = x.contiguous().view(-1, E)  # (B*N, E)
        selected_indices = selected_indices.view(-1, self.module_top_k)  # (B*N, topk)
        selected_weights = self.module_top_k * selected_weights.view(-1, self.module_top_k)  # (B*N, topk), assume skip connections are inside each block
        
        # Track batches
        batch_indices = torch.arange(B, device=device)[:, None, None].expand(B, N, self.module_top_k).reshape(-1, self.module_top_k)  # (B*N, top_k)
        
        # Process through all modules
        y = torch.zeros_like(x)
        for module_idx, block in enumerate(self.modules_list):
            mask = (selected_indices == module_idx)
            if not mask.any():
                continue
            
            idx, top = torch.where(mask)  # (selected,), (selected,)
            match block.__class__.__name__:
                case "NestedPreLNAttnBlock" | "NestedTransformerBlock":
                    module_input = x[idx].unsqueeze(0)  # (1, selected, E)
                    module_batch_indices = batch_indices[idx, top]  # (selected,)
                    
                    module_output = self._nested_attn_like_forward(
                        block,
                        module_input,
                        module_batch_indices,
                        selected_weights=selected_weights[idx, top, None],
                    )
                case "PreLNMLPBlock":
                    # MLP-like blocks process tokens independently, no complex batching needed here.
                    module_output = self._mlp_like_forward(
                        block,
                        x[idx],
                        selected_weights=selected_weights[idx, top, None] if self.in_block_rescale else None
                    )
                case _:
                    module_output = self._unknown_forward(x[idx])

            # Apply weights and combine (if not done internally by _mlp_like_forward with in_block_rescale)
            if self.in_block_rescale:
                y[idx] += module_output
            else:
                y[idx] += module_output * selected_weights[idx, top, None]
        # The final division by self.module_top_k normalizes the contributions
        # if the weights represent a distribution summing to module_top_k.
        return y.view(B, N, E) / self.module_top_k
