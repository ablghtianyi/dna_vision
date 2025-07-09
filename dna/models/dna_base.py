import copy

from typing import Type, List
from types import SimpleNamespace

import torch
import torch.nn as nn

from dna.modules import PatchEmbedding
from dna.nested_modules import NestedModuleTokenizer
from dna.routers import LinearRouter

################################################################################
# RouterStream - Collection of Routers
################################################################################
class RouterStream(nn.Module):
    def __init__(
        self, 
        model_config: SimpleNamespace, 
        module_names: List[str],
        router_class: Type[LinearRouter],
        eps: float = 1e-6,
        use_bias: bool = False
    ):
        super().__init__()
        self.model_config = model_config
        self.module_names = module_names
        self.n_modules = len(self.module_names)
        self.module_top_k = model_config.module_top_k
        self.max_path_len = model_config.max_path_len
        
        # Load penalization hyperpms
        self.use_bias = use_bias
        self.bias_u = model_config.bias_u
        self.skip_factor = model_config.skip_factor
        
        # Encoder for attn router
        self.module_token_encoder = None
            
        # step-wise selectors
        self.routers = nn.ModuleList([
            router_class(
                model_config=model_config,
                n_modules=self.n_modules,
                module_top_k=self.module_top_k, 
                module_names=self.module_names, 
                bias_u=self.bias_u,
                skip_factor=self.skip_factor,
                eps=eps,
                use_bias=self.use_bias
                )
            for _ in range(self.max_path_len)
        ])
    
    def __len__(self):
        return len(self.routers)

    def __iter__(self):
        return iter(self.routers)

################################################################################
# Vision Transformer (ViT) - Modified with Block Token Encoder
################################################################################
class DNA(nn.Module):
    def __init__(
        self, 
        model_config, 
        router_collection: RouterStream,
        module_tokenizer: NestedModuleTokenizer, 
    ):
        super().__init__()
        # Load config
        self.model_config = model_config
        self.module_top_k = model_config.module_top_k
        self.max_path_len = model_config.max_path_len
        self.module_temperature = model_config.module_temperature

        # Load modules
        self.module_tokenizer = module_tokenizer
        self.n_modules = len(module_tokenizer)  # module_top_k included
        self.token_paths = None

        # Build router collection
        self.router_collection = router_collection
        
        # Load penalization hyperpms
        self.bias_u = model_config.bias_u
        self.skip_factor = model_config.skip_factor
        
        # Initialize patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=model_config.in_channels,
            patch_size=model_config.patch_size,
            embed_dim=model_config.embed_dim,
            image_size=model_config.image_size,
            start_node_depth=model_config.start_node_depth, 
            num_heads=model_config.num_heads, 
            mlp_ratio=model_config.mlp_ratio
        )
        self.num_patches = self.patch_embed.num_patches
        
        # Initialize module tokens and positional embedding
        self.module_tokens = None
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, model_config.embed_dim))
        
        # Final norm + classification head
        self.head = nn.Linear(model_config.embed_dim, model_config.num_classes, bias=False)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialization remains the same"""
        if self.module_tokens is not None:
            nn.init.trunc_normal_(self.module_tokens, std=0.02)
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
        
    def get_usage_stats(self):
        """Get statistics on module usage"""
        stats_list = []
        for router in self.router_collection:
            stats_list.append(copy.deepcopy(router.get_usage_stats()))
            router.reset_counts()
        return stats_list

    @torch.inference_mode()
    def sync_bias(self):
        for router in self.router_collection:
            router._sync_bias()

    def forward(self, x):
        # 1) Patch embedding and token setup
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # 2) Process through maximum path length in the graph
        module_mask = None  # Can be used to add constraint to the path of the token
        for router_idx, router in enumerate(self.router_collection):
            # Extra returns for analysis
            (router_logits, selected_indices), x, (module_mask, selected_weights) = router(
                x, 
                module_mask=module_mask, 
                module_temperature=self.module_temperature,
            )
            
            x = self.module_tokenizer(x, selected_indices, selected_weights)
            
        # 3) Final classification
        x = x.mean(dim=1)
        logits = self.head(x)
        
        return logits