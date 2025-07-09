from types import SimpleNamespace

import torch.nn as nn

from dna.modules import (
    MLP,
    MultiHeadSelfAttention,
)


def build_model(model_config: SimpleNamespace) -> nn.Module:

    """Create and initialize the model based on the specified model type."""
    model_type = getattr(model_config, 'model_type', 'vit')  # Default to 'linear' if not specified
    model_type = model_type.lower()

    if "dna" in model_type:
        from dna.nested_modules import NestedMHA, NestedModuleTokenizer
        from .dna_base import DNA, RouterStream
        from dna.routers import LinearRouter
        
        
        match model_type:
            case "dna_linear_nested_tf":
                RouterCls = LinearRouter
                AttnCls = NestedMHA
                MlpCls = MLP
                module_tokenizer = NestedModuleTokenizer(model_config)
            case _:
                raise ValueError(f"Unknown model type: {model_type}.")
        
        
        # Build the collection of modules based on the config
        router_collection = RouterStream(
            model_config=model_config, 
            module_names=module_tokenizer.module_names, 
            router_class=RouterCls, 
            use_bias=model_config.use_bias
    )
        model = DNA(
            model_config, 
            router_collection,
            module_tokenizer,
        )
        
    else:
        from .vit import ViT

        match model_type:
            case "vit":
                AttnCls = MultiHeadSelfAttention
                MlpCls = MLP
            case _:
                raise ValueError(f"Unknown model type: {model_type}.")
        
        model = ViT(model_config, attn_module=AttnCls, mlp_module=MlpCls)

    print(f"Using model type: {model_type}")
    
    return model