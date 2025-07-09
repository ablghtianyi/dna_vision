import yaml

from typing import Tuple, Dict, Optional
from typing import Any
from types import SimpleNamespace

from torch.nn import Identity

from dna.modules import (
    MLP,
    MultiHeadSelfAttention,
    TransformerBlock,
    PreLNAttnBlock,
    PreLNMLPBlock
)


def load_config(
    config_path: str, 
    overrides: Optional[Dict[str, str]] = None
) -> Tuple[SimpleNamespace, ...]:
    """
    Load and parse configuration file with optional overrides.
    Args:
        config_path (str): Path to YAML configuration file.
        overrides (Optional[Dict[str, str]], optional): Dictionary of configuration overrides using dot notation. Defaults to None.
    Returns:
        Tuple[SimpleNamespace, ...]: Tuple of SimpleNamespace objects representing each section of the configuration.
    """
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    # Apply command-line overrides using dot notation
    if overrides:
        for key_path, value in overrides.items():
            keys = key_path.split('.')
            current = config_dict
            for key in keys[:-1]:
                current = current.setdefault(key, {})
                
            # Only parse strings with YAML, keep other types as-is
            if isinstance(value, str):
                if value.lower() == 'none':
                    final_value = None
                else:
                    final_value = yaml.safe_load(value)
            else:
                final_value = value
            
            current[keys[-1]] = final_value
    
    # Define sections to extract from configuration dictionary
    sections = ['model', 'training', 'data', 'logging']
    
    # Convert each section to SimpleNamespace
    configs = []
    for section in sections:
        section_config = config_dict.get(section, {})
        configs.append(SimpleNamespace(**section_config))
    
    return tuple(configs)


def prepare_trainer_and_namestr(
    model_config: SimpleNamespace
) -> Dict[str, Dict[str, Any]]:
    """Prepare module dictionary based on configuration"""
    
    # Initialize module dictionary
    module_dict = {}
    
    # Add identity module
    module_dict["identity"] = {
        "class": Identity,
        "num": model_config.module_top_k
    }
    
    # Add Transformer module
    module_dict["tf"] = {
        "class": lambda: TransformerBlock(
            MultiHeadSelfAttention, 
            MLP,
            model_config.embed_dim, 
            model_config.num_heads, 
            model_config.mlp_ratio
        ),
        "num": model_config.n_tf
    }

    # Add attention module
    module_dict["attn"] = {
        "class": lambda: PreLNAttnBlock(
            MultiHeadSelfAttention, 
            model_config.embed_dim, 
            model_config.num_heads
        ),
        "num": model_config.n_attn
    }
    
    # Add MLP module
    module_dict["mlp"] = {
        "class": lambda: PreLNMLPBlock(
            MLP, 
            model_config.embed_dim, 
            model_config.mlp_ratio
        ),
        "num": model_config.n_mlp
    }
    return module_dict