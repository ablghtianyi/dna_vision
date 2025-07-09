import torch.optim as optim
from torch.nn import Module

from typing import Optional, Dict, Callable
from types import SimpleNamespace

def _low_lr_for_non_matrix(model, base_lr, base_wd):
    """
    Create a dictionary of parameters with lower learning rate for non-matrix parameters.
    
    Args:
    model (Module): The model to create the parameter dictionary for.
    
    Returns:
    Dict: A dictionary of parameters with lower learning rate for non-matrix parameters.
    """
    low_lr_params = []
    router_params = []
    normal_lr_params = []
    for id, param in model.named_parameters():
        if param.requires_grad is True:
            if len(param.shape) < 2 and 'pos' not in id and 'tokens' not in id:  # if the parameter is not a matrix and not embedding or module tokens
                print(id)
                low_lr_params.append(param)
            else:
                normal_lr_params.append(param)
    return [
        {'params': low_lr_params, 'lr': 0.1 * base_lr, 'weight_decay': base_wd},  # set a lower learning rate for non-matrix parameters
        {'params': normal_lr_params, 'lr': base_lr, 'weight_decay': base_wd}
    ]

def setup_optimizer(
    model: Module, 
    training_config: SimpleNamespace,
    customized_param_func: Optional[Callable] = _low_lr_for_non_matrix
) -> optim.Optimizer:
    """Create optimizer."""
    if customized_param_func:
        customized_param_dict = customized_param_func(
            model, 
            base_lr=training_config.learning_rate,
            base_wd=training_config.weight_decay,
        )
        optimizer = optim.AdamW(
            customized_param_dict,
            betas=(0.9, 0.99),
            fused=True
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(0.9, 0.99),
            fused=True
        )
    return optimizer
