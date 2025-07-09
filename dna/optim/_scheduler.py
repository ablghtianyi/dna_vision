import math
import torch

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from types import SimpleNamespace

def setup_scheduler(optimizer: Optimizer, 
                    training_config: SimpleNamespace, 
                    train_loader: DataLoader,
                    init_lr: float = 1e-6,
                    ending_lr: float = 1e-8,) -> object:
    """Create learning rate scheduler."""
    num_training_steps = training_config.num_epochs * len(train_loader)
    num_warmup_steps = training_config.warmup_epochs * len(train_loader)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        init_lr=init_lr,
        ending_lr=ending_lr,
        decay=training_config.__dict__.get('decay', True)
    )
    return scheduler


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, init_lr=1e-6, ending_lr=1e-8, decay=True):
    initial_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def lr_lambda(current_step):
        base_lr = initial_lrs[0]  # Assumes a single learning rate group
        
        if current_step < num_warmup_steps:
            warmup_fraction = current_step / max(1, num_warmup_steps)
            lr = init_lr + (base_lr - init_lr) * warmup_fraction
            return lr / base_lr
        else:
            progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            progress = min(progress, 1.0)  # Ensure progress doesn't exceed 1.0
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = ending_lr + (base_lr - ending_lr) * cosine_factor
            return lr / base_lr
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)