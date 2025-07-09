import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from typing import List

################################################################################
# Linear Selector *without* Aux Loss
################################################################################
class LinearRouter(nn.Module):
    """
    Linear router for selecting modules
    
    Args:
        embed_dim (int): Embedding dimension
        n_modules (int): Number of modules to choose from
        module_top_k (int): Number of modules to select per token
        module_names (list): List of module identifiers
        bias_u (float): learning rate for the deepseekv3 bias
        skip_factor (float): control number of tokens that go to identity module
        eps (float): regulate denominators
    
    Returns:
        tuple: (routing logits, updated_mask), seq, (selected_indices, selected_weights)
    """

    def __init__(
        self, 
        model_config,
        n_modules: int, 
        module_top_k: int, 
        module_names: List, 
        bias_u: float, 
        skip_factor: float, 
        eps: float = 1e-6,
        use_bias: bool = False
    ):
        super().__init__()
        self.model_config = model_config
        self.n_modules = n_modules
        self.module_top_k = module_top_k
        self.module_names = module_names
        self.bias_u = bias_u
        self.skip_factor = skip_factor
        self.eps = eps
        
        self.embed_dim = model_config.embed_dim
        self.use_id = model_config.use_id
        self.router = nn.Linear(self.embed_dim, n_modules, bias=False)
        
        self.latest_usage = dict()
        for i in self.module_names:
            self.latest_usage[i] = 0

        # ======== Added for deepseek v3 like moe =========
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.zeros(self.n_modules), requires_grad=self.use_bias)
        
        # Early exit
        self.early_exit = model_config.early_exit
        self.hard_exit = model_config.hard_exit
        self.register_buffer('_override', torch.arange(module_top_k), persistent=True)
        self.in_block_rescale = model_config.in_block_rescale

    @torch.inference_mode()
    def _get_counts(self, selected_indices):
        flattened_indices = selected_indices.view(-1)
        counts = torch.bincount(flattened_indices, minlength=self.n_modules)  # Count occurrence
        for i, module_id in enumerate(self.module_names):
            self.latest_usage[module_id] += counts[i].item()  # Convert to a standard Python integer

    @torch.inference_mode()
    def _get_inst_counts(self, selected_indices):
        flattened_indices = selected_indices.view(-1)
        counts = torch.bincount(flattened_indices, minlength=self.n_modules)
        return counts  # Return instantaneous count
    
    def reset_counts(self):
        for i, module_id in enumerate(self.module_names):
            self.latest_usage[module_id] = 0  # Convert to a standard Python integer

    def get_usage_stats(self):
        return self.latest_usage
    
    @torch.inference_mode()
    def _sync_bias(self):
        dist.all_reduce(self.bias.data, op=dist.ReduceOp.AVG)

    @torch.inference_mode()
    def _override_mask(self, module_mask, selected_indices):
        non_id_mask = ~((selected_indices < self.module_top_k).sum(-1) > 0)  # for all choices that are not id
        module_mask = module_mask & non_id_mask  # update the mask
        return module_mask
    
    def forward(
        self, 
        seq, 
        *,
        module_mask=None, 
        module_temperature: float = None, 
    ):
        """
        Args:
            seq (torch.Tensor): Input sequence tensor of shape (B, N, E)
            module_embeddings (None): for API purpose
            module_mask (torch.Tensor, optional): Mask tensor to prevent selection of previously selected modules
            module_temperature (float, optional): Temperature for sampling, used only during inference
            
        Returns:
            tuple: (routing logits, updated_mask), seq, (selected_indices, selected_weights)
        """
        B, N, E = seq.shape

        # Initialize mask
        if (module_mask is None) and ((self.early_exit is True) or (self.hard_exit is True)):
            module_mask = torch.ones((B, N), device=seq.device, dtype=torch.bool)  # All tokens pass initially
        
        # Get routing logits
        routing_logits = self.router(seq)  # (B, N, n_modules)
        
        # Selection logic
        if module_temperature is None:
            # Hard top-k selection
            routing_probs = F.softmax(routing_logits, dim=-1)
            adjusted_logits = routing_probs + self.bias.unsqueeze(0).unsqueeze(0)
            # select indices based on adjusted logits
            _, selected_indices = torch.topk(adjusted_logits , k=self.module_top_k, dim=-1)            
        else:
            # Temperature-based sampling
            routing_probs = F.softmax(routing_logits / module_temperature, dim=-1)
            adjusted_logits = routing_probs + self.bias.unsqueeze(0).unsqueeze(0)

            selected_indices = torch.multinomial(
                adjusted_logits.view(-1, self.n_modules), 
                self.module_top_k, 
                replacement=False
            ).view(B, N, self.module_top_k)
        
        # Update mask and selected indices
        if self.hard_exit is True:
            module_mask = self._override_mask(module_mask, selected_indices)
            override_mask = ~module_mask.unsqueeze(-1).expand(-1, -1, self.module_top_k)
            selected_indices[override_mask] = self._override.expand_as(selected_indices)[override_mask]
            module_mask = None  # Don't propagate
        elif self.early_exit is True:
            module_mask = self._override_mask(module_mask, selected_indices)
            override_mask = ~module_mask.unsqueeze(-1).expand(-1, -1, self.module_top_k)
            selected_indices[override_mask] = self._override.expand_as(selected_indices)[override_mask]

        # use the selected weights
        if self.use_bias:
            selected_weights = torch.gather(adjusted_logits, -1, selected_indices)
        else:
            selected_weights = torch.gather(routing_probs, -1, selected_indices)
        
        if self.in_block_rescale is False:
            selected_weights = selected_weights / (selected_weights.sum(dim=-1, keepdim=True) + self.eps)

        if self.training and self.bias_u > 0.0 and self.use_id:
            counts_float = self._get_inst_counts(selected_indices).float()
            delta_bias = ((self.skip_factor / self.module_top_k) * counts_float.sum() - counts_float).sign()
            self.bias.data[:self.module_top_k] += self.bias_u * delta_bias[:self.module_top_k]
            self._sync_bias()
        else:
            self._get_counts(selected_indices)
        
        
        return (routing_logits, selected_indices), seq, (module_mask, selected_weights)
