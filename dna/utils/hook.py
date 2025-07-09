import torch
import torch.nn.functional as F
from xformers.ops import unbind


class Hook:
    def __init__(self, module, if_prob=False):
        self.hook = module.register_forward_hook(self._hook_fn)
        self.module = module  # Store the module
        self.if_prob = if_prob

    @torch.inference_mode()
    def _hook_fn(self, module, input, output):
        self.selected_modules = output[0][1].data  # Access selected indices
        if self.if_prob:
            self.raw_probs = output[-1][-1].detach().data
            bias_expanded = self.module.bias.unsqueeze(0).unsqueeze(0).expand(1, self.raw_probs.size(1), self.module.bias.size(-1))
            self.probs = self.raw_probs + torch.gather(bias_expanded, -1, self.selected_modules)

    def close(self):
        self.hook.remove()

    def get_usage_stats(self):
        """Retrieve usage statistics from the hooked module."""
        return self.module.get_usage_stats()  # Use the stored module


class OverrideHook(Hook):
    def __init__(self, module):
        super().__init__(module)

    @torch.inference_mode()
    def _hook_fn(self, module, input, output):
        with torch.no_grad():
            output[0][1].data[:, :, 1:] = 0


class AttnHook(Hook):
    def __init__(self, module):
        super().__init__(module)
        self.attn_scores = []

    @torch.inference_mode()
    def _hook_fn(self, module, input, output):
        x = input[0]
        # attn_bias = input[1]
        with torch.no_grad():
            B, N, C = x.shape
            qkv = module.to_qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
            q, k, v = unbind(qkv, 2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            attn_weight = (q @ k.transpose(-2, -1) / q.size(-1)**0.5)
            attn_score = torch.softmax(attn_weight, dim=-1)
            
            # Record attention score
            self.attn_scores.append(attn_score.detach().cpu())

class RouterHook(Hook):
    def __init__(self, module):
        super().__init__(module)
        self.hh_T = None
        self.rhhr_T = None
        self.step = 0

    @torch.inference_mode()
    def _hook_fn(self, module, input, output):
        with torch.no_grad():
            B, T, D = input[0].shape
            _, _, Dp = output[0][0].shape
            if self.hh_T is None:
                self.hh_T = torch.zeros(D, D, device=input[0].device)
                self.hh_T += torch.einsum('btd,bth->dh', input[0], input[0]) / (B*T)  # HH^T
            else:
                self.hh_T += torch.einsum('btd,bth->dh', input[0], input[0]) / (B*T)
            
            if self.rhhr_T is None:
                self.rhhr_T = torch.zeros(Dp, Dp, device=input[0].device)
                self.rhhr_T += torch.einsum('btd,bth->dh', output[0][0], output[0][0]) / (B*T)  # RHHR^T
            else:
                self.rhhr_T += torch.einsum('btd,bth->dh', output[0][0], output[0][0]) / (B*T)  # RHHR^T

            self.step += 1

    @torch.inference_mode()
    def _reset(self):
        self.hh_T = None
        self.rhhr_T = None
        self.step = 0

    @torch.inference_mode()
    def readout(self):
        if self.step > 0:
            return self.hh_T / self.step, self.rhhr_T / self.step
        self._reset()
            
class SteerHook(Hook):
    def __init__(self, module, module_top_k):
        super().__init__(module)
        self.layer_decision = None
        self.module_top_k = module_top_k

    @torch.inference_mode()
    def _hook_fn(self, module, input, output):
        if self.layer_decision is not None:
            output[0][1].data[:, :, :] = self.layer_decision
        
            bias = module.bias.unsqueeze(0).unsqueeze(0)
            prob = F.softmax(output[0][0], dim=-1)
            adjusted_logits = prob + bias
            
            output_list = list(output)  # [(output_0), *, (output_2)]
            output_2 = list(output[2])  # [*, *]
            output_2[1] = torch.gather(adjusted_logits, -1, output[0][1])  # Overwrite the selected weights based on the choice
            output_list[2] = tuple(output_2)
            output = tuple(output)

    def update_choice(self, choice):
        self.external_choice = choice
        self.routing_probs = None