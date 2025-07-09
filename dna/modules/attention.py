import torch.nn as nn
import torch.nn.functional as F

################################################################################
# MultiHeadSelfAttention
################################################################################
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.to_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, attn_mask=None):
        B, N, E = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, E)
        return self.out(attn_output)