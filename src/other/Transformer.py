import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    '''Layer Normalization Module'''
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out
    
class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.linear(model_dim, model_dim)
        self.FC_K = nn.linear(model_dim, model_dim)
        self.Fc_V = nn.linear(model_dim, model_dim)

        self.out_proj = nn.linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1,-2) # (num_heads * batch_size, ..., head_dim, src_length)
        attn_score = (query @ key) / self.head_dim**0.5 # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril() # lower triangular part of the matrix
            attn_score.masked_fill_(~mask,-torch.inf) # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1) # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim) 
        out = self.out_proj(out)

        return out
    