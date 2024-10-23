import torch
import torch.nn as nn
import torch.nn.functional as F
from .ASTUC import *
import src.loralib as lora
from torchinfo import summary

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

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

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out

class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

# MultiLayer Residual Fusion Block
class MLRF(nn.Module):
    def __init__(self, args):
        super(MLRF, self).__init__()
        self.args = args
        self.device = args.device
        self.node_num = args.node_num
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.embed_dim = args.embed_dim
        self.num_nalls = args.num_nalls
        self.supports = args.supports

        self.nalls_spatial = nn.ModuleList()
        self.nalls_temporal = nn.ModuleList()

        self.bn = nn.BatchNorm1d(self.embed_dim)

        for i in range(self.num_nalls):
#            self.nalls_temporal.append(ASTUC(self.embed_dim*self.horizon, self.embed_dim*self.horizon, self.args))
#            self.nalls_spatial.append(ASTUC(self.embed_dim*self.node_num, self.embed_dim*self.node_num, self.args))
            self.nalls_temporal.append(nn.Linear(self.embed_dim*self.horizon, self.embed_dim*self.horizon))
            self.nalls_spatial.append(nn.Linear(self.embed_dim*self.node_num, self.embed_dim*self.node_num))

        self.dropout = nn.Dropout(0.3)

        self.norm = RMSNorm(self.embed_dim, self.device)

    def forward(self, x, label=None):
        B, T, E = x.shape
        N, D = E//self.embed_dim, self.embed_dim
        x = x.reshape(B, T, N, D) 
        for i in range(len(self.nalls_spatial)):
        
            output = self.norm(x)
        
            output = output.reshape(B,T, N*D) # B,T, N*D
            
            output = self.nalls_spatial[i](output).reshape(B, T, N, D)
        
            output = output.transpose(1, 2).reshape(B, N, T*D) # B, N, T*D
            
            output = self.nalls_temporal[i](output).transpose(1,2).reshape(B, T, N, D)

        return output.reshape(B,T,E)

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 device: str,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model).to(device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
    
if __name__ == "__main__":
    model = MLRF()
    summary(model, [64, 12, 207, 3])