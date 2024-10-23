import torch
import torch.nn as nn
import torch.nn.functional as F
from .MLRF import *
from .MLP import *
from .ASTUC import *
from src.base.basemodel import BaseModel
from src.base.engine import BaseEngine
import random

class STLoRA(nn.Module):
    def __init__(self, backbone, args):
        super(STLoRA, self).__init__()
        self.args = args
        self.device = args.device
        self.node_num = args.node_num
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.seq_length = args.seq_length
        self.horizon = args.horizon
        self.num_nalls = args.num_nalls
        self.num_blocks = args.num_mlrfs
        self.backbone = backbone
        self.supports = args.supports
        self.dropout = nn.Dropout(0.3)
        self.lr = args.lrate
        self.weight_decay = args.wdecay
        self.embed_dim = args.embed_dim
        self.blocks = nn.ModuleList()

        self.stu_emb = nn.Parameter(torch.empty(self.seq_length, self.node_num, self.embed_dim))
        nn.init.xavier_uniform_(self.stu_emb)

        self.gate = nn.Linear(self.input_dim + self.output_dim, self.output_dim)

        self.adp_extractor = ASTUC(self.input_dim*self.node_num, self.embed_dim*self.node_num,self.args)
        self.adp_predictor = ASTUC(self.embed_dim*self.node_num, self.output_dim*self.node_num,self.args)
        self.fc_extractor = nn.Linear(self.seq_length, self.horizon)
        self.fc_predictor = nn.Linear(self.embed_dim+self.input_dim, self.output_dim)
        for _ in range(self.num_blocks):
            if self.args.mlp:
                self.blocks.append(MLP(self.args)) # Compare with mlp to see the effect
            else:
                self.blocks.append(MLRF(self.args)) 

#        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size, node_num):
        init_states = []
        for i in range(self.num_nalls):
            init_states.append(self.blocks[i].init_hidden_state(batch_size, node_num))
        return torch.stack(init_states, dim=0)

    def forward(self, x, label=None, iter=None):
        B,T,N,D = x.shape
        x0 = x.clone()

        if  self.args.without_backbone or self.backbone is None:
            model_emb = self.stu_emb.expand(size=(B, *self.stu_emb.shape))
            features = [x, model_emb]
            x = torch.cat(features ,dim=-1) # [B,T,N,D]->[B,T,N,D+embed_dim]
            output = self.fc_extractor(x.transpose(1,3)).transpose(1,3) # [B,T,N,E]->[B,E,N,T]->[B,E,N,H]
            output = self.fc_predictor(output) # [B,H,N,E]->[B,H,N,output_dim]
        elif self.backbone is not None:
            output = self.backbone(x,iter)
        else:
            output = torch.zeros(B, self.horizon, N, self.output_dim).to(self.device)

        x = x0
        x = self.adp_extractor(x.reshape(B,T,N*D)) # [B,T,N,input_dim]->[B,T,N,embed_dim]

        for i in range(self.num_blocks): # [B,T,N,embed_dim]
            x = self.blocks[i](x) # Relu / Leakly  are the same

        tuning = self.adp_predictor(x).reshape(B,T,N,-self.output_dim) # [B,T,N,embed_dim]->[B,T,N,output_dim]

        #return tuning

        #return output + tuning 

        gating_input = torch.cat([x0, tuning], dim=-1)
        r = F.relu(self.gate(gating_input))
        return r * output + (1-r) * tuning # gate