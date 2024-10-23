import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ASTUC(nn.Linear):
    def __init__(self, in_features, out_features, args, bias=True, **kwargs):
        super(ASTUC, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self.args = args

        self.lora_A = nn.Parameter(torch.zeros(args.embed_dim, in_features), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(out_features, args.embed_dim), requires_grad=True)

        self.lora_alpha = 16
        self.lora_dropout = nn.Dropout(p=0.3)
        self.r = args.embed_dim  # rank
        self.scaling = self.lora_alpha / self.r

        self.weight.requires_grad = False

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.))
        nn.init.zeros_(self.lora_B)

        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, self.bias)
        lora_result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        result += lora_result * self.scaling

        return result

    def train_mode_toggle(self, mode: bool = True):
        if mode:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
        else:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        super(ASTUC, self).train(mode)