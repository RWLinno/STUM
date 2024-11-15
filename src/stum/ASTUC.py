import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ASTUC(nn.Linear):
    def __init__(self, in_features, out_features, args, bias=True, **kwargs):
        super(ASTUC, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self.args = args

        self._A = nn.Parameter(torch.zeros(args.embed_dim, in_features), requires_grad=True)
        self._B = nn.Parameter(torch.zeros(out_features, args.embed_dim), requires_grad=True)

        self._alpha = 16
        self._dropout = nn.Dropout(p=0.3)
        self.r = args.embed_dim  # rank
        self.scaling = self._alpha / self.r

        self.weight.requires_grad = False

        nn.init.kaiming_uniform_(self._A, a=math.sqrt(5.))
        nn.init.zeros_(self._B)

        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, self.bias)
        _result = (self._dropout(x) @ self._A.transpose(0, 1)) @ self._B.transpose(0, 1)
        result += _result * self.scaling

        return result

    def train_mode_toggle(self, mode: bool = True):
        if mode:
            self.weight.data -= (self._B @ self._A) * self.scaling
        else:
            self.weight.data += (self._B @ self._A) * self.scaling
        super(ASTUC, self).train(mode)