import torch
import torch.nn as nn
import torch.nn.functional as F
import src.loralib.layers as lora

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = lora.Conv1d(n_state, nx)
        self.c_proj = lora.Conv1d(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2
    
class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, len_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, len_past=len_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

# 节点卷积操作
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

# 通过 1x1 的卷积操作实现的全连接层
# 如果我使用LoRA机制的线性层/嵌入层
'''
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)
'''

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        # self.mlp = lora.Linear(in_features=c_in, out_features=c_out,r=10,lora_alpha=0.1,lora_dropout=0.1)
        self.mlp = lora.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True,
                 aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2,aptonly_r=10):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.aptonly_r = aptonly_r
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = lora.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                # 为了自适应邻接矩阵（Adaptive Adjacency Matrix）而设计的，但在 LoRA 中可能不需要。
                # 这东西师兄说跟lora如出一辙，考虑换成lora的分解形式
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, self.aptonly_r).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(self.aptonly_r, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                
                
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(lora.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(lora.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(lora.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(lora.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
        
        # 图最后两个linear是用Conv2d实现的,这里考虑做修改
        #self.end_conv_1 = linear(skip_channels, end_channels)
        # self.end_conv_2 = linear(end_channels, out_dim)

        self.end_conv_1 = lora.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        self.end_conv_2 = lora.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution   
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

'''
class LoRA4STG(nn.Module,lora.LoRALayer):
    def __init__(self, model, lora_params):
        super(LoRA4STG, self).__init__()
        self.model = model
        print(model)
        self.lora_params = lora_params
        self.ml = lora.MergedLinear(in_features=model.in_dim, 
                                    out_features=model.out_dim, 
                                    r=lora_params['r'], 
                                    lora_alpha=lora_params['lora_alpha'], 
                                    lora_dropout=lora_params['lora_dropout'])
        

        print("LoRA4STG.init")
        print(self.ml)
        all_p = 0
        for layer, params in model.items():
            if isinstance(layer, lora.Linear):
                print("Linear")
            elif isinstance(layer, lora.Conv2d):
                print('Conv2d')
            elif isinstance(layer, lora.Conv1d):
                print('Conv1d')
            else:
                print(type(layer))
            all_p += params.numel()
            print(layer, params.numel())
            if params.ndim == 1:
                params.data = params.data.to(torch.float32)
        print("all params:",all_p)
        layers = [layer for layer in self.model.modules()][:]
        self.lora_layers = [layer for layer in layers if isinstance(layer, lora.LoRALayer)]

        for layer in layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    print(f"{layer.__class__.__name__} - {name}: {param.shape}")

    def forward(self, x):
        x = self.model(x)
        for lora_layer in self.lora_layers:
            x = lora_layer(x)
        return x
    
->print(layers)
[gwnet(
  (filter_convs): ModuleList(
    (0): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1))
    (1): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
    (2): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1))
    (3): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
    (4): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1))
    (5): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
    (6): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1))
    (7): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
  )
  (gate_convs): ModuleList(
    (0): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,))
    (1): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,))
    (2): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,))
    (3): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,))
    (4): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,))
    (5): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,))
    (6): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,))
    (7): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,))
  )
  (residual_convs): ModuleList(
    (0): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
    (1): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
    (2): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
    (3): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
    (4): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
    (5): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
    (6): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
    (7): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
  )
  (skip_convs): ModuleList(
    (0): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
    (1): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
    (2): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
    (3): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
    (4): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
    (5): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
    (6): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
    (7): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
  )
  (bn): ModuleList(
    (0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (gconv): ModuleList()
  (start_conv): Conv2d(2, 32, kernel_size=(1, 1), stride=(1, 1))
  (end_conv_1): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
  (end_conv_2): Conv2d(512, 12, kernel_size=(1, 1), stride=(1, 1))
), ModuleList(
  (0): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1))
  (1): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
  (2): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1))
  (3): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
  (4): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1))
  (5): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
  (6): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1))
  (7): Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
), Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1)), Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2)), Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1)), Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2)), Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1)), Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2)), Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1)), Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2)), ModuleList(
  (0): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,))
  (1): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,))
  (2): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,))
  (3): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,))
  (4): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,))
  (5): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,))
  (6): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,))
  (7): Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,))
), Conv1d(32, 32, kernel_size=(1, 2), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,)), Conv1d(32, 32, kernel_size=(1, 2), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,)), Conv1d(32, 32, kernel_size=(1, 2), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,)), Conv1d(32, 32, kernel_size=(1, 2), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 2), stride=(1,), dilation=(2,)), ModuleList(
  (0): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
  (1): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
  (2): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
  (3): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
  (4): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
  (5): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
  (6): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
  (7): Conv1d(32, 32, kernel_size=(1, 1), stride=(1,))
), Conv1d(32, 32, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 32, kernel_size=(1, 1), stride=(1,)), ModuleList(
  (0): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
  (1): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
  (2): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
  (3): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
  (4): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
  (5): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
  (6): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
  (7): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
), Conv1d(32, 256, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 256, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 256, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 256, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 256, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 256, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 256, kernel_size=(1, 1), stride=(1,)), Conv1d(32, 256, kernel_size=(1, 1), stride=(1,)), ModuleList(
  (0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
), BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ModuleList(), Conv2d(2, 32, kernel_size=(1, 1), stride=(1, 1)), Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1)), Conv2d(512, 12, kernel_size=(1, 1), stride=(1, 1))]
'''