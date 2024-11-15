import torch
import torch.nn as nn

class MLP(nn.Module):
    """Two fully connected layer."""

    def __init__(self, args, hidden_dim=256):
        super().__init__()
        self.args = args
        self.device = args.device
        self.fc1 = nn.Linear(args.embed_dim*args.node_num, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, args.embed_dim*args.node_num)
            
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)
    
    def forward(self, history_data):
        """
        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, E].
        Returns:
            torch.Tensor: outputs with shape [B, L, N, E]
        """
        prediction = self.fc1(history_data)
        prediction = self.relu(prediction)
        prediction = self.fc2(prediction)
        return prediction