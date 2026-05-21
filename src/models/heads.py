import torch, torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512, hidden=0, norm=True):
        super().__init__()
        layers = []
        if hidden > 0:
            layers += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)]
        else:
            layers += [nn.Linear(in_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self.norm = norm

    def forward(self, x):
        y = self.net(x)
        if self.norm:
            y = y / (y.norm(dim=-1, keepdim=True) + 1e-8)
        return y
