# federated/model/mlp.py

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        input_dim = args.input_dim
        num_classes = 1     # Binary classification
        dropout = args.dropout

        # Three-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)