import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 512]):
        super(MLP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        return x
    
    
def projection_mlp(in_dims: int,
                    h_dims: int,
                    out_dims: int,
                    num_layers: int = 3) -> nn.Sequential:
    """Projection MLP. The original paper's implementation has 3 layers, with
    BN applied to its hidden fc layers but no ReLU on the output fc layer.
    The CIFAR-10 study used a MLP with only two layers.

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims:
            Hidden dimension of all the fully connected layers.
        out_dims:
            Output Dimension of the final linear layer.
        num_layers:
            Controls the number of layers; must be 2 or 3. Defaults to 3.

    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims, bias=False),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Sequential(nn.Linear(h_dims, h_dims, bias=False),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l3 = nn.Sequential(nn.Linear(h_dims, out_dims),
                       nn.BatchNorm1d(out_dims))

    if num_layers == 3:
        projection = nn.Sequential(l1, l2, l3)
    elif num_layers == 2:
        projection = nn.Sequential(l1, l3)
    else:
        raise NotImplementedError("Only MLPs with 2 and 3 layers are implemented.")

    return projection

