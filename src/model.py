import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def build_model(input_dim: int) -> nn.Module:

    model = MLP(
        input_dim=input_dim,
        hidden_sizes=[64, 32],
        dropout=0.2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    return model