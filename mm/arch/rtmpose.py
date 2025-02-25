import torch
import torch.nn as nn


class End2End(nn.Module):
    def __init__(self, model, data_samples) -> None:
        super().__init__()

        self.model = model.eval()
        self.data_samples = data_samples

    def forward(self, inputs):
        simcc_x, simcc_y = self.model(inputs, self.data_samples)
        return simcc_x, simcc_y


def get_simcc_maximum(self,
                        simcc_x: torch.Tensor,
                        simcc_y: torch.Tensor,
                        simcc_split_ratio: float = 2.0) -> torch.Tensor:
    """Get maximum response location and value from simcc representations.

    rewrite to support `torch.Tensor` input type.

    Args:
        simcc_x (torch.Tensor): x-axis SimCC in shape (N, K, Wx)
        simcc_y (torch.Tensor): y-axis SimCC in shape (N, K, Wy)

    Returns:
        tuple:
        - locs (torch.Tensor): locations of maximum heatmap responses in shape
            (N, K, 2)
        - vals (torch.Tensor): values of maximum heatmap responses in shape
            (N, K)
    """
    N, K, _ = simcc_x.shape
    simcc_x = simcc_x.flatten(0, 1)
    simcc_y = simcc_y.flatten(0, 1)
    x_locs = simcc_x.argmax(dim=1, keepdim=True)
    y_locs = simcc_y.argmax(dim=1, keepdim=True)
    locs = torch.cat((x_locs, y_locs), dim=1).to(torch.float32)
    max_val_x, _ = simcc_x.max(dim=1, keepdim=True)
    max_val_y, _ = simcc_y.max(dim=1, keepdim=True)
    vals, _ = torch.cat([max_val_x, max_val_y], dim=1).min(dim=1)
    locs = locs.reshape(N, K, 2)
    locs /= simcc_split_ratio
    vals = vals.reshape(N, K)
    return locs, vals