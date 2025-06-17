import torch
import torch.nn as nn


class ERuleLoss(nn.Module):
    def __init__(self, P_raw, M_raw): # P_raw and M_raw are np arrays
        super(ERuleLoss, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = torch.tensor(P_raw, dtype=torch.float32).to(device)
        self.M = torch.maximum(torch.tensor(M_raw, dtype=torch.float32), torch.tensor(1)).to(device)
  
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]
        N = y_pred.shape[1]
        K = self.P @ y_pred.T
        M_exp = self.M.unsqueeze(1).repeat(1, batch_size)
        K = torch.div(K, M_exp)
        K = y_pred @ K
        K = K.diagonal()
        losses = K / N # shape (batch_size,)
        total_loss = torch.sum(losses) / batch_size
        return total_loss