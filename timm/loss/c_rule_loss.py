import torch
import torch.nn as nn

class CRuleLoss(nn.Module):
    def __init__(self, H_raw):
        super(CRuleLoss, self).__init__()
        # H_raw = load(path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.H = get_H_matrix(path_to_csv_tree)
        self.H = H_raw
        H = torch.tensor(self.H, dtype=torch.float32).to(device)
        self.H = H
        self.cardinal = torch.sum(self.H, dim=None)
  
    # In rest of lib, y_pred is written as x, and y_true as target
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # batch_size = y_pred.shape[0]
        Hs = self.H @ y_pred.T # (N, batch_size)
        sTHs = torch.matmul(y_pred, Hs) # (batch_size, batch_size)
        sTHs = sTHs.diagonal().unsqueeze(0) # (1, batch_size)
        batch_losses = ((torch.sum(Hs, dim=0) - sTHs) / self.cardinal) # shape = [1, batch_size]
        batch_losses = batch_losses.squeeze(0)
        # total_loss = torch.sum(batch_losses, dim=0) / batch_size
        # total_loss = torch.sum(batch_losses, dim=1)
        return batch_losses