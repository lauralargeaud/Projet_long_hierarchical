import torch
import torch.nn as nn

class CRuleLoss(nn.Module):
    def __init__(self, H_raw): # H_raw is a np array
        super(CRuleLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.H = torch.tensor(H_raw, dtype=torch.float32).to(device) #torch tensor
        self.cardinal = torch.sum(self.H, dim=None).to(device) #torch tensor scalar
  
    # In rest of lib, y_pred is written as x, and y_true as target
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]
        Hs = self.H @ y_pred.T # (N, batch_size)
        sTHs = torch.matmul(y_pred, Hs) # (batch_size, batch_size)
        sTHs = sTHs.diagonal().unsqueeze(0) # (1, batch_size)
        batch_losses = ((torch.sum(Hs, dim=0) - sTHs) / self.cardinal) # shape = [1, batch_size]
        batch_losses = batch_losses.squeeze(0)
        total_loss = torch.sum(batch_losses) / batch_size
        return total_loss