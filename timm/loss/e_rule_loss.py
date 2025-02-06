import torch
import torch.nn as nn

class ERuleLoss(nn.Module):
    def __init__(self, P_raw, M_raw):
        super(ERuleLoss, self).__init__()
        # P_raw = load(path)
        # M_raw = load(path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = torch.tensor(P_raw, dtype=torch.float32).to(device)
        self.M = torch.tensor(M_raw, dtype=torch.float32).to(device)
  
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]
        N = y_pred.shape[1]
        K = self.P @ y_pred.T
        M_exp = self.M.unsqueeze(1).repeat(1, batch_size)
        K = torch.div(K, M_exp)
        K = y_pred @ K
        K = K.diagonal()
        losses = K / N # shape (batch_size)
        # total_loss = torch.sum(losses, dim=1)
        return losses
    
    # OLD
    # # In rest of lib, y_pred is written as x, and y_true as target
    # def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    #     batch_size = y_pred.shape[0]
    #     N = y_pred.shape[1] # number of nodes in the tree
    #     S = y_pred.unsqueeze(2).repeat(1, 1, N)  # shape [batch_size, N, N]
    #     P_exp = self.P.unsqueeze(0).repeat(batch_size, 1, 1)
    #     A = S * P_exp
    #     As = A @ y_pred.T
    #     As = As[torch.arange(N), torch.arange(N), :] # [N, batch_size]
    #     M_exp = self.M.unsqueeze(2).repeat(1, 1, batch_size)
    #     batch_losses =  (torch.sum(torch.div(As, M_exp), dim=0) / N).T # shape = [batch_size, 1]
    #     # total_loss = torch.sum(batch_losses, dim=0) / batch_size
    #     total_loss = torch.sum(batch_losses, dim=0)
    #     return total_loss