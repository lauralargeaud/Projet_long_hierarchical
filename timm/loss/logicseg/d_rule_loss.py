import torch
import torch.nn as nn

class DRuleLoss(nn.Module):
    def __init__(self, H_raw): # H_raw is a np array
        super(DRuleLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.H = torch.tensor(H_raw, dtype=torch.float32).to(device)
        one_tensor = torch.tensor(1).unsqueeze(0).repeat((self.H.shape[0],1))
        sum = torch.sum(self.H, dim=1).unsqueeze(1)
        self.branches = torch.minimum(sum, one_tensor).to(device)
        self.branch_count = torch.sum(self.branches).to(device)
        # BON jusque là

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        batch_size = y_pred.shape[0] # nombre d'images dans le batch
        N = y_pred.shape[1] # nombre de noeuds dans l'arbre

        S = y_pred.unsqueeze(2).repeat(1, 1, N)  # shape [batch_size, N, N]
        H_batch = self.H.unsqueeze(0).repeat(batch_size, 1, 1,)

        branches = self.branches.T.repeat(batch_size, 1)
        s_branch = branches * y_pred

        max_values = torch.max(S*H_batch, dim=2).values

        losses = (torch.sum(s_branch, dim=1) - torch.sum(s_branch*max_values, dim=1))/self.branch_count
        
        total_loss = torch.sum(losses) / batch_size

        return total_loss