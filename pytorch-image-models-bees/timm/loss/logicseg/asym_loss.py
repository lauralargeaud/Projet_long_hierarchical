import torch
import torch.nn as nn

class ASL(nn.Module):
    def __init__(self, gamma_pos, gamma_neg, thresh_shifting): # H_raw is a np array
        super(ASL, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.thresh_shifting = thresh_shifting
        self.eps = 1e-8
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.gamma_pos = torch.tensor(gamma_pos, dtype=torch.float32).to(device)
        # self.gamma_neg = torch.tensor(gamma_neg, dtype=torch.float32).to(device)
        # self.m = torch.tensor(m, dtype=torch.float32).to(device)
        # self.eps = torch.tensor(1e-8, dtype=torch.float32).to(device)
    
    # In rest of lib, y_pred is written as x, and y_true as target
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Positive loss (for y=1)
        pos_loss = y_true * torch.pow((1 - y_pred), self.gamma_pos) * torch.log(y_pred + self.eps) # (batch_size, N)
        # Negative loss (for y=0) with thresholding 
        y_pred_m = (y_pred - self.thresh_shifting).clamp(min=0) # (batch_size, N)
        neg_loss = (1 - y_true) * torch.pow(y_pred_m, self.gamma_neg) * torch.log(1 - y_pred_m + self.eps) # (batch_size, N)
        # Compute total loss (sum over all samples and classes)
        loss = -pos_loss - neg_loss # (batch_size, N)
        return loss.mean()