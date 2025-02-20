import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBCE(nn.Module):
    def __init__(self, La_raw, alpha_layer): # H_raw is a np array
        super(MultiBCE, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.La = torch.tensor(La_raw).to(device)
        h = self.La.shape[0]
        hc = torch.tensor([i for i in range(h, 0, -1)]).to(device)
        self.lam = torch.exp(-alpha_layer * hc).to(device)
    
    # In rest of lib, y_pred is written as x, and y_true as target
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]
        loss = 0
        for i in range(batch_size):
            loss_img = 0
            for layer in range(self.La.shape[0]):
                y_true_layer_img = torch.mul(y_true[i,:], self.La[layer,:])
                y_pred_layer_img = torch.mul(y_pred[i,:], self.La[layer,:])
                loss_img += F.binary_cross_entropy(y_pred_layer_img, y_true_layer_img) * self.lam[layer]
            loss += loss_img
        return loss