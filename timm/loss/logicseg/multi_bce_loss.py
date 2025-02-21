import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBCE(nn.Module):
    def __init__(self, La_raw, alpha_layer): # H_raw is a np array
        super(MultiBCE, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.La = torch.tensor(La_raw).to(device) # La[i,j] = 1 si le noeud d'index j est de profondeur i, sinon 0
        self.h = self.La.shape[0] # profondeur de l'arbre
        hc = torch.tensor([i for i in range(self.h, 0, -1)]).to(device) # [h, h-1, ..., 1]
        self.lam = torch.exp(-alpha_layer * hc).to(device) # [exp(-alpha*h), ..., exp(-alpha)]
    
    # In rest of lib, y_pred is written as x, and y_true as target
    # def forward_old(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    #     batch_size = y_pred.shape[0]
    #     loss = 0
    #     for i in range(batch_size):
    #         loss_img = 0
    #         for layer in range(self.h):
    #             y_true_layer_img = torch.mul(y_true[i,:], self.La[layer,:])
    #             y_pred_layer_img = torch.mul(y_pred[i,:], self.La[layer,:])
    #             loss_img += F.binary_cross_entropy(y_pred_layer_img, y_true_layer_img) * self.lam[layer]
    #         loss += loss_img

    #     return loss / batch_size
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        batch_size = y_pred.shape[0]
        nb_nodes = y_pred.shape[1]

        La_rep = self.La.unsqueeze(0).repeat(batch_size, 1, 1)
        
        Y_true = torch.empty((batch_size, self.h, nb_nodes), dtype=torch.float32)
        Y_true = y_true.unsqueeze(1).repeat(1, self.h, 1)
        Y_true = Y_true * La_rep
        
        Y_pred = torch.empty((batch_size, self.h, nb_nodes), dtype=torch.float32)
        Y_pred = y_pred.unsqueeze(1).repeat(1, self.h, 1)
        Y_pred = Y_pred * La_rep

        L = F.binary_cross_entropy(Y_pred, Y_true, reduction='none') # (batch_size, self.h, nb_nodes)
        L = torch.sum(L, dim=2) / nb_nodes # (batch_size, self.h)
        L = torch.sum(L, dim=0) # (self.h,)
        L = (self.lam * L) 
        L = torch.sum(L) / batch_size   # scalar
    
        return L