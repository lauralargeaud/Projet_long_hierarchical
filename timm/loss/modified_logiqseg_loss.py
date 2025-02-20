import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss.logicseg.c_rule_loss import CRuleLoss
from timm.loss.logicseg.d_rule_loss import DRuleLoss
from timm.loss.logicseg.e_rule_loss import ERuleLoss

class ModifiedLogicSegLoss(nn.Module):
    def __init__(self, H_raw, P_raw, M_raw, La_raw, alpha_c, alpha_d, alpha_e, alpha_bce, alpha_lam): # H_raw is a np array
        super(ModifiedLogicSegLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.c_rule = CRuleLoss(H_raw)
        self.alpha_c = alpha_c

        self.d_rule = DRuleLoss(H_raw)
        self.alpha_d = alpha_d

        self.e_rule = ERuleLoss(P_raw, M_raw)
        self.alpha_e = alpha_e

        # self.bce = BinaryCrossEntropy()
        self.alpha_bce = alpha_bce

        self.La = torch.tensor(La_raw).to(device)
        h = self.La.shape[0]
        hc = torch.tensor([i for i in range(h, 0, -1)]).to(device)
        self.lam = torch.exp(-alpha_lam * hc).to(device)
        print(self.La)
        print(self.lam)
  
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, verbose: bool = False) -> torch.Tensor:

        y_pred, y_true = y_pred.float(), y_true.float()

        # apply the sigmoid function in order to compute the nb_nodes probabilities for each image
        y_pred_sigmoid = torch.sigmoid(y_pred)

        # print("sigmoid", y_pred_sigmoid)
        # print("target", y_true)

        batch_c_losses = self.c_rule(y_pred_sigmoid, y_true)
        if verbose:
            print("c_losses", batch_c_losses.item())

        batch_d_losses = self.d_rule(y_pred_sigmoid, y_true)
        if verbose:
            print("d_losses", batch_d_losses.item())

        batch_e_losses = self.e_rule(y_pred_sigmoid, y_true)
        if verbose:
            print("e_losses", batch_e_losses.item())

        batch_size = y_pred.shape[0]
        bce_loss = 0
        for i in range(batch_size):
            loss_img = 0
            for layer in range(self.La.shape[0]):
                y_true_layer_img = torch.mul(y_true[i,:], self.La[layer,:])
                y_pred_layer_img = torch.mul(y_pred_sigmoid[i,:], self.La[layer,:])
                loss_img += F.binary_cross_entropy(y_pred_layer_img, y_true_layer_img) * self.lam[layer]
            bce_loss += loss_img
            if verbose:
                print("loss_img", loss_img.item())

        # batch_bce_ce_losses = F.binary_cross_entropy(y_pred_sigmoid, y_true)

        if verbose:
            # print("bce_losses", batch_bce_ce_losses.item())
            print("bce_layer", bce_loss.item() / batch_size)

        return self.alpha_c * batch_c_losses + \
            self.alpha_d * batch_d_losses + \
            self.alpha_e * batch_e_losses + \
            self.alpha_bce * bce_loss