import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss.logicseg.c_rule_loss import CRuleLoss
from timm.loss.logicseg.d_rule_loss import DRuleLoss
from timm.loss.logicseg.e_rule_loss import ERuleLoss

class LogicSegLoss(nn.Module):
    def __init__(self, H_raw, P_raw, M_raw, alpha_c, alpha_d, alpha_e, alpha_bce): # H_raw is a np array
        super(LogicSegLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.c_rule = CRuleLoss(H_raw)
        self.alpha_c = alpha_c

        self.d_rule = DRuleLoss(H_raw)
        self.alpha_d = alpha_d

        self.e_rule = ERuleLoss(P_raw, M_raw)
        self.alpha_e = alpha_e

        # self.bce = BinaryCrossEntropy()
        self.alpha_bce = alpha_bce
  
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, with_print: bool = False) -> torch.Tensor:

        y_pred, y_true = y_pred.float(), y_true.float()

        # apply the sigmoid function in order to compute the nb_nodes probabilities for each image
        y_pred = torch.sigmoid(y_pred)
        
        if with_print:
            print("L_c =", self.c_rule(y_pred, y_true).item())
            print("L_d =", self.d_rule(y_pred, y_true).item())
            print("L_e =", self.e_rule(y_pred, y_true).item())
            # print(self.bce(y_pred, y_true))
            print("L_BCE =", F.binary_cross_entropy(y_pred, y_true).item())

        batch_size = y_pred.shape[0]
        print("initial Y_pred shape", y_pred.shape)
        print("initial Y_true shape", y_true.shape)
        
        batch_losses = batch_losses + self.alpha_c * self.c_rule(y_pred, y_true)
        print("Y_pred shape after c rule", y_pred.shape)
        print("Y_true shape after c rule", y_true.shape)

        batch_losses = batch_losses + self.alpha_d * self.d_rule(y_pred, y_true)
        print("Y_pred shape after d rule", y_pred.shape)
        print("Y_true shape after d rule", y_true.shape)

        batch_losses = batch_losses + self.alpha_e * self.e_rule(y_pred, y_true)
        print("Y_pred shape after e rule", y_pred.shape)
        print("Y_true shape after e rule", y_true.shape)

        print("Y_pred after e rule", y_pred)
        print("Y_true after e rule", y_true)

        batch_losses = batch_losses + self.alpha_bce * (F.binary_cross_entropy(y_pred, y_true) / batch_size)
        return batch_losses
        # return self.alpha_bce * (F.binary_cross_entropy(y_pred, y_true) / batch_size)