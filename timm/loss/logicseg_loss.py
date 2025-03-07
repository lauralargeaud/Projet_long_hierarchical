import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss.logicseg.c_rule_loss import CRuleLoss
from timm.loss.logicseg.d_rule_loss import DRuleLoss
from timm.loss.logicseg.e_rule_loss import ERuleLoss
from timm.loss.logicseg.asym_loss import ASL
from timm.loss.logicseg.multi_bce_loss import MultiBCE

class LogicSegLoss(nn.Module):
    def __init__(self, method, H_raw, P_raw, M_raw, La_raw, alpha_c, alpha_d, alpha_e, alpha_target_loss, alpha_layer, gamma_pos = 1, gamma_neg = 1, thresh_shifting = 0): # H_raw is a np array
        super(LogicSegLoss, self).__init__()
        
        self.c_rule = CRuleLoss(H_raw)
        self.alpha_c = alpha_c

        self.d_rule = DRuleLoss(H_raw)
        self.alpha_d = alpha_d

        self.e_rule = ERuleLoss(P_raw, M_raw)
        self.alpha_e = alpha_e

        # self.bce = BinaryCrossEntropy()
        self.method = method
        self.alpha_target_loss = alpha_target_loss

        if method == "asl":
            self.asl = ASL(gamma_pos, gamma_neg, thresh_shifting)
        elif method == "multi_bce":
            self.multi_bce = MultiBCE(La_raw, alpha_layer)

  
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, verbose: bool = False, losses_dict=None) -> torch.Tensor:

        y_pred, y_true = y_pred.float(), y_true.float()

        # apply the sigmoid function in order to compute the nb_nodes probabilities for each image
        # y_pred_sigmoid = torch.sigmoid(y_pred)
        y_pred_sigmoid = y_pred
        batch_c_losses = self.c_rule(y_pred_sigmoid, y_true)
        if verbose:
            print("c_losses", batch_c_losses.item())

        batch_d_losses = self.d_rule(y_pred_sigmoid, y_true)     
        if verbose:
            print("d_losses", batch_d_losses.item())

        batch_e_losses = self.e_rule(y_pred_sigmoid, y_true)
        if verbose:
            print("e_losses", batch_e_losses.item())

        target_loss = 0
        match self.method:
            case "ce":
                target_loss = F.cross_entropy(y_pred, y_true)
            case "bce":
                target_loss = F.binary_cross_entropy(y_pred_sigmoid, y_true)
            case "asl":
                target_loss = self.asl(y_pred_sigmoid, y_true)
            case "multi_bce":
                target_loss = self.multi_bce(y_pred_sigmoid, y_true)
                
        if losses_dict != None:
            losses_dict["C_loss"] = batch_c_losses
            losses_dict["D_loss"] = batch_d_losses
            losses_dict["E_loss"] = batch_e_losses
            losses_dict["target_loss"] = target_loss
      
        if verbose:
            print("target_loss", target_loss.item())

        return self.alpha_c * batch_c_losses + \
            self.alpha_d * batch_d_losses + \
            self.alpha_e * batch_e_losses + \
            self.alpha_target_loss * target_loss