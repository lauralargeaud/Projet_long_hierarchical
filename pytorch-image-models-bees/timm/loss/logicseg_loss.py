import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss.logicseg.c_rule_loss import CRuleLoss
from timm.loss.logicseg.d_rule_loss import DRuleLoss
from timm.loss.logicseg.e_rule_loss import ERuleLoss
from timm.loss.logicseg.asym_loss import ASL
from timm.loss.logicseg.multi_bce_loss import MultiBCE
from timm.loss.binary_cross_entropy import BinaryCrossEntropy

class LogicSegLoss(nn.Module):
    def __init__(self, method, H_raw, P_raw, M_raw, La_raw, alpha_c, alpha_d, alpha_e, alpha_target_loss, alpha_layer, gamma_pos=1, gamma_neg=1, thresh_shifting=0, class_indices=None, order_indices=None, family_indices=None, genus_indices=None, species_indices=None, images_count_by_family=None, images_count_by_order=None, images_count_by_genus=None, images_count_by_class=None, images_count_by_species=None):
        super(LogicSegLoss, self).__init__()
        
        self.c_rule = CRuleLoss(H_raw)
        self.alpha_c = alpha_c

        self.d_rule = DRuleLoss(H_raw)
        self.alpha_d = alpha_d

        self.e_rule = ERuleLoss(P_raw, M_raw)
        self.alpha_e = alpha_e
        
        self.method = method
        self.alpha_target_loss = alpha_target_loss

        # Enregistrer les indices
        self.class_indices = class_indices
        self.order_indices = order_indices
        self.family_indices = family_indices
        self.genus_indices = genus_indices
        self.species_indices = species_indices

        # Enregistrer les comptages d'images
        self.images_count_by_family = images_count_by_family  
        self.images_count_by_order = images_count_by_order 
        self.images_count_by_genus = images_count_by_genus
        self.images_count_by_class = images_count_by_class
        self.images_count_by_species = images_count_by_species

        if method == "asl":
            self.asl = ASL(gamma_pos, gamma_neg, thresh_shifting)
        elif method == "multi_bce":
            self.multi_bce = MultiBCE(La_raw, alpha_layer)

  
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, verbose: bool = False, losses_dict=None) -> torch.Tensor:

        y_pred, y_true = y_pred.float(), y_true.float()

        # apply the sigmoid function in order to compute the nb_nodes probabilities for each image
        y_pred_sigmoid = torch.sigmoid(y_pred)
        
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
            case "bce_weight":
                # Instancier BinaryCrossEntropy avec les pondérations hiérarchiques et les indices
                bce_weighted_loss = BinaryCrossEntropy(
                    smoothing=0.1,
                    target_threshold=None,
                    sum_classes=False,
                    pos_weight=None,
                    class_indices=self.class_indices,
                    order_indices=self.order_indices,
                    family_indices=self.family_indices,
                    genus_indices=self.genus_indices,
                    species_indices=self.species_indices,
                    images_count_by_class=self.images_count_by_class,
                    images_count_by_order=self.images_count_by_order,
                    images_count_by_family=self.images_count_by_family,
                    images_count_by_genus=self.images_count_by_genus,
                    images_count_by_species=self.images_count_by_species,
                )
                # Calculer la perte
                target_loss = bce_weighted_loss(y_pred, y_true)

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