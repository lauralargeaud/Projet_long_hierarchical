import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import pandas as pd
import numpy as np

from logic_seg_utils import *

class DRuleLoss(nn.Module):
    def __init__(self, path_to_csv_tree):
        super(DRuleLoss, self).__init__()
        self.H,_,_ = get_tree_matrix(path_to_csv_tree)
        self.branches = torch.minimum(torch.sum(self.H, dim=1), torch.tensor(1))
        self.branch_count = torch.sum(self.branches)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        batch_size = y_pred.size(0) # nombre d'images dans le batch
        n = y_pred.size(1) # nombre de noeuds dans l'arbre

        S = y_pred.unsqueeze(2).repeat(1, 1, n)  # shape [batch_size, N, N]
        H_batch = self.H.unsqueeze(0).repeat(batch_size, 1, 1,)

        branches = self.branches.T.repeat(batch_size, 1)
        s_branch = branches * y_pred

        losses = (torch.sum(s_branch, dim=1) + torch.sum(s_branch*torch.max(S*H_batch, dim=2), dim=1))/self.branch_count
        
        loss = sum(losses,0)
        return loss