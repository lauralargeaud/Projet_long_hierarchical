import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import pandas as pd
import numpy as np

from logic_seg_utils import get_H_matrix

class DRuleLoss(nn.Module):
    def __init__(self, path_to_csv_tree):
        super(DRuleLoss, self).__init__()
        self.H = H
        self.branches = torch.minimum(torch.sum(H, dim=1), torch.tensor(1))
        self.branch_count = torch.sum(self.branches)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        s_branch = self.branches * y_pred

        n = y_pred.size(0)
        S = y_pred.repeat(n, 1)

        loss = (torch.sum(s_branch) + torch.sum(s_branch*torch.max(S*self.H, dim=1), dim=0))/self.branch_count

        return loss