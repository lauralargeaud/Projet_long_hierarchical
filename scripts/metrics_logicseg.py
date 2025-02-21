from scripts.logic_seg_utils import *
from math import *

""" Métriques de logicSeg
"""

def topk_accuracy_logicseg(probas_branches_input, onehot_targets, topk=1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    topk = min(topk,probas_branches_input.shape[1])
    _, indices_branches_target = onehot_targets.topk(1, dim=1) # (nb_pred, 1), (nb_pred, 1)
    indices_branches_target = indices_branches_target.repeat(1, topk) # (nb_pred, top_k)
    _, indices_branches_in = probas_branches_input.topk(topk , dim=1) # (nb_pred, top_k), (nb_pred, top_k)
    acc = torch.sum(torch.any(indices_branches_in == indices_branches_target, dim=1), dim=0) / indices_branches_in.size(0)
    return torch.tensor(acc)