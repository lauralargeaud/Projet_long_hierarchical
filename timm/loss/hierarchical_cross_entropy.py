""" 
Hierarchical Cross Entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchicalCrossEntropy(nn.Module):
    
    def __init__(self, L, path_to_root, alpha, h):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L = torch.tensor(L, dtype=int).to(device)
        self.path_to_root = torch.tensor(path_to_root, dtype=int).to(device)
        self.alpha = torch.tensor(alpha).to(device)
        self.h = torch.tensor(h).to(device)

    def forward(self, logits, target_classes):
        """
        :param logits: Tensor de forme (batch_size, num_classes) contenant les log-probabilités.
        :param target_classes: Tensor de forme (batch_size,) avec les indices des classes cibles.
        :return: Perte HXE scalaire.
        """
        batch_size = logits.shape[0]
        loss = 0.0

        log_probs = F.log_softmax(logits, dim=1)  # Convertir les logits en log-probabilités

        for i in range(batch_size):
            target = target_classes[i].item()
            path = self.path_to_root[target]
            print(path)

            for i in range(self.h - 2):
                lambda_weight = torch.exp(-self.alpha * (self.h - i - 1))
                print(self.L.size(), logits[i,:].size())
                log_p = (self.L[path[i], :] * logits[i,:]) / ((self.L[path[i+1], :] * logits[i,:]))
                loss -= lambda_weight * log_p
                
        return loss / batch_size  # Moyenne sur le batch
