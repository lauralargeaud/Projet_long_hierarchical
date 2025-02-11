""" 
Hierarchical Cross Entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchicalCrossEntropy(nn.Module):
    
    def __init__(self, L, alpha, h):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hc = torch.tensor([i for i in range(h, 0, -1)]).to(device)
        print(hc)
        self.L = torch.tensor(L, dtype=torch.float32).to(device)
        self.alpha = torch.tensor(alpha).to(device)
        self.h = torch.tensor(h).to(device)
        self.lam = torch.exp(-self.alpha * hc).to(device)

    def forward(self, logits, target_classes):
        """
        :param logits: Tensor de forme (batch_size, num_classes) contenant les log-probabilités.
        :param target_classes: Tensor de forme (batch_size,) avec les indices des classes cibles.
        :return: Perte HXE scalaire.
        """
        batch_size = logits.shape[0]
        loss = 0.0
        log_probs = F.softmax(logits, dim=1)  # Convertir les logits en log-probabilités

        for i in range(batch_size):
            target = target_classes[i].item()
            probs = torch.unsqueeze(log_probs[i,:], 0)
            print(target)
            print(torch.unsqueeze(logits[i,:], 0))
            print(probs)

            LS = self.L @ torch.transpose(probs, 0, 1)

            target_parent = self.L[:, target]

            P = torch.mul(target_parent, torch.transpose(LS, 0, 1))

            PP = P[torch.unsqueeze(target_parent, 0) != 0]

            num = PP[1:]
            den = PP[:-1]

            log = torch.log(num / den)
            term_sum = - self.lam * log
            print(f"loss: {torch.sum(term_sum)}")
            loss += torch.sum(term_sum)

        return loss / batch_size  # Moyenne sur le batch
