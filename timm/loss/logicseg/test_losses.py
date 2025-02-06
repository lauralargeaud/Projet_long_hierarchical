import torch
# import pytest
from e_rule_loss import ERuleLoss  # Assure-toi que le fichier contenant la classe est bien nommé `erule_loss.py`
from c_rule_loss import CRuleLoss
from d_rule_loss import DRuleLoss

def test_losses_forward():
    # Définition des matrices P et M factices
    P_raw = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1.0, 0, 0], [0, 0, 0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1.0],
             [0, 0, 0, 0, 0, 1.0, 0]]
    M_raw = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    H_raw = [[0, 1.0, 1.0, 0, 0, 0, 0], [0, 0, 0, 1.0, 1.0, 0, 0], [0, 0, 0, 0, 0, 1.0, 1.0],
             [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]

    # C, D, E ok
    loss_ind = "E"
    if loss_ind == "C":

        # Création de l'instance de la classe C
        criterion = CRuleLoss(H_raw)

    elif loss_ind == "D":

        # Création de l'instance de la classe D
        criterion = DRuleLoss(H_raw)

    elif loss_ind == "E":
    
        # Création de l'instance de la classe E
        criterion = ERuleLoss(P_raw, M_raw)
    
    # Tenseur d'entrée simulé (batch_size=2, N=2)
    # y_pred = torch.tensor([[1.0, 1.0, 0, 1.0, 0, 0, 0], [1.0, 0, 1.0, 0, 0, 1.0, 0]], dtype=torch.float32)
    # ==> on a bien des loss toujours nulles

    y_pred = torch.tensor([[1.0, 0.8, 0.2, 0.7, 0.3, 0, 0], [1.0, 0.8, 0.2, 0.7, 0.1, 0.2, 0]], dtype=torch.float32)
    # on a:
    # lossC = tensor([0.0333, 0.0533])
    # lossD = tensor([0.1067, 0.1067])
    # lossE = tensor([0.0600, 0.0200])

    # y_pred = torch.tensor([[1.0, 0.6, 0.5, 0.5, 0.5, 0, 0], [1.0, 0.3, 0.7, 0.7, 0.1, 0.2, 0]], dtype=torch.float32)
    # on a:
    # lossC = tensor([0.0667, 0.1033])
    # lossD = tensor([0.1633, 0.1400])
    # lossE = tensor([0.0714, 0.0200])

    y_true = torch.tensor([], dtype=torch.float32)

    # Calcul de la perte en appelant explicitement forward
    loss = criterion.forward(y_pred, y_true)
    
    # Vérification de la forme de sortie
    assert loss.shape == torch.Size([2]), "La sortie de la perte doit être de dimension (batch_size,)"
    
    # Vérification des valeurs attendues (exemple avec valeurs pré-calculées si possible)
    assert torch.isfinite(loss).all(), "La perte ne doit contenir que des valeurs finies"

    print("loss = ", loss)

test_losses_forward()
