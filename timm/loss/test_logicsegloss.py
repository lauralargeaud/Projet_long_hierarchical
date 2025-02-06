import torch
from logicseg_loss import LogicSegLoss

def test_loss_forward():
    # Définition des matrices P et M factices
    P_raw = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1.0, 0, 0], [0, 0, 0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1.0],
             [0, 0, 0, 0, 0, 1.0, 0]]
    M_raw = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    H_raw = [[0, 1.0, 1.0, 0, 0, 0, 0], [0, 0, 0, 1.0, 1.0, 0, 0], [0, 0, 0, 0, 0, 1.0, 1.0],
             [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]

    alpha_c = 1
    alpha_d = 1
    alpha_e = 1
    alpha_bce = 1
    criterion = LogicSegLoss(H_raw, P_raw, M_raw, alpha_c, alpha_d, alpha_e, alpha_bce)
    
    y_true = torch.tensor([[1.0, 1.0, 0, 1.0, 0, 0, 0], [1.0, 0, 1.0, 0, 0, 1.0, 0]], dtype=torch.float32)

    # Tenseur d'entrée simulé (batch_size=2, N=2)
    y_pred = torch.tensor([[1.0, 1.0, 0, 1.0, 0, 0, 0], [1.0, 0, 1.0, 0, 0, 1.0, 0]], dtype=torch.float32)
    # ==> on a bien des loss toujours nulles

    # y_pred = torch.tensor([[1.0, 0.8, 0.2, 0.7, 0.3, 0, 0], [1.0, 0.8, 0.2, 0.7, 0.1, 0.2, 0]], dtype=torch.float32)
    # on a:

    # y_pred = torch.tensor([[1.0, 0.6, 0.5, 0.5, 0.5, 0, 0], [1.0, 0.3, 0.7, 0.7, 0.1, 0.2, 0]], dtype=torch.float32)
    # on a:

    # Calcul de la perte en appelant explicitement forward
    loss = criterion.forward(y_pred, y_true)
    
    # # Vérification de la forme de sortie
    # assert loss.shape == torch.Size([2]), "La sortie de la perte doit être de dimension (batch_size,)"
    
    # Vérification des valeurs attendues (exemple avec valeurs pré-calculées si possible)
    assert torch.isfinite(loss).all(), "La perte ne doit contenir que des valeurs finies"

    print("loss_totale =", loss.item())

test_loss_forward()
