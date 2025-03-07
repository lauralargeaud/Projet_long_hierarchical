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
    criterion = LogicSegLoss("bce", H_raw, P_raw, M_raw, [], alpha_c, alpha_d, alpha_e, alpha_bce, [])
    
    y_true = torch.tensor([[1.0, 1.0, 0.01, 1.0, 0.01, 0.01, 0.01], [1.0, 0.01, 1.0, 0.01, 0.01, 1.0, 0.01]], dtype=torch.float32)


    # cas de prédictions parfaites
    y_pred = torch.tensor([[1.0, 1.0, 0.01, 1.0, 0.01, 0.01, 0.01], [1.0, 0.01, 1.0, 0.01, 0.01, 1.0, 0.01]], dtype=torch.float32)
    print("Cas de préditions parfaites:")
    loss = criterion.forward(y_pred, y_true, True)

    # En dégradant les prédictions pour Lc 
    y_pred = torch.tensor([[0.99, 0.99, 0.01, 0.99, 0.01, 0.01, 0.01], [0.99, 0.01, 0.99, 0.99, 0.01, 0.99, 0.01]], dtype=torch.float32)
    print("Dégradation Lc:")
    loss = criterion.forward(y_pred, y_true, True)

    # En dégradant les prédictions pour Ld
    y_pred = torch.tensor([[0.99, 0.99, 0.01, 0.99, 0.01, 0.01, 0.01], [0.99, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01]], dtype=torch.float32)
    print("Dégradation Ld:")
    loss = criterion.forward(y_pred, y_true, True)

    # En dégradant les prédictions pour Le
    y_pred = torch.tensor([[0.99, 0.99, 0.01, 0.99, 0.01, 0.01, 0.01], [0.99, 0.01, 0.99, 0.01, 0.01, 0.99, 0.99]], dtype=torch.float32)
    print("Dégradation Le:")
    loss = criterion.forward(y_pred, y_true, True)

    # Vérification des valeurs attendues (exemple avec valeurs pré-calculées si possible)
    # assert torch.isfinite(loss).all(), "La perte ne doit contenir que des valeurs finies"

    # print("loss_totale =", loss.item())

test_loss_forward()
