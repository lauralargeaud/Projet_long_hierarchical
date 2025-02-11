from scripts.hierarchy_better_mistakes_utils import *
from timm.loss.hierarchical_cross_entropy import HierarchicalCrossEntropy
import torch

def test(L, logits, targets, lam):
    print("============= Logits =============")
    print("logits", logits)
    print("============= Target =============")
    print("targets", targets)
    print("============= L * logits =============")
    LS = L @ torch.transpose(logits, 0, 1)
    print("LS", LS)
    print("============= Poids Arbre =============")
    Ltarget = L[:,targets]
    print("Ltargets", Ltarget)
    print("LS.T", torch.transpose(LS, 0, 1))
    P = torch.mul(Ltarget, torch.transpose(LS, 0, 1))
    print("P", P)
    PP = P[torch.unsqueeze(Ltarget, 0) != 0]
    print("PP", PP)
    print("============= Num Den =============")
    num = PP[1:]
    den = PP[:-1]
    print(num)
    print(den)
    print("============= Log =============")
    log = - torch.log(num / den)
    print(log)
    print("============= Lam =============")
    term_sum = lam * log
    print(term_sum)
    print("============= Loss =============")
    print(f"loss = {torch.sum(term_sum)}")
    print("================================")
    print("================================")


if __name__ == "__main__":
    hierarchy_filename = "data/small-collomboles/hierarchy_test.csv"
    hierarchy_lines = read_csv(hierarchy_filename)
    hierarchy_lines_without_names = hierarchy_lines[1:]
    
    tree = build_tree(hierarchy_lines)
    print("============= Tree =============")
    print_tree(tree)
    print("================================")

    nodes, leafs, nodes_to_id, leafs_to_id = get_id_from_nodes(hierarchy_lines_without_names)
    nodes_leafs = nodes + leafs
    print("nodes:", nodes_to_id)    
    print("leafs:", leafs_to_id)

    paths = get_path_from_leafs(hierarchy_lines_without_names, nodes_to_id)
    L = compute_full_L(hierarchy_lines_without_names, nodes_to_id, leafs_to_id)
    # print("============= L =============")
    # for i in range(len(L)):
    #     print(L[i,:], nodes_leafs[i])

    # alpha = torch.tensor(0.1)
    # n = torch.tensor([i for i in range(len(hierarchy_lines[0])-1, 0, -1)])
    # print("============= Lam =============")
    # lam = torch.exp(-2*n*alpha)
    # print(n)
    # print(lam)

    # L = torch.tensor(L, dtype=torch.float32)
    
    # logits = torch.zeros(1, 9, dtype=torch.float32)
    # logits[0,0] = 0.9
    # logits[0,1] = 0.0
    # logits[0,3] = 0.1
    # targets = torch.tensor(0, dtype=torch.int)
    # test(L, logits, targets, lam)


    # logits = torch.zeros(1, 9, dtype=torch.float32)
    # logits[0,0] = 0.8
    # logits[0,1] = 0.1
    # logits[0,8] = 0.000001
    # targets = torch.tensor(8, dtype=torch.int)
    # test(L, logits, targets, lam)

    hxe_loss = HierarchicalCrossEntropy(L, alpha=0.1, h=len(hierarchy_lines[0])-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.mul(torch.ones(4, 9), -10).to(device)
    logits[0, 0] = 10
    logits[1, 0] = 10
    logits[2, 0] = 5
    logits[2, 1] = 5
    logits[2, 2] = 5
    logits[3, 3] = -10
    logits[3, 7] = 10
    logits[3, 8] = -10

    targets = torch.tensor([0, 8, 1, 8]).to(device)
    # Initialisation et calcul de la perte
    loss_value = hxe_loss.forward(logits, targets)
# 
    print("Valeur de la perte HXE :", loss_value.item())



