from scripts.hierarchy_better_mistakes_utils import *
from timm.loss.hierarchical_cross_entropy import HierarchicalCrossEntropy
import torch
import time

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

    hxe_loss = HierarchicalCrossEntropy(L, alpha=0.1, h=len(hierarchy_lines[0])-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.mul(torch.ones(4, 9), -10).to(device)
    logits[0, 0] = 10
    logits[1, 1] = 10
    logits[2, 0] = 5
    logits[2, 1] = 5
    logits[2, 2] = 5
    logits[3, 3] = -10
    logits[3, 7] = 10
    logits[3, 8] = -10

    targets = torch.tensor([0, 8, 1, 8]).to(device)
    # Initialisation et calcul de la perte
    start_time = time.time()
    loss_value = hxe_loss.forward_with_for_loop(logits, targets)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Valeur de la perte HXE (for loop) :", loss_value.item())
    print("=======================================================")
    start_time = time.time()
    loss_value = hxe_loss.forward(logits, targets)
    print("--- %s seconds ---" % (time.time() - start_time))
# 
    print("Valeur de la perte HXE :", loss_value.item())



