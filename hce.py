from scripts.hierarchy_better_mistakes_utils import *
from scripts.hce_results import *
from timm.loss.hierarchical_cross_entropy import HierarchicalCrossEntropy
import torch
import time

def test_hce():
    tree_filename = "data/small-collomboles/hierarchy_test.csv"
    tree_lines = read_csv(tree_filename)
    tree_lines_without_names = tree_lines[1:]
    
    tree = build_tree(tree_lines)
    print("============= Tree =============")
    print_tree(tree)
    print("================================")

    nodes, leafs, nodes_to_id, leafs_to_id = get_id_from_nodes(tree_lines_without_names)
    print("nodes:", nodes_to_id)    
    print("leafs:", leafs_to_id)

    paths = get_path_from_leafs(tree_lines_without_names, nodes_to_id)
    L = compute_full_L(tree_lines_without_names, nodes_to_id, leafs_to_id)

    hxe_loss = HierarchicalCrossEntropy(L, alpha=0.1, h=len(tree_lines[0])-1)
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

def print_result():
    # filename_cce = "output/train/CCE-resnet50_a1_in1k/summary.csv"
    # filename_hce_0_1 = "output/train/HCE-a0.1-resnet50_a1_in1k/summary.csv"
    # filename_hce_0_5 = "output/train/HCE-a0.5-resnet50_a1_in1k/summary.csv"
    # show_results_from_csv_summary_cce_hce_alpha(filename_cce, filename_hce_0_1, filename_hce_0_5)
    output_folder_cce = "output/test/cce"
    output_folder_hce_0_1 = "output/test/hce_0_1"
    output_folder_hce_0_5 = "output/test/hce_0_5"
    filename_cce = os.path.join(output_folder_cce, "confusion_matrix.out")
    filename_hce_0_1 = os.path.join(output_folder_hce_0_1, "confusion_matrix.out")
    filename_hce_0_5 = os.path.join(output_folder_hce_0_5, "confusion_matrix.out")
    filename_classes = "data/small-collomboles/class_mapping.txt"
    classes = load_classnames(filename_classes)
    cm_cce = load_confusion_matrix(filename_cce)
    cm_hce_0_1 = load_confusion_matrix(filename_hce_0_1)
    cm_hce_0_5 = load_confusion_matrix(filename_hce_0_5)
    save_confusion_matrix(cm_cce, "confusion_matrix_cce.png", classes)
    save_confusion_matrix(cm_hce_0_1, "confusion_matrix_hce_0_1.png", classes)
    save_confusion_matrix(cm_hce_0_5, "confusion_matrix_hce_0_5.png", classes)

    cm_cce = save_metrics(cm_cce, output_folder_cce, classes)
    cm_hce_0_1 = save_metrics(cm_hce_0_1, output_folder_hce_0_1, classes)
    cm_hce_0_5 = save_metrics(cm_hce_0_5, output_folder_hce_0_5, classes)


if __name__ == "__main__":
    # test_hce()
    print_result()

