import json
import time

import torch
from timm.loss import HierarchicalCrossEntropy
from timm.loss.logicseg import multi_bce_loss

from scripts.hierarchy_better_mistakes_utils import *
from scripts.results import *
from scripts.utils import *
from scripts.logic_seg_utils import *

def test_hce():
    tree_filename = "data/small-collomboles/hierarchy_test.csv"
    tree_lines = read_csv(tree_filename)
    tree_lines_without_names = tree_lines[1:]
    
    tree = build_tree(tree_lines)
    print("============= Tree =============")
    print_tree(tree)
    print("================================")

    _, _, nodes_to_id, leafs_to_id = get_id_from_nodes(tree_lines_without_names)
    print("nodes:", nodes_to_id)    
    print("leafs:", leafs_to_id)

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

def test_modified_logiqseg_loss():
    tree_filename = "/mnt/c/Users/rubcr/OneDrive/Bureau/projet_long/pytorch-image-models-bees/scripts/data_test/hierarchy_test.csv"
    La_raw = get_layer_matrix(tree_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    La_raw = torch.tensor(La_raw).to(device)
    
    loss_fn = multi_bce_loss.MultiBCE(La_raw, 0.1)
    
    logits = torch.mul(torch.ones(2, 17), -10).to(device)
    target = torch.zeros(2, 17).to(device)
    logits[0,[0, 1, 2, 3, 4]] = 10
    target[0,[0, 1, 2, 3, 4]] = 1
    logits[1,[0, 1, 2, 3, 7]] = 10
    target[1,[0, 1, 2, 3, 4]] = 1

    y_pred = torch.sigmoid(logits)
    loss1 = loss_fn.forward(y_pred, target)
    loss2 = loss_fn.forward_old(y_pred, target)
    print("Valeur de la perte 1:", loss1.item())
    print("Valeur de la perte 2:", loss2.item())


def print_results():
    # filename_cce = "output/train/CCE-resnet50_a1_in1k/summary.csv"
    # filename_hce_0_1 = "output/train/HCE-a0.1-resnet50_a1_in1k/summary.csv"
    # filename_hce_0_5 = "output/train/HCE-a0.5-resnet50_a1_in1k/summary.csv"
    # show_results_from_csv_summary_cce_hce_alpha(filename_cce, filename_hce_0_1, filename_hce_0_5)
    filename_classes = "data/small-collomboles/class_mapping.txt"
    classes = load_classnames(filename_classes)

    hierarchy_filename = "data/small-collomboles/hierarchy.csv"
    hierarchy_lines = read_csv(hierarchy_filename)
    hierarchy_lines_without_names = hierarchy_lines[1:]
    parents = get_parents(hierarchy_lines_without_names)
    hierarchy_names = hierarchy_lines[0]
    hierarchy_names.reverse()
    output_folder_bce = "output/test/bce"
    output_folder_cce = "output/test/cce"
    output_folder_hce_0_1 = "output/test/hce_0_1"
    output_folder_hce_0_5 = "output/test/hce_0_5"
    save_confusion_matrix_and_metrics(output_folder_bce, "bce", classes, parents, hierarchy_names)
    save_confusion_matrix_and_metrics(output_folder_cce, "cce", classes, parents, hierarchy_names)
    save_confusion_matrix_and_metrics(output_folder_hce_0_1, "hce_0_1", classes, parents, hierarchy_names)
    save_confusion_matrix_and_metrics(output_folder_hce_0_5, "hce_0_5", classes, parents, hierarchy_names)

if __name__ == "__main__":
    # test_hce()
    # print_results()
    # test_modified_logiqseg_loss()
    display_models_summary("output/train")
    display_models_barplots("output/test", output_folder="output/img")