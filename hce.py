import json
import time

import torch

from scripts.hierarchy_better_mistakes_utils import *
from scripts.hce_results import *
from scripts.utils import *
from timm.loss.hierarchical_cross_entropy import HierarchicalCrossEntropy

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
    output_folder_cce = "output/test/cce"
    output_folder_hce_0_1 = "output/test/hce_0_1"
    output_folder_hce_0_5 = "output/test/hce_0_5"
    save_confusion_matrix_and_metrics(output_folder_cce, "cce", classes, parents, hierarchy_names)
    save_confusion_matrix_and_metrics(output_folder_hce_0_1, "hce_0_1", classes, parents, hierarchy_names)
    save_confusion_matrix_and_metrics(output_folder_hce_0_5, "hce_0_5", classes, parents, hierarchy_names)

def save_confusion_matrix_and_metrics(output_folder, name, classes, parents, hierarchy_names):
    filename_cm_leaves = os.path.join(output_folder, "confusion_matrix.out")
    cm_leaves = load_confusion_matrix(filename_cm_leaves)
    save_confusion_matrix(cm_leaves, f"confusion_matrix_{hierarchy_names[0]}.png", classes, folder=output_folder)
    df = save_metrics(cm_leaves, output_folder, f"metrics_{hierarchy_names[0]}.csv", classes, hierarchy_names[0])
    next_cm = cm_leaves
    next_classes = classes
    for i in range(1, len(hierarchy_names)):
        next_cm, next_classes = get_parent_confusion_matrix(next_cm, next_classes, parents)
        save_confusion_matrix(next_cm, f"confusion_matrix_{hierarchy_names[i]}.png", next_classes, folder=output_folder)
        next_df = save_metrics(next_cm, output_folder, f"metrics_{hierarchy_names[i]}.csv", next_classes, hierarchy_names[i])
        df = pd.concat([df, next_df])
    
    df.to_csv(os.path.join(output_folder, "metrics_all.csv"), index=False)
    tree = create_tree_json(df, parents)
    with open(os.path.join(output_folder, "tree.json"), "w") as outfile: 
        json.dump(tree, outfile)

def create_tree_json(df, parents):
    childrens = {}
    for k, v in parents.items():
        if v not in childrens:
            childrens[v] = [k]
        else:
            childrens[v].append(k)

    root_row = df.iloc[-2]
    root = {
        "name": root_row["Classe"], 
        "pred": root_row["Pred"], 
        "true": root_row["True"], 
        "tp": root_row["TP"], 
        "fp": root_row["FP"], 
        "fn": root_row["FN"], 
        "precision": root_row["Précision"], 
        "recall": root_row["Rappel"], 
        "f1-score": root_row["F1-score"], 
        "children": []
    }
    for child in childrens[root_row["Classe"]]:
        create_tree(df, child, root, childrens)
    return root
    
def create_tree(df, name, root, childrens):
    row = df[df["Classe"] == name]
    node = {
        "name": row["Classe"].values[0], 
        "pred": row["Pred"].values[0], 
        "true": row["True"].values[0], 
        "tp": row["TP"].values[0], 
        "fp": row["FP"].values[0], 
        "fn": row["FN"].values[0], 
        "precision": row["Précision"].values[0], 
        "recall": row["Rappel"].values[0], 
        "f1-score": row["F1-score"].values[0], 
        "children": []
    }
    root["children"].append(node)
    if name in childrens:
        for child in childrens[name]:
            create_tree(df, child, node, childrens)

if __name__ == "__main__":
    # test_hce()
    # print_results()

    dirname = "data/small-collomboles_small/dataset/train"
    keep_only_first_image(dirname)
