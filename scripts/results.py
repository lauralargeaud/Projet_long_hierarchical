import os
from unidecode import unidecode
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import normalize

from scripts.read_yaml import compute_model_name
from scripts.hierarchy_better_mistakes_utils import read_csv
from scripts.utils import read_csv, get_parents, get_taxon_level

def generate_barplots(values, labels, title, filename, output_folder="output/img"):
    """
    Generate barplot.

    Args:
        values (float[]): values to display
        output_folder (string[]): labels
        title (string): title of the plot
        filename (string): filename of the output image
        output_folder (string): output folder
    """
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Value')
    # plt.ylim([0, 1])
    plt.title(title)
    plt.xticks(rotation=45)
    
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * -0.10, f"{v:.3f}", ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(output_folder, exist_ok=True)
    
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

def generate_boxplots(values, labels, title, filename, output_folder="output/img"):
    """
    Generate boxplot.

    Args:
        values (float[]): values to display
        output_folder (string[]): labels
        title (string): title of the plot
        filename (string): filename of the output image
        output_folder (string): output folder
    """
    plt.figure(figsize=(10, 5))
    plt.boxplot(values, labels=labels)
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title(title)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    os.makedirs(output_folder, exist_ok=True)
    
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

def display_models_barplots_multiple(test_output_folder, output_folder="output/img", hierarchy_filename="data/small-collomboles/hierarchy.csv"):
    """
    Generates barplots for differents metrics for differentes models.

    Args:
        test_output_folder (string): output folder from inference
        output_folder (string): output folder to save images
        hierarchy_filename (string): filepath to hierachy csv
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    hierarchy_lines = read_csv(hierarchy_filename)
    hierarchy_names = hierarchy_lines[0]
    
    metrics = ["Précision", "Rappel", "F1-score"]
    labels = set()
    # Get all differentes labels
    for folder in os.listdir(test_output_folder):
        args_path = os.path.join(test_output_folder, folder, "args.yaml")
        title, _ = compute_model_name(args_path)
        labels.add(title)
    labels = sorted(list(labels))

    values = {metric: {hierarchy_name: {label: [] for label in labels} for hierarchy_name in hierarchy_names} for metric in metrics}
    acc_metrics = ["Top-1 Accuracy", "Top-5 Accuracy", "Top-1 hierarchical distance", "Top-5 hierarchical distance", "Hierarchical distance mistakes"]
    acc_values = {metric: {label: [] for label in labels} for metric in acc_metrics}
    # Get all data from csv files
    for folder in sorted(os.listdir(test_output_folder)):
        csv_path = os.path.join(test_output_folder, folder, "metrics_all.csv")
        args_path = os.path.join(test_output_folder, folder, "args.yaml")
        title, _ = compute_model_name(args_path)
        df = pd.read_csv(csv_path)
        # F1-Score, Precision and Recall
        for name in hierarchy_names:
            line = df[(df['Etage'] == name) & (df['Classe'] == 'Moyenne')]
            for metric in metrics:
                values[metric][name][title].append(line[metric].values[0])
        
        # Top-k metrics
        acc_path = os.path.join(test_output_folder, folder, "metrics_results.csv")
        df = pd.read_csv(acc_path)
        acc_values["Top-1 Accuracy"][title].append(df.iloc[0]["Top 1 accuracy"])
        acc_values["Top-5 Accuracy"][title].append(df.iloc[0]["Top 5 accuracy"])
        acc_values["Top-1 hierarchical distance"][title].append(df.iloc[0]["Top 1 hierarchical distance prediction"])
        acc_values["Top-5 hierarchical distance"][title].append(df.iloc[0]["Top 5 hierarchical distance prediction"])
        acc_values["Hierarchical distance mistakes"][title].append(df.iloc[0]["hierarchical distance mistakes"])
    
    # Save plots
    for metric, hierarchy in values.items():
        for name, data_dict in hierarchy.items():
            data_barplot = []
            data_boxplot = []
            for label in labels:
                data_barplot.append(np.mean(data_dict[label]))
                data_boxplot.append(data_dict[label])
            img_output_folder = os.path.join(output_folder, name, metric)
            generate_barplots(data_barplot, labels, f"{metric} {name}", f"{unidecode(metric).lower()}_{name}_barplot.png", output_folder=img_output_folder)
            generate_boxplots(data_boxplot, labels, f"{metric} {name}", f"{unidecode(metric).lower()}_{name}_boxplot.png", output_folder=img_output_folder)

    for name, data_dict in acc_values.items():
        data_barplot = []
        data_boxplot = []
        for label in labels:
            data_barplot.append(np.mean(data_dict[label]))
            data_boxplot.append(data_dict[label])
        img_output_folder = os.path.join(output_folder)
        generate_barplots(data_barplot, labels, f"{name}", f"{unidecode(name).lower()}_barplot.png", output_folder=img_output_folder)
        generate_boxplots(data_dict.values(), labels, f"{name}", f"{unidecode(name).lower()}_boxplot.png", output_folder=img_output_folder)

def generate_results_from_csv_summary(filepath, title, model_name, output_folder="output/img"):
    """
    Show results from summary csv produced by TIMM.

    Args:
        filepath (string): filepath of the csv
        title (string): title of the plot
        model_name (string): name of the model
        output_folder (string): output folder
    """
    os.makedirs(output_folder, exist_ok=True)

    data = pd.read_csv(filepath)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(data['epoch'], data['train_loss'], label='Train Loss', color='blue')
    ax1.plot(data['epoch'], data['eval_loss'], label='Eval Loss', color='red')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title(f'{title} Training and Evaluation Loss')
    ax1.grid()
    fig.savefig(os.path.join(output_folder, f'loss_summary_{model_name.lower().replace(" ", "_")}'))

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data['epoch'], data['eval_top1'], label='Eval Top-1 Accuracy', color='green')
    ax2.plot(data['epoch'], data['eval_top5'], label='Eval Top-5 Accuracy', color='lime')
    if 'train_top1' in data:
        ax2.plot(data['epoch'], data['train_top1'], label='Train Top-1 Accuracy', color='red')
        ax2.plot(data['epoch'], data['train_top5'], label='Train Top-5 Accuracy', color='orange')

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.set_ylim([0, 1])
    ax2.set_title(f"{title} Accuracy")
    ax2.grid()


    fig.savefig(os.path.join(output_folder, f'acc_summary_{model_name.lower().replace(" ", "_")}'))

def display_models_summary(train_output_folder, output_folder="output/img/summary"):
    """
    Display summary plot from train folder

    Args:
        train_output_folder (string): output folder from training
        output_folder (string): output folder to save images
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder in os.listdir(train_output_folder):
        summary_path = os.path.join(train_output_folder, folder, "summary.csv")
        args_path = os.path.join(train_output_folder, folder, "args.yaml")
        title, model_filename = compute_model_name(args_path)
        model_output_folder = os.path.join(output_folder, folder)
        generate_results_from_csv_summary(summary_path, title, model_filename, model_output_folder)

def load_confusion_matrix(filepath):
    """
    Load confusion matrix from a txt file.

    Args:
        filepath (string): filepath of the txt file
    
    Returns:
        np.array: confusion matrix
    """
    cm = np.loadtxt(filepath)
    cm = cm.astype(float)
    return cm

def save_confusion_matrix(cm, output_filename, classes, output_folder="output/img"):
    """
    Save confusion matrix image.

    Args:
        cm (np.array): confusion matrix
        output_filename (string): image filename
        classes (string[]): list of the classes
        output_folder (string): output folder to save the image

    Returns:
        np.array: confusion matrix
    """
    figsize = len(classes) / 2 if len(classes) > 10 else 5
    plt.figure(figsize=(figsize, figsize))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.title("Matrice de Confusion")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, output_filename))
    
    return cm

def calculate_metrics(cm):
    """
    Compute F1-score, Precision and Recall from a confusion matrix.

    Args:
        cm (np.array): confusion matrix

    Returns:
        float: precision
        float: recall
        float: f1_score
        float: tot_pred
        float: tot_true
        float: TP
        float: FP
        float: FN
    """
    TP = np.diag(cm)    
    FP = np.sum(cm, axis=0) - TP    
    FN = np.sum(cm, axis=1) - TP    
    precision = np.divide(TP, TP + FP, where=(TP + FP) != 0)
    recall = np.divide(TP, TP + FN, where=(TP + FN) != 0)    
    f1_score = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    tot_pred = np.sum(cm, axis=0)    
    tot_true = np.sum(cm, axis=1)
    return precision, recall, f1_score, tot_pred, tot_true, TP, FP, FN

def save_metrics(cm, output_folder, output_filename, classes, hierarchy_name):
    """
    Save metrics from a confusion matrix in a file.

    Args:
        cm (np.array): confusion matrix
        output_folder (string): output folder to save the image
        output_filename (string): image filename
        classes (string[]): list of the classes
        hierarchy_name (string): name of the hierarchy level

    Returns:
        pd.DataFrame: dataframe with metrics
    """
    precision, recall, f1_score, tot_pred, tot_true, TP, FP, FN = calculate_metrics(cm)

    # Création d'un DataFrame Pandas
    df = pd.DataFrame({
        "Classe": classes, 
        "Etage": [hierarchy_name for i in range(len(classes))],
        "Pred": tot_pred,
        "True": tot_true,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Précision": precision, 
        "Rappel": recall,
        "F1-score": f1_score
    })

    # Ajout des moyennes globales
    df.loc["Moyenne Macro"] = ["Moyenne", hierarchy_name, np.mean(tot_pred), np.mean(tot_true), np.mean(TP), np.mean(FP), np.mean(FN), np.mean(precision), np.mean(recall), np.mean(f1_score)]

    # Sauvegarde en CSV
    df.to_csv(os.path.join(output_folder, output_filename), index=False)
    return df

def get_id_from_nodes(hierarchy_lines):
    """
    Get Nodes and Leafs ID.

    Args:
        hierarchy_lines (string[]): list of lines of the hierarchy csv

    Returns:
        string[]: nodes
        string[]: leafs
        dict{}: nodes to id
        dict{}: leafs to id
    """
    h = len(hierarchy_lines[0])
    nodes = []
    for i in range(h-1):
        for line in hierarchy_lines:
            if line[i] not in nodes:
                nodes.append(line[i])
    leafs = []
    for line in hierarchy_lines:
        leafs.append(line[h-1])
    nodes_to_id = {node: i for i, node in enumerate(nodes)}
    leafs_to_id = {leaf: i for i, leaf in enumerate(leafs)}
    return nodes, leafs, nodes_to_id, leafs_to_id

def get_parent_confusion_matrix(cm, classes, parents):
    """
    Get parent confusion matrix from current classes.

    Args:
        cm (np.array): confusion matrix
        classes (string[]): list of classes
        parents (dict{}): dict to get parents from children

    Returns:
        np.array: parent confusion matrix
        string[]: list of parents
    """
    next_classes = set()
    for class_ in classes:
        next_classes.add(parents[class_])
    next_classes = list(next_classes)
    next_classes_id = {class_: i for i, class_ in enumerate(next_classes)}
    
    next_cm = np.zeros((len(next_classes),len(next_classes)), dtype=int)
    for i, class_1 in enumerate(classes):
        for j, class_2 in enumerate(classes):
            next_class_1 = parents[class_1]
            next_class_2 = parents[class_2]
            next_class_1_id = next_classes_id[next_class_1]
            next_class_2_id = next_classes_id[next_class_2]
            next_cm[next_class_1_id, next_class_2_id] += cm[i,j]

    return next_cm, next_classes

# df contient les données de metrics_all.csv
# on veut construire le csv requis par plot_hierarchical_perf
def build_F1_perfs_csv(metrics_filepath, output_filepath, hierarchy_filepath):
    """
    Build F1-score in csv file.

    Args:
        metrics_filepath (string): path to metrics csv
        output_filepath (string): output filepath
        hierarchy_filepath (string): path to hierarchy csv
    """
    df = pd.read_csv(metrics_filepath)

    df_filtered = df[df["Etage"] != "branches"]
    df_filtered = df_filtered[df_filtered["Classe"] != "Moyenne"]
    
    # Construire le nouveau DataFrame
    new_df = pd.DataFrame({
        "Taxon_level": None,  # non renseigné
        "Name": df_filtered["Classe"],
        "Parent": None,  # Parent non renseigné
        "Count": df_filtered["True"],
        "F1-score": df_filtered["F1-score"]
    })

    # Trier par ordre alphabétique des 'Name'
    new_df = new_df.sort_values(by=["Name"])

    hierarchy_filename = hierarchy_filepath
    hierarchy_lines = read_csv(hierarchy_filename)
    hierarchy_lines_without_names = hierarchy_lines[1:]
    parents = get_parents(hierarchy_lines_without_names)
    new_df["Parent"] = new_df["Name"].map(parents)

    taxon_levels = get_taxon_level(hierarchy_lines)
    taxon_levels = dict(sorted(taxon_levels.items()))
    new_df["Taxon_level"] = new_df["Name"].map(taxon_levels)

    new_df.to_csv(output_filepath, index=False)

def save_confusion_matrix_and_metrics(output_folder, cm_leaves_path, classes, parents, hierarchy_names):
    """
    Save confusion matrix and metrics for each layer in files.
    """
    cm_leaves = load_confusion_matrix(cm_leaves_path)
    save_confusion_matrix(cm_leaves, f"confusion_matrix_{hierarchy_names[0]}.png", classes, folder=output_folder)
    df = save_metrics(cm_leaves, output_folder, f"metrics_{hierarchy_names[0]}.csv", classes, hierarchy_names[0])
    next_cm = cm_leaves
    next_classes = classes
    for i in range(1, len(hierarchy_names)):
        next_cm, next_classes = get_parent_confusion_matrix(next_cm, next_classes, parents)
        next_cm_norm = normalize(next_cm, axis=1, norm='l1')
        save_confusion_matrix(next_cm, f"confusion_matrix_{hierarchy_names[i]}.png", next_classes, folder=output_folder)
        save_confusion_matrix(next_cm_norm, f"confusion_matrix_norm_{hierarchy_names[i]}.png", next_classes, folder=output_folder)
        next_df = save_metrics(next_cm, output_folder, f"metrics_{hierarchy_names[i]}.csv", next_classes, hierarchy_names[i])
        df = pd.concat([df, next_df])
    
    df.to_csv(os.path.join(output_folder, "metrics_all.csv"), index=False)
    tree = create_tree_json(df, parents)
    with open(os.path.join(output_folder, "tree.json"), "w") as outfile: 
        json.dump(tree, outfile)

def create_tree_json(df, parents):
    """
    Create a tree with metrics from a dataframe.

    Args:
        df (pd.DataFrame): csv metrics
        parents (dict{}): dict to get parents from children

    Returns:
        dict{}: root of the tree
    """
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
    
def create_tree(df, name, parent, childrens):
    """
    Create a tree with metrics from a dataframe.

    Args:
        df (pd.DataFrame): csv metrics
        name (string): name of the current node
        parent (dict{}): parent of the current node
        childrens (dict{}): dict to get childrens from parent
    """
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
    parent["children"].append(node)
    if name in childrens:
        for child in childrens[name]:
            create_tree(df, child, node, childrens)