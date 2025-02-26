import os
from unidecode import unidecode

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv

from scripts.read_yaml import compute_model_name
from scripts.hierarchy_better_mistakes_utils import read_csv

def generate_barplots(values, labels, title, filename, folder="output/img/"):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Catégories')
    plt.ylabel('Valeurs')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, filename))


def display_models_barplots(test_output_folder, output_folder="output/img", hierarchy_filename="data/small-collomboles/hierarchy.csv"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    hierarchy_lines = read_csv(hierarchy_filename)
    hierarchy_names = hierarchy_lines[0]
    
    metrics = ["Précision", "Rappel", "F1-score"]
    values = {metric: {hierarchy_name: [] for hierarchy_name in hierarchy_names} for metric in metrics}
    labels = []
    for folder in os.listdir(test_output_folder):
        csv_path = os.path.join(test_output_folder, folder, "metrics_all.csv")
        args_path = os.path.join(test_output_folder, folder, "args.yaml")
        title, _ = compute_model_name(args_path)
        labels.append(title)

        df = pd.read_csv(csv_path)
        for name in hierarchy_names:
            line = df[(df['Etage'] == name) & (df['Classe'] == 'Moyenne')]
            for metric in metrics:
                values[metric][name].append(line[metric].values[0])
    for metric, hierarchy in values.items():
        for name, data in hierarchy.items():
            generate_barplots(data, labels, f"{metric} {name}", f"{unidecode(metric).lower()}_{name}.png")

def show_results_from_csv_summary(filename, title, model_name, folder="output/img"):
    """
    Show results from summary csv produced by TIMM.
    """
    data = pd.read_csv(filename)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(data['epoch'], data['train_loss'], label='Train Loss', color='blue')
    ax1.plot(data['epoch'], data['eval_loss'], label='Eval Loss', color='red')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title(f'{title} Training and Evaluation Loss')
    ax1.grid()
    fig.savefig(os.path.join(folder, f'loss_summary_{model_name.lower().replace(" ", "_")}'))

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

    fig.savefig(os.path.join(folder, f'acc_summary_{model_name.lower().replace(" ", "_")}'))

def show_results_from_csv_summarys(filename1, filename2, model_name1, model_name2, folder="output/img"):
    """
    Show results from summary csv produced by TIMM with 2 models.
    """
    data1 = pd.read_csv(filename1)
    data2 = pd.read_csv(filename2)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(data1['epoch'], data1['train_loss'], label=f'{model_name1} Train Loss', color='red', linestyle='--')
    ax1.plot(data1['epoch'], data1['eval_loss'], label=f'{model_name1} Eval Loss', color='salmon')
    ax1.plot(data2['epoch'], data2['train_loss'], label=f'{model_name2} Train Loss', color='green', linestyle='--')
    ax1.plot(data2['epoch'], data2['eval_loss'], label=f'{model_name2} Eval Loss', color='lime')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Evaluation Loss")
    ax1.grid()
    fig.savefig(os.path.join(folder, f'loss_summary_{model_name1.lower().replace(" ", "_")}_{model_name2.lower().replace(" ", "_")}'))

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data1['epoch'], data1['eval_top1'], label=f'{model_name1} Eval Top-1 Accuracy', color='red')
    ax2.plot(data1['epoch'], data1['eval_top5'], label=f'{model_name1} Eval Top-5 Accuracy', color='salmon')
    ax2.plot(data2['epoch'], data2['eval_top1'], label=f'{model_name2} Eval Top-1 Accuracy', color='green')
    ax2.plot(data2['epoch'], data2['eval_top5'], label=f'{model_name2} Eval Top-5 Accuracy', color='lime')
    if 'train_top1' in data1 and 'train_top1' in data2:
        ax2.plot(data1['epoch'], data1['eval_top1'], label=f'{model_name1} Train Top-1 Accuracy', color='orange')
        ax2.plot(data1['epoch'], data1['eval_top5'], label=f'{model_name1} Train Top-5 Accuracy', color='wheat')
        ax2.plot(data2['epoch'], data2['eval_top1'], label=f'{model_name2} Train Top-1 Accuracy', color='blue')
        ax2.plot(data2['epoch'], data2['eval_top5'], label=f'{model_name2} Train Top-5 Accuracy', color='lightblue')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.set_ylim([0, 1])
    ax2.set_title("Accuracy")
    ax2.grid()
    fig.savefig(os.path.join(folder, f'acc_summary_{model_name1.lower().replace(" ", "_")}_{model_name2.lower().replace(" ", "_")}'))

    plt.show()

def display_models_summary(train_output_folder, output_folder="output/img"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder in os.listdir(train_output_folder):
        summary_path = os.path.join(train_output_folder, folder, "summary.csv")
        args_path = os.path.join(train_output_folder, folder, "args.yaml")
        title, model_filename = compute_model_name(args_path)
        show_results_from_csv_summary(summary_path, title, model_filename, output_folder)

def load_confusion_matrix(filename):
    """
    Load confusion matrix from a txt file.
    """
    cm = np.loadtxt(filename)
    cm = cm.astype(float)
    return cm

def save_confusion_matrix(cm, output_filename, classes, folder="output/img"):
    """
    Save confusion matrix image.
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
    plt.savefig(os.path.join(folder, output_filename))
    
    return cm

def load_classnames(filename):
    """
    Load classnames from a file.
    """
    classes = []
    with open(filename, 'r') as f:
        data = f.readlines()
        for line in data:
            classes.append(line.replace('\n', ''))
    return classes

def calculate_metrics(cm):
    """
    Compute F1-score, Precision and Recall from a confusion matrix.
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

def save_metrics(cm, folder, filename, classes, hierarchy_name):
    """
    Save metrics from a confusion matrix in a file.
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
    df.to_csv(os.path.join(folder, filename), index=False)
    return df

def get_id_from_nodes(hierarchy_lines):
    """
    Get Nodes and Leafs ID.
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


def read_csv(filename):
    """
    Read CSV.
    """
    lines = []
    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lines.append(row)
    return lines

def get_parents(hierarchy_lines):
    parents = {}
    for line in hierarchy_lines:
        for i, node in enumerate(line[1:]):
            parents[node] = line[i]
    return parents

def get_taxonLevel(hierarchy_lines):
    taxon_levels = {}
    for line in hierarchy_lines[1:]:
        for i, node in enumerate(line):
            taxon_levels[node] = hierarchy_lines[0][i]
    return taxon_levels

# df contient les données de metrics_all.csv
# on veut construire le csv requis par plot_hierarchical_perf
def build_F1_perfs_csv(df, path_generated_csv, path_hierarchy):

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

    hierarchy_filename = path_hierarchy
    hierarchy_lines = read_csv(hierarchy_filename)
    hierarchy_lines_without_names = hierarchy_lines[1:]
    parents = get_parents(hierarchy_lines_without_names)
    new_df["Parent"] = new_df["Name"].map(parents)

    taxon_levels = get_taxonLevel(hierarchy_lines)
    taxon_levels = dict(sorted(taxon_levels.items()))
    new_df["Taxon_level"] = new_df["Name"].map(taxon_levels)

    new_df.to_csv(path_generated_csv, index=False)