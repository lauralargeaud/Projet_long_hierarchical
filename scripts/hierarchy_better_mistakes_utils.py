import csv
import sys
from collections import defaultdict
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

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

def build_tree(data):
    """
    Build tree.
    """
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    names = set()
    for entry in data[1:]:
        class_, order, family, genus, species = entry
        names.add(class_)
        names.add(order)
        names.add(family)
        names.add(genus)
        names.add(species)
        tree[class_][order][family][genus].append(species)

    return tree

def print_tree(tree, indent=0):
    """
    Print tree.
    """
    for key, value in tree.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            print_tree(value, indent + 4)
        else:
            for item in value:
                print(" " * (indent + 4) + str(item))

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

def compute_L(hierarchy_lines, nodes_to_id, leafs_to_id):
    """
    Compute L matrix.
    """
    h = len(hierarchy_lines[0])
    L = np.zeros((len(nodes_to_id), len(leafs_to_id)))
    for line in hierarchy_lines:
        leaf_id = leafs_to_id[line[h-1]]
        for node in line[:-1]:
            node_id = nodes_to_id[node]
            L[node_id, leaf_id] = 1
    return L

def compute_full_L(hierarchy_lines, nodes_to_id, leafs_to_id):
    h = len(hierarchy_lines[0])
    L = np.zeros((len(nodes_to_id)+len(leafs_to_id), len(leafs_to_id)))
    for line in hierarchy_lines:
        leaf_id = leafs_to_id[line[h-1]]
        for node in line[:-1]:
            node_id = nodes_to_id[node]
            L[node_id, leaf_id] = 1
        L[leaf_id+len(nodes_to_id),leaf_id] = 1
    return L

def get_path_from_leafs(hierarchy_lines, nodes_to_id):
    """
    Get path from each leafs to root.
    """
    paths = []
    for line in hierarchy_lines:
        path = []
        for node in line[:-1]:
            path.append(nodes_to_id[node])
        paths.append(path)
    return paths

def get_parents(hierarchy_lines):
    parents = {}
    for line in hierarchy_lines:
        for i, node in enumerate(line[1:]):
            parents[node] = line[i]
    return parents


def get_hce_tree_data(tree_filename):
    tree_lines = read_csv(tree_filename)
    tree_lines_without_names = tree_lines[1:]
    _, _, nodes_to_id, leafs_to_id = get_id_from_nodes(tree_lines_without_names)
    L = compute_full_L(tree_lines_without_names, nodes_to_id, leafs_to_id)
    h = len(tree_lines[0]) - 1
    return L, h

if __name__ == "__main__":
    hierarchy_filename = "data/small-collomboles/hierarchy_test.csv"
    hierarchy_lines = read_csv(hierarchy_filename)
    hierarchy_lines_without_names = hierarchy_lines[1:]
    
    tree = build_tree(hierarchy_lines)
    print("============= Tree =============")
    print_tree(tree)
    print("================================")

    nodes, leafs, nodes_to_id, leafs_to_id = get_id_from_nodes(hierarchy_lines_without_names)
    print("nodes:", nodes_to_id)    
    print("leafs:", leafs_to_id)

    L = compute_L(hierarchy_lines_without_names, nodes_to_id, leafs_to_id)
    paths = get_path_from_leafs(hierarchy_lines_without_names, nodes_to_id)
    for i in range(len(L)):
        print(L[i,:], nodes[i])