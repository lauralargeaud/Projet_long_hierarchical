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

def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
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
    names = list(names)
    # names.sort()
    return tree, names

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

def get_class_to_id(data):
    """
    Get an id for each class (only leaf).
    """
    class_to_id = {}
    for i, class_ in enumerate(data):
        classname = class_.replace("\n", "")
        class_to_id[classname] = i
    return class_to_id

def get_tree_elem_to_id(data):
    """
    Get an id for each element in the tree.
    """
    tree_elem_to_id = {}
    # data.sort()
    for i, d in enumerate(data):
        tree_elem_to_id[d] = i
    return tree_elem_to_id

def get_leaves_from_nodes(tree):
    """
    Get all leaves from a nodes in the tree.
    """
    leaves = {}
    get_leaves(tree, leaves)
    tree_elem_to_id = get_tree_elem_to_id(list(leaves.keys()))
    return leaves, tree_elem_to_id

def get_leaves(tree, leaves_dict):
    """
    Get leaves from a node recursively.
    """
    all_leaves = []
    for key, value in tree.items():
        leaves_dict[key] = []
        if isinstance(value, dict):
            leaves = get_leaves(value, leaves_dict)
        else:
            for v in value:
                leaves_dict[v] = []
            leaves = value
        leaves_dict[key] = leaves_dict[key] + leaves
        all_leaves = all_leaves + leaves
    return all_leaves

def compute_L(leaves_from_nodes, tree_elem_to_id, class_to_id):
    class_names = class_to_id.keys()
    node_names = [hname for hname in tree_elem_to_id.keys() if hname not in class_names]
    # node_names.sort()
    node_names_to_id = {name: i for i, name in enumerate(node_names)}
    L = np.zeros((len(node_names), len(class_names)))
    for node_name, node_id in node_names_to_id.items():
        leaves = leaves_from_nodes[node_name]
        for leave in leaves:
            leaf_id = class_to_id[leave]
            L[node_id, leaf_id] = 1
    return L, node_names_to_id


if __name__ == "__main__":
    hierarchy_filename = "data/small-collomboles/hierarchy_test.csv"
    hierarchy_lines = read_csv(hierarchy_filename)
    tree, names = build_tree(hierarchy_lines)
    print("============= Tree =============")
    print_tree(tree)
    print("================================")
    class_filename = "data/small-collomboles/class_mapping_test.txt"
    class_lines = read_file(class_filename)
    class_to_id = get_class_to_id(class_lines)

    leaves_from_nodes, tree_elem_to_id = get_leaves_from_nodes(tree)

    L, node_names_to_id = compute_L(leaves_from_nodes, tree_elem_to_id, class_to_id)
    print(class_to_id)
    for name, i in node_names_to_id.items():
        print(L[i,:], name)
    print("============= Leaves from nodes =============")
    print("L = ")
    print(L)
    print("=============================================")
    