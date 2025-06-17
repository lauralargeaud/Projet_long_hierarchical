import csv
import sys
from collections import defaultdict
import numpy as np

from scripts.utils import read_csv

np.set_printoptions(threshold=sys.maxsize)

def get_id_from_nodes(hierarchy_lines):
    """
    Get Nodes and Leafs ID.

    Args:
        hierarchy_lines (string[]): lines of the hierarchy csv

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

def compute_L(hierarchy_lines, nodes_to_id, leafs_to_id):
    """
    Compute L matrix without leaves.

    Args:
        hierarchy_lines (string[]): lines of the hierarchy csv
        nodes_to_id (dict{}): nodes to id
        leafs_to_id (dict{}): leafs to id

    Returns:
        np.array: L matrix
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
    """
    Compute L matrix with leaves.

    Args:
        hierarchy_lines (string[]): lines of the hierarchy csv
        nodes_to_id (dict{}): nodes to id
        leafs_to_id (dict{}): leafs to id

    Returns:
        np.array: L matrix
    """
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

    Args:
        hierarchy_lines (string[]): lines of the hierarchy csv
        nodes_to_id (dict{}): nodes to id

    Returns:
        string[][]: path from leafs to root
    """
    paths = []
    for line in hierarchy_lines:
        path = []
        for node in line[:-1]:
            path.append(nodes_to_id[node])
        paths.append(path)
    return paths

def get_hce_tree_data(tree_filename):
    """
    Get L and h from the tree.

    Args:
        tree_filename (string): csv tree filename
    
    Returns:
        np.array: L matrix
        int: height of the tree
    """
    tree_lines = read_csv(tree_filename)
    tree_lines_without_names = tree_lines[1:]
    _, _, nodes_to_id, leafs_to_id = get_id_from_nodes(tree_lines_without_names)
    L = compute_full_L(tree_lines_without_names, nodes_to_id, leafs_to_id)
    h = len(tree_lines[0]) - 1
    return L, h