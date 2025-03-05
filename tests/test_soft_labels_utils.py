import torch
import pandas as pd
import numpy as np

from script.soft_labels_utils import (
    build_index_nodes,
    map_and_vectorize_hierarchy,
    build_hierarchy_tensors,
    generate_index_pairs,
    find_lca_index,
    compute_lca_matrix,
    compute_soft_labels
)
data_path = "./tests/subset_hierarchy.csv"
###############################################################################
# TEST : build_index_nodes
###############################################################################
def test_build_index_nodes():
    """
    Teste la création des dictionnaires (internal_node_to_index, leaf_to_index)
    et le tenseur des hauteurs (internal_nodes_heights).
    """
    # Chargement du CSV et définition des niveaux
    df_hierarchy = pd.read_csv(data_path)
    hierarchy_levels = ["species", "genus", "family", "order", "class"]

    # Exécution
    internal_node_to_index, leaf_to_index, internal_nodes_heights = build_index_nodes(
        df_hierarchy, hierarchy_levels
    )

    # Valeurs attendues
    expected_leaf_to_index = {
        'Allacma fusca': 0,
        'Anurida maritima': 1,
        'Bilobella aurantiaca': 2,
        'Bilobella braunerae': 3,
        'Bourletiella arvalis': 4,
        'Bourletiella hortensis': 5,
        'Brachystomella parvula': 6,
        'Caprainea marginata': 7,
        'Ceratophysella denticulata': 8,
        'Ceratophysella longispina': 9
    }
    expected_internal_node_to_index = {
        'Allacma': 0,
        'Anurida': 1,
        'Bilobella': 2,
        'Bourletiella': 3,
        'Brachystomella': 4,
        'Caprainea': 5,
        'Ceratophysella': 6,
        'Bourletiellidae': 7,
        'Brachystomellidae': 8,
        'Hypogastruridae': 9,
        'Neanuridae': 10,
        'Sminthuridae': 11,
        'Poduromorpha': 12,
        'Symphypleona': 13,
        'Entognatha': 14
    }
    expected_internal_nodes_heights = torch.tensor([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0, 2.0,
        3.0, 3.0,
        4.0
    ])

    # Vérifications
    assert internal_node_to_index == expected_internal_node_to_index, \
        f"internal_node_to_index incorrect. Obtenu : {internal_node_to_index}"
    assert leaf_to_index == expected_leaf_to_index, \
        f"leaf_to_index incorrect. Obtenu : {leaf_to_index}"
    assert torch.equal(internal_nodes_heights, expected_internal_nodes_heights), \
        f"internal_nodes_heights incorrect. Obtenu : {internal_nodes_heights}"


###############################################################################
# TEST : map_and_vectorize_hierarchy
###############################################################################
def test_map_and_vectorize_hierarchy():
    """
    Teste la fonction map_and_vectorize_hierarchy en comparant les indices
    obtenus avec des valeurs de référence.
    """
    df_hierarchy = pd.read_csv(data_path)
    hierarchy_levels = ["species", "genus", "family", "order", "class"]

    internal_node_to_index, leaf_to_index, _ = build_index_nodes(df_hierarchy, hierarchy_levels)
    device = torch.device("cpu")

    parent_indices, child_indices = map_and_vectorize_hierarchy(
        df_hierarchy,
        hierarchy_levels,
        internal_node_to_index,
        leaf_to_index,
        device
    )

    expected_parent_indices = torch.tensor(
        [0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 11, 10, 10, 10, 7, 7,
         8, 11, 9, 9, 13, 12, 12, 12, 13, 13, 12, 13, 12, 12],
        dtype=torch.long
    )

    expected_child_indices = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        dtype=torch.long
    )

    assert parent_indices.shape == child_indices.shape, \
        f"Shapes différents : {parent_indices.shape} vs {child_indices.shape}"
    assert torch.equal(parent_indices, expected_parent_indices), \
        f"parent_indices incorrect. Obtenu : {parent_indices.tolist()}"
    assert torch.equal(child_indices, expected_child_indices), \
        f"child_indices incorrect. Obtenu : {child_indices.tolist()}"


###############################################################################
# TEST : build_hierarchy_tensors
###############################################################################
def test_build_hierarchy_tensors():
    """
    Teste la construction de la matrice nodes_to_leaves et du vecteur
    internal_nodes_heights complet.
    """
    hierarchy_csv = data_path
    hierarchy_levels = ["species", "genus", "family", "order", "class"]

    nodes_to_leaves, internal_nodes_heights = build_hierarchy_tensors(hierarchy_csv, hierarchy_levels)

    expected_nodes_to_leaves = torch.tensor(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
         [1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.bool
    )
    expected_internal_nodes_heights = torch.tensor([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0, 2.0,
        3.0, 3.0,
        4.0
    ])

    assert torch.equal(nodes_to_leaves, expected_nodes_to_leaves), \
        f"nodes_to_leaves incorrect. Obtenu : {nodes_to_leaves.tolist()}"
    assert torch.equal(internal_nodes_heights, expected_internal_nodes_heights), \
        f"internal_nodes_heights incorrect. Obtenu : {internal_nodes_heights.tolist()}"


###############################################################################
# TEST : generate_index_pairs
###############################################################################
def test_generate_index_pairs():
    """
    Teste la génération des couples (i, j) pour num_leaves=4.
    """
    num_leaves = 4
    i_idx, j_idx = generate_index_pairs(num_leaves, device="cpu")

    expected_i_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    expected_j_idx = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

    assert torch.equal(i_idx, expected_i_idx), \
        f"i_idx incorrect. Obtenu : {i_idx.tolist()}"
    assert torch.equal(j_idx, expected_j_idx), \
        f"j_idx incorrect. Obtenu : {j_idx.tolist()}"


###############################################################################
# TEST : find_lca_index
###############################################################################
def test_find_lca_index():
    """
    Teste find_lca_index avec un exemple plus grand (4 feuilles, 7 nœuds internes).
    """
    # Notre matrice 7×4 de nœuds internes (lignes) vs feuilles (colonnes)
    nodes_to_leaves = torch.tensor([
        [True,  False, False, False],  # nœud 0
        [False, True,  False, False],  # nœud 1
        [True,  True,  False, False],  # nœud 2
        [False, False, True,  False],  # nœud 3
        [False, False, False, True ],  # nœud 4
        [False, False, True,  True ],  # nœud 5
        [True,  True,  True,  True ]   # nœud 6 (racine)
    ], dtype=torch.bool)

    i_idx, j_idx = generate_index_pairs(num_leaves=4, device="cpu")
    lca_idx = find_lca_index(nodes_to_leaves, i_idx, j_idx)

    expected_lca_idx = torch.tensor([
        0, 2, 6, 6,
        2, 1, 6, 6,
        6, 6, 3, 5,
        6, 6, 5, 4
    ], dtype=torch.long)

    assert torch.equal(lca_idx, expected_lca_idx), \
        f"lca_idx incorrect. Obtenu : {lca_idx.tolist()}"


###############################################################################
# TEST : compute_lca_matrix
###############################################################################
def test_compute_lca_matrix():
    """
    Teste la fonction compute_lca_matrix sur un arbre plus grand (4 feuilles, 7 nœuds internes).
    """
    nodes_to_leaves = torch.tensor([
        [True,  False, False, False],  # nœud 0
        [False, True,  False, False],  # nœud 1
        [True,  True,  False, False],  # nœud 2
        [False, False, True,  False],  # nœud 3
        [False, False, False, True ],  # nœud 4
        [False, False, True,  True ],  # nœud 5
        [True,  True,  True,  True ]   # nœud 6 (racine)
    ], dtype=torch.bool)

    internal_nodes_heights = torch.tensor([
        1.0,  # nœud 0
        1.0,  # nœud 1
        2.0,  # nœud 2
        1.0,  # nœud 3
        1.0,  # nœud 4
        2.0,  # nœud 5
        3.0   # nœud 6 (racine)
    ], dtype=torch.float32)

    lca_height_matrix = compute_lca_matrix(nodes_to_leaves, internal_nodes_heights)

    expected_lca_height_matrix = torch.tensor([
        [0.3333, 0.6667, 1.0000, 1.0000],
        [0.6667, 0.3333, 1.0000, 1.0000],
        [1.0000, 1.0000, 0.3333, 0.6667],
        [1.0000, 1.0000, 0.6667, 0.3333]
    ], dtype=torch.float32)

    assert torch.allclose(lca_height_matrix, expected_lca_height_matrix, atol=1e-4), \
        f"\nMatrice LCA incorrecte.\nReçue:\n{lca_height_matrix}\n\nAttendue:\n{expected_lca_height_matrix}"


###############################################################################
# TEST : compute_soft_label
###############################################################################
def test_compute_soft_label():
    """
    Test unitaire pour compute_soft_labels avec 4 feuilles, 7 nœuds internes,
    et un mini-batch de 3 exemples.
    """
    # Matrice nodes_to_leaves (7 noeuds × 4 feuilles)
    nodes_to_leaves = torch.tensor([
        [True,  False, False, False],  # nœud 0
        [False, True,  False, False],  # nœud 1
        [True,  True,  False, False],  # nœud 2
        [False, False, True,  False],  # nœud 3
        [False, False, False, True ],  # nœud 4
        [False, False, True,  True ],  # nœud 5
        [True,  True,  True,  True ]   # nœud 6 (racine)
    ], dtype=torch.bool)

    # Hauteurs
    internal_nodes_heights = torch.tensor([
        1.0,  # nœud 0
        1.0,  # nœud 1
        2.0,  # nœud 2
        1.0,  # nœud 3
        1.0,  # nœud 4
        2.0,  # nœud 5
        3.0   # nœud 6 (racine)
    ], dtype=torch.float32)

    # y_true: batch_size=3, num_classes=4
    # Exemple 1 -> classe 0
    # Exemple 2 -> classe 2
    # Exemple 3 -> classe 3
    y_true = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    beta = 1.5

    # Calcul
    soft_labels = compute_soft_labels(
        y_true=y_true,
        beta=beta,
        nodes_to_leaves=nodes_to_leaves,
        internal_nodes_heights=internal_nodes_heights
    )

    # Résultat attendu
    expected = torch.tensor([
        [0.4270, 0.2590, 0.1570, 0.1570],
        [0.1570, 0.1570, 0.4270, 0.2590],
        [0.1570, 0.1570, 0.2590, 0.4270]
    ], dtype=torch.float32)

    assert soft_labels.shape == (3, 4), \
        f"Shape inattendu. Reçu : {soft_labels.shape}, attendu : (3, 4)"

    assert torch.allclose(soft_labels, expected, atol=1e-03), \
        f"\nSoft labels incorrects.\nReçus:\n{soft_labels}\n\nAttendus:\n{expected}"