import itertools
import torch
import pandas as pd
import numpy as np

###############################################################################
### CALCUL DES SOFT LABELS 
###############################################################################

def compute_soft_labels(y_true: torch.Tensor, 
                       beta: torch.float,
                       nodes_to_leaves: torch.Tensor,
                       internal_nodes_heights: torch.Tensor):
  """
  Calcule le soft label en utilisant la matrice des hauteurs LCA.

  Arg
      y_true (torch.Tensor): Tenseur one-hot encoding des labels réels (batch_size, num_classes)
      beta (torch.float) : l'hyperparamètre beta du soft labels
      nodes_to_leaves (torch.Tensor): Matrice binaire (num_internal_nodes, num_leaves)
      internal_nodes_heights (torch.Tensor): Tenseur des hauteurs des noeuds internes + racine

  Returns:
      torch.Tensor: Tenseur des soft labels (batch_size, num_classes)
  """
  batch_size = y_true.shape[0]  # Taille du batch
  
  # Trouver les indices des labels y_true
  y_true_idx = torch.argmax(y_true, dim=1) # (batch_size,)

  # Calcul de la matrice des hauteurs LCA (num_classes, num_classes)
  lca_height_matrix = compute_lca_matrix(nodes_to_leaves, internal_nodes_heights)

  # Sélectionner les lignes correspondant aux indices des labels réels (batch processing)
  selected_heights = lca_height_matrix[y_true_idx, :]  # (batch_size, num_classes)

  # Calcul du numérateur de la formule du soft label : exp(-beta * hauteur du LCA)
  numerator = torch.exp(-beta * selected_heights) # (batch_size, num_classes)

  # Calcul du dénominateur de la formule du soft label : somme sur les classes
  denominator = numerator.sum(dim=1, keepdim=True)  # (batch_size, 1)

  return numerator / denominator # (batch_size, num_classes)

###############################################################################
### CONSTRUCTION DE nodes_to_leaves ET internal_nodes_heights
###############################################################################

def build_hierarchy_tensors(hierarchy_csv: str, 
                            hierarchy_levels: list[str]
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construit les tenseurs de correspondance hiérarchique entre les noeuds internes
    et les feuilles avec une approche entièrement vectorisée.

    Args:
        hierarchy_csv: Chemin du fichier CSV contenant la hiérarchie.
        hierarchy_levels: Liste des niveaux hiérarchiques, du plus bas au plus haut.

    Returns:
        Tenseur de mapping hiérarchique et Tenseur des hauteurs des noeuds internes + racine.
    """
    # Charger le fichier CSV dans un DataFrame
    df_hierarchy = pd.read_csv(hierarchy_csv)

    # Détecter le GPU et transférer sur CUDA si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Créer les dictionnaires d'indexation des nœuds internes et des feuilles
    internal_node_to_index, leaf_to_index, internal_nodes_heights = build_index_nodes(
        df_hierarchy, hierarchy_levels, device)

    internal_nodes_idx, leaves_idx = map_and_vectorize_hierarchy(
        df_hierarchy, hierarchy_levels, internal_node_to_index, leaf_to_index, device)

    # Initialisation de la matrice binaire sur GPU
    nodes_to_leaves = torch.zeros((len(internal_node_to_index), 
                                   len(leaf_to_index)),
                                  dtype=torch.bool, 
                                  device=device)

    # Remplissage vectorisé de la matrice
    nodes_to_leaves[internal_nodes_idx, leaves_idx] = True

    # La dernière ligne (index -1) = racine : couvre toutes les feuilles
    nodes_to_leaves[-1, :] = True

    return nodes_to_leaves, internal_nodes_heights


def build_index_nodes(df_hierarchy: pd.DataFrame, 
                      hierarchy_levels: list[str], 
                      device: torch.device
                      ) -> tuple[dict[str, int], dict[str, int], torch.Tensor]:
    """
    Génère les dictionnaires d'indexation des noeuds internes et des feuilles,
    ainsi que le tenseur des hauteurs des noeuds internes indexé par leur numéro.

    Args:
        df_hierarchy (pd.DataFrame): DataFrame contenant la hiérarchie des noeuds.
        hierarchy_levels (List[str]): Liste des niveaux hiérarchiques, du plus bas au plus haut.

    Returns:
        Tuple[Dict[str, int], Dict[str, int], torch.Tensor]:
            - internal_node_to_index (dict): Dictionnaire des noeuds internes avec indices.
            - leaf_to_index (dict): Dictionnaire des feuilles avec indices triés.
            - internal_nodes_heights (torch.Tensor): Tenseur des hauteurs indexé par l'indice du noeud interne.
    """
    # Dictionnaire associant chaque noeud interne à un index
    internal_node_to_index = {}
    node_index = itertools.count(0)  # Générateur d'indices uniques

    # On crée une liste temporaire pour stocker les hauteurs des nœuds
    internal_nodes_heights_list = []

    # Parcourir les niveaux hiérarchiques du plus spécifique (genus) au plus général (class)
    for height, level in enumerate(hierarchy_levels[1:], start=1):  # Hauteur commence à 1
        nodes_at_level = sorted(df_hierarchy[level].unique())  # Valeurs uniques et triées
        for node in nodes_at_level:
            idx = next(node_index)
            internal_node_to_index[node] = idx
            internal_nodes_heights_list.append(height)  # Stocke la hauteur temporairement

    # Transformer en tenseur PyTorch
    internal_nodes_heights = torch.tensor(internal_nodes_heights_list, dtype=torch.float32, device=device)

    # Générer le dictionnaire des feuilles
    leaf_to_index = {name: i for i, name in enumerate(sorted(df_hierarchy[hierarchy_levels[0]].unique()))}

    return internal_node_to_index, leaf_to_index, internal_nodes_heights


def map_and_vectorize_hierarchy(df_hierarchy: pd.DataFrame, 
                                hierarchy_levels: list[str],
                                internal_node_to_idx: dict,
                                leaf_to_idx: dict,
                                device: torch.device
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transforme un DataFrame hiérarchique en indices et les étend pour une affectation vectorisée.

    Args:
        df_hierarchy (pd.DataFrame): DataFrame contenant la hiérarchie dans l'ordre du plus haut au plus bas.
        hierarchy_levels (list[str]): Liste des niveaux hiérarchiques (du plus bas au plus haut).
        internal_node_to_idx (dict): Dictionnaire mappant les noeuds internes à leurs indices.
        leaf_to_idx (dict): Dictionnaire mappant les feuilles à leurs indices.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - `internal_nodes_idx`: Indices des noeuds internes (concaténés) sans la racine.
            - `leaves_idx`: Indices des feuilles (répliqués).
    """
    # Exclure la première colonne (= racine de l'arbre) et appliquer l'indexation
    mapped_df = df_hierarchy.iloc[:, 1:].apply(
        lambda col: col.map(internal_node_to_idx) if col.name != "species"
        else col.map(leaf_to_idx)
    )

    # Sélectionner les colonnes parents (exclut species et class)
    internal_nodes_columns = hierarchy_levels[1:-1] 

    # Concaténer les indices des parents en une seule colonne
    internal_nodes_idx = pd.concat([mapped_df[col] for col in internal_nodes_columns], axis=0).to_numpy()

    # Répliquer les indices des feuilles pour correspondre à la taille des parents concaténés
    leaves_idx = np.tile(mapped_df["species"].to_numpy(), len(internal_nodes_columns))

    # Transformer en tenseurs Pytorch
    internal_nodes_idx = torch.tensor(internal_nodes_idx, device=device)
    leaves_idx = torch.tensor(leaves_idx, device=device)

    return internal_nodes_idx, leaves_idx

###############################################################################
### CONSTRUCTION DE lca_matrix
###############################################################################


def generate_index_pairs(num_leaves, device="cpu"):
    """
    Génère tous les couples (i, j) d'indices des feuilles sous forme de tenseurs.

    Args:
        num_leaves (int): Nombre total de feuilles.
        device (str): Périphérique d'exécution (CPU ou GPU).

    Returns:
        torch.Tensor, torch.Tensor: Tenseurs contenant tous les couples (i, j).
    """
    indices = torch.arange(num_leaves, device=device)

    # Création de matrices d'indices (i, j)
    i_idx, j_idx = torch.meshgrid(indices, indices, indexing="ij")

    # Convertir en vecteurs aplatis pour une exécution vectorisée
    return i_idx.flatten(), j_idx.flatten()


def find_lca_index(nodes_to_leaves: torch.Tensor, 
                   i_idx: torch.Tensor, 
                   j_idx: torch.Tensor):
    """
    Trouve les LCA pour tous les couples de feuilles vectorisés en une seule passe.

    Args:
        nodes_to_leaves (torch.Tensor): Matrice binaire (nœuds internes × feuilles).
        i_idx (torch.Tensor): Indices i des couples (i,j).
        j_idx (torch.Tensor): Indices j des couples (i,j).

    Returns:
        torch.Tensor: Tenseur des indices des LCA pour chaque couple.
    """
    # Sélection des colonnes correspondant aux feuilles i et j
    col_i = nodes_to_leaves[:, i_idx]  # (nœuds internes, nb_couples)
    col_j = nodes_to_leaves[:, j_idx]  # (nœuds internes, nb_couples)

    # Produit logique AND pour trouver les ancêtres communs
    common_ancestors = col_i & col_j  # (nœuds internes, nb_couples)

    # Trouver l’indice du premier 1 (le plus bas ancêtre commun)
    lca_index = torch.argmax(common_ancestors.float(), dim=0)  # (nb_couples,)

    return lca_index


def compute_lca_matrix(nodes_to_leaves: torch.Tensor,
                       internal_nodes_heights: torch.Tensor
                       ) -> torch.Tensor:
    """
    Construit la matrice des distances entre toutes les feuilles avec une exécution vectorisée.

    Args:
        nodes_to_leaves (torch.Tensor): Matrice (nœuds internes × feuilles).
        internal_nodes_heights (torch.Tensor): Vecteur des hauteurs des noeuds internes + racine.

    Returns:
        torch.Tensor: Matrice des distances normalisées entre feuilles.
    """
    num_leaves = nodes_to_leaves.shape[1]
    device = nodes_to_leaves.device

    # Générer tous les couples possibles (i, j)
    i_idx, j_idx = generate_index_pairs(num_leaves, device)

    # Trouver tous les LCA en une seule passe
    lca_indices = find_lca_index(nodes_to_leaves, i_idx, j_idx)

    # Initialisation de la matrice des distances
    lca_height_matrix = torch.zeros((num_leaves, num_leaves), dtype=torch.float32, device=device)

    # Remplir la matrice des distances avec les indices des LCA
    lca_height_matrix[i_idx, j_idx] = internal_nodes_heights[lca_indices]

    # Normalisation par la hauteur de la racine
    return lca_height_matrix / internal_nodes_heights[-1] # (num_classes, num_classes)

if __name__ == "__main__":
    # Exemple avec 10 feuilles et 5 niveaux hiérarchiques
    hierarchy_csv = "./tests/subset_hierarchy.csv"
    hierarchy_levels = ["species", "genus", "family", "order", "class"]
    
    nodes_to_leaves, internal_nodes_heights = build_hierarchy_tensors(hierarchy_csv, hierarchy_levels)
    print("============= nodes_to_leaves =============")
    print(nodes_to_leaves)
    print("==================================================")
    print("============= internal_nodes_heights =============")
    print(internal_nodes_heights)
    print("==================================================")

    beta = 1.5
    y_true = torch.tensor([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Exemple 1 (classe 2)
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Exemple 2 (classe 5)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Exemple 3 (classe 9)
    ], dtype=torch.float32)

    soft_labels = compute_soft_labels(y_true, beta, nodes_to_leaves, internal_nodes_heights)
    print("============= y_true =============")
    print(y_true)
    print("==================================================")
    print("============= soft_labels =============")
    print(soft_labels)
    print("==================================================")
    print("============= soft_labels.shape =============")
    print(soft_labels.shape)
    print("==================================================")
    print("============= soft_labels.sum(dim=1) =============")
    print(soft_labels.sum(dim=1))
    print("==================================================")


