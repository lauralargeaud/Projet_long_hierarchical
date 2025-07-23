import numpy as np
import pickle
import pandas as pd
import torch
import os

def get_layer_matrix(path_to_csv_tree, verbose=False):
  """
  Get layer matrix La from csv.

  La (np.array): layer matrix of shape (height_tree, nb_nodes)
    La[i,j] = 1 if node j is at the i-th hierarchical level of the tree, with 0 being the root

  """
  csv = pd.read_csv(path_to_csv_tree)
  unique_nodes = pd.unique(csv.values.ravel())
  unique_nodes = unique_nodes[~pd.isnull(unique_nodes)]  # On enlève les NaN au cas ou
  node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
  # if verbose:
  #   print(node_to_index)
  La = np.zeros((len(csv.columns), len(unique_nodes)))
  for _, row in csv.iterrows():
     for layer, node in enumerate(row):
        La[layer, node_to_index[node]] = 1

  if verbose:
     print(La)

  return La

def get_tree_matrices(path_to_csv_tree, data_dir, verbose=False):
  """
  Get H, P and M matrix from csv.

  H (np.array): childs matrix of shape (nb_nodes, nb_nodes)
    H[i,j] = 1 if node j is a child of node i, and 0 otherwise

  P (np.array): peers matrix of shape (nb_nodes, nb_nodes)
    P[i,j] stores 1 if j is a peer of i, 0 otherwise

  M (np.array): number of peers matrix of shape (nb_nodes,)
    M[i] = number of peers of node i

  """
  csv = pd.read_csv(path_to_csv_tree)
  unique_nodes = pd.unique(csv.values.ravel())
  unique_nodes = unique_nodes[~pd.isnull(unique_nodes)]  # On enlève les NaN au cas ou
  node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}

  n = len(unique_nodes)
  H = np.zeros((n, n), dtype=int)

  # On parcours le csv lignes par lignes (donc chemin par chemin dans l'arbre)
  for _, row in csv.iterrows():
      for parent, child in zip(row[:-1], row[1:]):
          if pd.notna(parent) and pd.notna(child):
              H[node_to_index[parent], node_to_index[child]] = 1

  ##Partie pour équilibrer les poids des classes !!
  # Indices des niveaux hiérarchiques
  class_indices = [node_to_index[node] for node in csv['class'].unique()]
  order_indices = [node_to_index[node] for node in csv['order'].unique()]
  family_indices = [node_to_index[node] for node in csv['family'].unique()]
  genus_indices = [node_to_index[node] for node in csv['genus'].unique()]
  species_indices = [node_to_index[node] for node in csv['species'].unique()]

  # Compter le nombre d'images par espèce
  species_image_count = {}
  train_dir = os.path.join(data_dir, "train")
  for species in csv['species'].unique():
      species_dir = os.path.join(train_dir, species)
      if os.path.exists(species_dir):
          species_image_count[species] = len(os.listdir(species_dir))
      else:
          species_image_count[species] = 0


  # Associer les images aux niveaux hiérarchiques
  csv['images'] = csv['species'].map(species_image_count)
  images_count_by_class = csv.groupby('class')['images'].sum()
  images_count_by_order = csv.groupby('order')['images'].sum()
  images_count_by_family = csv.groupby('family')['images'].sum()
  images_count_by_genus = csv.groupby('genus')['images'].sum()
  images_count_by_species = csv.groupby('species')['images'].sum()

  if verbose:
    print("\nNombre d'images par classe :")
    for class_, count in images_count_by_class.items():
        print(f"Classe '{class_}': {count} images")

    print("\nNombre d'images par ordre :")
    for order, count in images_count_by_order.items():
        print(f"Ordre '{order}': {count} images")

    print("\nNombre d'images par famille :")
    for family, count in images_count_by_family.items():
        print(f"Famille '{family}': {count} images")
            
    print("\nNombre d'images par genre :")
    for genus, count in images_count_by_genus.items():
        print(f"Genre '{genus}': {count} images")

    print("\nNombre d'images par espèce :")
    for species, count in images_count_by_species.items():
        print(f"Espèce '{species}': {count} images")

  
  if verbose:
    print("Nodes:", unique_nodes)
    print("H:\n", H)

  peer_matrix = np.matmul(H.T,H)
  peer_matrix = np.maximum(peer_matrix - np.eye(n),0)

  M = np.sum(peer_matrix, axis=0) # peer_matrix est symétrique donc l'axe n'est pas important
  
  return H, peer_matrix, M, class_indices, order_indices, family_indices, genus_indices, species_indices, images_count_by_class, images_count_by_order, images_count_by_family, images_count_by_genus, images_count_by_species

def create_class_to_labels(path_to_csv_tree, path_to_temporary_class_to_labels_file, verbose=False):
  """
  Create class to labels .pkl file.
  # The ".pkl" file stores the dictionary storing classes names as keys and assoiacted labels as values.
  # A label is a tensor of nb_nodes elements, with ones in the root to leaf path to the leaf to predict and 0 elsewhere
  """
  label_matrix, _, index_to_node = get_label_matrix(path_to_csv_tree, verbose)

  class_to_labels = get_class_to_label(label_matrix, index_to_node, verbose)

  # Sauvegarder dans un fichier .pkl
  with open(path_to_temporary_class_to_labels_file, "wb") as f:
      pickle.dump(class_to_labels, f)  # Écriture binaire


def get_label_matrix(path_to_csv_tree, verbose=False):
  """
  Get label matrix from csv.

  The label matrix of size (nb_leaves, nb_nodes) stores for each leaf the vector 
  corresponding to its leaf to root path in the tree.

  """
  csv = pd.read_csv(path_to_csv_tree)
  unique_nodes = pd.unique(csv.values.ravel())
  unique_nodes = unique_nodes[~pd.isnull(unique_nodes)]  # On enlève les NaN au cas ou
  node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
  index_to_node = {v: k for k, v in node_to_index.items()}
  n = len(unique_nodes)
  label_matrix = np.zeros((len(csv),n), dtype=int)
  # On parcours le csv lignes par lignes (donc chemin par chemin dans l'arbre)
  for i, row in csv.iterrows():
      for node in row:
        label_matrix[i,node_to_index[node]] = 1
  
  if(verbose):
    print('labels_matrix')
    print(label_matrix)
  return label_matrix, node_to_index, index_to_node

def get_class_to_label(label_matrix, index_to_node, verbose=False):
  """
  Get class to label from label matrix.

  """
  class_to_labels={}
  # On parcours le csv lignes par lignes (donc chemin par chemin dans l'arbre)
  for i in range(label_matrix.shape[0]):
      current_branch = label_matrix[i]
      class_index = np.where(current_branch == 1)[0][-1]
      class_name = index_to_node[class_index]
      class_to_labels[class_name] = current_branch.tolist()
  if(verbose):
    print('class_to_labels')
    print(class_to_labels)
    print("class_to_labels 1", class_to_labels)
  return class_to_labels

def get_logicseg_predictions(pred, label_matrix, device):
  """
  Get logicseg predictions.

  pred = sigmoid(y_pred) => la sigmoid a déjà été appliquée
  Pour chaque prédiction du modèle sur laquelle on a appliqué la sigmoid, on veut: 
    tenseur de taille (nb_branches,) contenant les probas de toutes les branches
  sortie: shape = (nb_pred, nb_branches)

  """
  label_matrix = torch.tensor(label_matrix, dtype=torch.float32).to(device) #torch tensor
  nb_pred = pred.shape[0]
  nb_feuilles = label_matrix.shape[0]
  probas_branches = torch.empty(size=(nb_pred, nb_feuilles), dtype=torch.float32)
  for i in range(nb_pred):
    pred_rep = pred[i,:].repeat(nb_feuilles, 1)
    probas_branches[i,:] = torch.sum(pred_rep*label_matrix, dim=1)
  return probas_branches


def get_branches_label(most_probable_branches_indices_in, most_probable_branches_indices_target, class_to_label):
  """
  Get branches label.

  most_probables_branches: indices des k branches les plus probables dans l'ordre décroissant de probabilité
  return: les k labels textuels associés
  most (nb_pred, top_k)

  """
  predicted_classes = np.empty(most_probable_branches_indices_in.shape, dtype=object)
  classes = list(class_to_label.keys())

  for p in range(most_probable_branches_indices_in.shape[0]):
    for i in range(most_probable_branches_indices_in.shape[1]):
      predicted_classes[p,i] = "pred: " + classes[most_probable_branches_indices_in[p,i]] + ", label:  " + classes[most_probable_branches_indices_target[p,i]]
  return predicted_classes


def add_nodes_to_output(path_to_csv_tree, output, classes, node_to_index):
  '''
  
    Adding branches and nodes to the output.
    In: output in which each prediction is of shape (nb_leaves,)
    Out: augmented output in which each prediction is in the "LogicSeg format" of size (nb_nodes,)
  
  '''
  H_raw, _, _ = get_tree_matrices(path_to_csv_tree, verbose=False)
  La_raw = get_layer_matrix(path_to_csv_tree, verbose=False)
  

  tree_height, _ = La_raw.shape
  nb_nodes,_ = H_raw.shape
  batch_size, nb_leafs = output.shape

  augmented_output = torch.zeros((nb_nodes, batch_size))
  
  # Premier étage de la hierarchy
  for i in range(nb_leafs):
    augmented_output[node_to_index[classes[i]], :] = output[:, i]
  
  # On remplit les étages suivants
  for i in range(tree_height-2, -1, -1):
    branches_idx = np.where(La_raw[i,:] == 1)[0]

    for branch_idx in branches_idx:
      child_idx = np.where(H_raw[branch_idx,:] == 1)[0]
      augmented_output[branch_idx, :] = augmented_output[child_idx, :].sum(axis=0)
  
  return augmented_output

def format_target(target, nb_nodes):
  '''pour mettre un target sous la forme necessaire pour les metriques ("format LogicSeg")'''
  batch_size = len(target)
  target_formated = torch.zeros((nb_nodes, batch_size))

  for i in range(batch_size):
     target_formated[target[i], i] = 1
  
  return target_formated


  