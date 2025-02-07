def get_tree_matrices(path_to_csv_tree, verbose=False):
  import pandas as pd
  import numpy as np

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

  if verbose:
    print("Nodes:", unique_nodes)
    print("H:\n", H)

  peer_matrix = np.matmul(H.T,H)
  peer_matrix = np.maximum(peer_matrix - np.eye(n),0)

  M = np.sum(peer_matrix, axis=0) # peer_matrix est symétrique donc l'axe n'est pas important
  
  return H, peer_matrix, M

def create_class_to_labels(path_to_csv_tree, path_to_temporary_class_to_labels_file):
  import pandas as pd
  import numpy as np
  import pickle

  csv = pd.read_csv(path_to_csv_tree)
  unique_nodes = pd.unique(csv.values.ravel())
  unique_nodes = unique_nodes[~pd.isnull(unique_nodes)]  # On enlève les NaN au cas ou
  node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
  n = len(unique_nodes)

  label_matrix = np.zeros((len(csv),n), dtype=int)

  # On parcours le csv lignes par lignes (donc chemin par chemin dans l'arbre)
  for i, row in csv.iterrows():
      for node in row:
        label_matrix[i,node_to_index[node]] = 1
        
  index_to_node = {v: k for k, v in node_to_index.items()}
  label_count = label_matrix.shape[0]
  class_to_labels={}
  for i in range(label_count):
    current_branch = label_matrix[i]
    class_index = np.where(current_branch == 1)[0][-1]
    class_name = index_to_node[class_index]
    class_to_labels[class_name] = current_branch.tolist()

  # Sauvegarder dans un fichier .pkl
  with open(path_to_temporary_class_to_labels_file, "wb") as f:
      pickle.dump(class_to_labels, f)  # Écriture binaire