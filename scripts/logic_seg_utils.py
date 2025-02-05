def get_H_matrix(path_to_csv_tree, verbose=False):
  import pandas as pd
  import numpy as np

  csv = pd.read_csv(path_to_csv_tree)

  unique_nodes = pd.unique(csv.values.ravel())
  unique_nodes = unique_nodes[~pd.isnull(unique_nodes)]  # On enlève les NaN au cas ou
  node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}

  n = len(unique_nodes)
  H = np.zeros((n, n), dtype=int)

  for _, row in csv.iterrows():
      for parent, child in zip(row[:-1], row[1:]):
          if pd.notna(parent) and pd.notna(child):
              H[node_to_index[parent], node_to_index[child]] = 1

  if verbose:
    print("Nodes:", unique_nodes)
    print("H:\n", H)

  return H