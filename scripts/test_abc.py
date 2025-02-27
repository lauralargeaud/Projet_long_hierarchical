import pandas as pd
import csv

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

    hierarchy_filename = "/mnt/c/Users/rubcr/OneDrive/Bureau/projet_long/pytorch-image-models-bees/scripts/hierarchy.csv"
    hierarchy_lines = read_csv(hierarchy_filename)
    hierarchy_lines_without_names = hierarchy_lines[1:]
    parents = get_parents(hierarchy_lines_without_names)
    # parents = dict(sorted(parents.items()))
    # parents = list(parents.values())
    # print(new_df["Name"])
    # print(parents)
    # new_df["Parent"] = parents
    print(parents)
    new_df["Parent"] = new_df["Name"].map(parents)

    taxon_levels = get_taxonLevel(hierarchy_lines)
    taxon_levels = dict(sorted(taxon_levels.items()))
    # taxon_levels = list(taxon_levels.values())
    # new_df["Taxon_level"] = 
    print(taxon_levels)
    new_df["Taxon_level"] = new_df["Name"].map(taxon_levels)

    new_df.to_csv(path_generated_csv, index=False)

df = pd.read_csv("/mnt/c/Users/rubcr/OneDrive/Bureau/projet_long/pytorch-image-models-bees/scripts/metrics_all.csv")
build_F1_perfs_csv(df, "/mnt/c/Users/rubcr/OneDrive/Bureau/projet_long/pytorch-image-models-bees/scripts/test1.csv", None)
