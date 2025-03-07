#!/bin/bash

# Vérifie si un argument (dossier) est fourni
if [ -z "$1" ]; then
    echo "Usage: $0 <dossier> <output>"
    exit 1
fi

dossier="$1"
output="$2"

echo "" > "$output"

# Vérifie si le dossier existe
if [ ! -d "$dossier" ]; then
    echo "Erreur : Le dossier '$dossier' n'existe pas."
    exit 1
fi

# Rendre les fichiers exécutables si nécessaire et les exécuter
for fichier in "$dossier"/*; do
    echo "Exécution de : $fichier"
    python3 inference.py --config "$fichier" &>> "$output"
    echo -e "\n\n\n\n\n" >> "$output"
done

python3 results.py