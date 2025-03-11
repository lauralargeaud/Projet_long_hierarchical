#!/bin/bash

# Vérifie si un argument (dossier) est fourni
if [ -z "$1" ]; then
    echo "Usage: $0 <dossier> <output> <chaine>"
    exit 1
fi

dossier="$1"
output="$2"
chaine_a_chercher="$3"

echo "" > "$output"

# Vérifie si le dossier existe
if [ ! -d "$dossier" ]; then
    echo "Erreur : Le dossier '$dossier' n'existe pas."
    exit 1
fi

# Rendre les fichiers exécutables si nécessaire et les exécuter
for fichier in "$dossier"/*; do
    if grep -q "$chaine_a_chercher" "$fichier"; then
        echo "Exécution de : $fichier"
        python3 inference.py --config "$fichier" &>> "$output"
        echo -e "\n\n\n\n\n" >> "$output"
    fi
done

# python3 results.py