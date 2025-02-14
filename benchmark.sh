#!/bin/bash

# Dossier contenant les sous-dossiers d'images
IMAGE_FOLDER="./data/"
# Programme à exécuter (remplace par ton programme)
PROGRAM="./build/mainFFT"
# Fichier de sortie
OUTPUT_FILE="execution_times.txt"

# Vérifier si le dossier existe
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Erreur : le dossier $IMAGE_FOLDER n'existe pas."
    exit 1
fi

# Supprimer le fichier de sortie s'il existe déjà
if [ -f "$OUTPUT_FILE" ]; then
    rm "$OUTPUT_FILE"
fi

# Boucle sur chaque sous-dossier dans le dossier principal
for subfolder in "$IMAGE_FOLDER"/*/; do
    if [ -d "$subfolder" ]; then
        echo "Traitement du sous-dossier : $subfolder"

        # Initialiser les variables pour chaque sous-dossier
        TOTAL_TIME=0
        COUNT=0

        # Boucle sur toutes les images du sous-dossier
        for image in "$subfolder"*.{jpg,jpeg,png,bmp}; do
            if [ -f "$image" ]; then
                START_TIME=$(date +%s.%N)  # Temps de début
                $PROGRAM "$image"  # Exécuter le programme sur l'image
                END_TIME=$(date +%s.%N)  # Temps de fin

                # Calcul du temps d'exécution
                EXEC_TIME=$(echo "$END_TIME - $START_TIME" | bc)

                # Accumuler le temps total
                TOTAL_TIME=$(echo "$TOTAL_TIME + $EXEC_TIME" | bc)
                COUNT=$((COUNT + 1))
            fi
        done

        # Calcul de la moyenne pour ce sous-dossier et écriture dans le fichier
        if [ "$COUNT" -gt 0 ]; then
            AVG_TIME=$(echo "scale=6; $TOTAL_TIME / $COUNT" | bc)
            echo "$subfolder : Temps moyen = $AVG_TIME secondes" >> "$OUTPUT_FILE"
        else
            echo "$subfolder : Aucun fichier image trouvé" >> "$OUTPUT_FILE"
        fi
    fi
done

echo "Résultats sauvegardés dans $OUTPUT_FILE"
