#!/bin/bash

# Dossier contenant les images
IMAGE_FOLDER="."
# Programme à exécuter (remplace par ton programme)
PROGRAM="./testFFTCPU"
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

TOTAL_TIME=0
COUNT=0

# Boucle sur toutes les images du dossier
for image in "$IMAGE_FOLDER"/*.{jpg,jpeg,png,bmp}; do
    if [ -f "$image" ]; then
        echo "Traitement de : $image"
        
        START_TIME=$(date +%s.%N)  # Temps de début
        $PROGRAM "$image"  # Exécuter le programme sur l'image
        END_TIME=$(date +%s.%N)  # Temps de fin

        # Calcul du temps d'exécution
        EXEC_TIME=$(echo "$END_TIME - $START_TIME" | bc)
        echo "$image : $EXEC_TIME secondes" | tee -a "$OUTPUT_FILE"

        # Accumuler le temps total
        TOTAL_TIME=$(echo "$TOTAL_TIME + $EXEC_TIME" | bc)
        COUNT=$((COUNT + 1))
    fi
done

# Calcul de la moyenne
if [ "$COUNT" -gt 0 ]; then
    AVG_TIME=$(echo "scale=6; $TOTAL_TIME / $COUNT" | bc)
    echo "Temps moyen : $AVG_TIME secondes" | tee -a "$OUTPUT_FILE"
else
    echo "Aucune image trouvée dans $IMAGE_FOLDER."
fi

echo "Résultats sauvegardés dans $OUTPUT_FILE"

