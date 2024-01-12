#!/bin/bash

# usage: delete.sh
# This shell script deletes some files

found=0

# List of files to delete
files=("resposta.txt" "response.txt" "indice.txt" "index.txt" "etiquetador.bin" "tagger.bin" "pesos.txt" "weights.txt")

# Loop through the list of files and delete each file if it exists
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        found=1
    fi
done

if [ $found -eq 0 ]; then
    echo "No files found to delete."
fi