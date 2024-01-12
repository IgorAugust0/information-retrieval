#!/bin/bash
# usage: ./delete.sh

found=0

# List of files to delete
files=("resposta.txt" "response.txt" "indice.txt" "index.txt" "etiquetador.bin" "tagger.bin")

# Loop through the list of files and delete each file if it exists
for file in "${files[@]}"
do
    if [ -f "$file" ]; then
        rm -f "$file"
        found=1
    fi
done

if [ $found -eq 0 ]; then
    echo "No files found to delete."
fi

# ---- Alternative way to delete files ----
# ----------------------------------------------------------------------------------------------
# if [ -f "resposta.txt" ]; then
#     rm -f resposta.txt
# elif [ -f "response.txt" ]; then
#     rm -f response.txt
# fi

# if [ -f "indice.txt" ]; then
#     rm -f indice.txt    
# elif [ -f "index.txt" ]; then
#     rm -f index.txt
# fi

# if [ -f "etiquetador.bin" ]; then
#     rm -f etiquetador.bin
# elif [ -f "tagger.bin" ]; then
#     rm -f tagger.bin
# fi
