#!/bin/bash

ORIG_BASE=/home/shenzheng_google_com/Projects/Inf_Perception/Datasets/OPV2V
ADDN_BASE=/home/shenzheng_google_com/Projects/Inf_Perception/Datasets/OPV2V/additional-001/additional
MERGED_BASE=/home/shenzheng_google_com/Projects/Inf_Perception/Datasets/OPV2V/merged

for SPLIT in train test validate; do
    ORIG="$ORIG_BASE/DATA_${SPLIT}/${SPLIT}"
    ADDN="$ADDN_BASE/${SPLIT}"
    MERGED="$MERGED_BASE/${SPLIT}"

    echo "Merging $SPLIT..."
    mkdir -p "$MERGED"

    for SOURCE in "$ORIG" "$ADDN"; do
        find "$SOURCE" -type f | while read -r f; do
            rel_path="${f#$SOURCE/}"
            dest="$MERGED/$rel_path"
            mkdir -p "$(dirname "$dest")"
            ln -sf "$f" "$dest"
        done
    done
done