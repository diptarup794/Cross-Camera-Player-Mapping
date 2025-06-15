#!/bin/bash

# Create and activate a virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Ensure pip is up-to-date within the virtual environment
python -m pip install --upgrade pip

# Install all dependencies within the virtual environment
python -m pip install -r requirements.txt

# No need for explicit torchreid import check here, main.py will handle it.

# Define paths (adjust as needed)
MODEL_PATH="models/best.pt"
BROADCAST_VIDEO="videos/broadcast.mp4"
TACICAM_VIDEO="videos/tacticam.mp4"
OUTPUT_DIR="outputs/"

# Run the main script with lightweight features and visualizations for the entire video
python src/main.py \
  --model "$MODEL_PATH" \
  --broadcast "$BROADCAST_VIDEO" \
  --tacticam "$TACICAM_VIDEO" \
  --output_dir "$OUTPUT_DIR" \
  --feature combined

# Deactivate virtual environment (optional, but good practice if more commands follow)
# deactivate

echo "
----------------------------------------------------
Processing complete. Check the 'outputs/' directory for:
  - Annotated videos (broadcast_annotated.mp4, tacticam_annotated.mp4)
  - Side-by-side video (side_by_side.mp4)
  - Player ID mapping (player_id_mapping.csv)
  - Feature distance matrix plot (feature_distance_matrix.png)
  - Ground plane matches plot (ground_plane_matches.png)
----------------------------------------------------" 