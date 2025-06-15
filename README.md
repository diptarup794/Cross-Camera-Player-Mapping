# Player Re-Identification Across Multi-View Videos

## Overview
This project detects and tracks players in two synchronized videos (broadcast and tacticam) of the same gameplay, assigning consistent player IDs across both views. It leverages a YOLOv11 model for detection, a custom tracking algorithm, advanced re-identification techniques using deep feature embeddings, and camera calibration for spatial constraints.

## Directory Structure
```
models/      # Stores the YOLOv11 weights (e.g., best.pt)
videos/      # Contains the input video files (e.g., broadcast.mp4, tacticam.mp4)
outputs/     # Stores generated outputs: annotated videos, player ID mappings, and visualization plots
src/         # Contains all source code modules for detection, tracking, feature extraction, re-identification, and utilities
.venv/       # Python virtual environment (created automatically by run.sh)
```

## Setup and Running the Code

### Prerequisites
Before running the project, ensure you have:
-   Python 3.8+ installed.
-   `best.pt` (YOLOv11 model weights) placed in the `models/` directory.
    *   If you don't have `best.pt`, you can download it (e.g., from a YOLOv11 release or training output) and place it in the `models/` folder.
-   Your two MP4 video files (`broadcast.mp4` and `tacticam.mp4`) placed in the `videos/` directory.

### Automated Setup and Execution
The easiest way to set up the environment and run the re-identification pipeline is by using the provided `run.sh` script. This script handles:
1.  **Virtual Environment Creation:** It creates and activates a Python virtual environment (`.venv/`) to isolate project dependencies.
2.  **Dependency Installation:** Installs all required Python packages listed in `requirements.txt` within the virtual environment. It also specifically handles potential `torchreid` installation issues by clearing the cache.
3.  **Camera Calibration Points Extraction (Optional but Recommended):** It's highly recommended to generate calibration points for accurate spatial matching.
    *   To manually select calibration points (recommended for initial setup):
        ```bash
        python src/calibrate_cameras.py --broadcast videos/broadcast.mp4 --tacticam videos/tacticam.mp4 --output calibration_points.txt --frame_numbers 100 200 300 # Example frame numbers
        ```
        Follow the on-screen instructions to click corresponding points in both video views. Aim for at least 4-8 well-distributed points across multiple frames. The more accurate and numerous your points, the better the spatial matching.
4.  **Running the Main Pipeline:** Executes `src/main.py` with the necessary arguments, including the model, input videos, and output directory.

To run the full pipeline, simply execute the `run.sh` script from the project root:
```bash
./run.sh
```

### Manual Setup (Advanced)
If you prefer to set up the environment manually, here are the step-by-step commands:

1.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt
    ```

3.  **Place Model Weights and Videos:**
    *   Ensure `best.pt` is in the `models/` directory.
    *   Ensure `broadcast.mp4` and `tacticam.mp4` are in the `videos/` directory.

4.  **Generate Camera Calibration Points (Optional but Recommended for Spatial Matching):**
    ```bash
    python src/calibrate_cameras.py --broadcast videos/broadcast.mp4 --tacticam videos/tacticam.mp4 --output calibration_points.txt --frame_numbers 100 200 300 # Adjust frame numbers for better coverage
    ```
    *   This will open interactive windows for you to click on corresponding points. Save the `calibration_points.txt` file.

5.  **Run the Main Player Re-Identification Script:**
    ```bash
    python src/main.py --model models/best.pt --broadcast videos/broadcast.mp4 --tacticam videos/tacticam.mp4 --output_dir outputs/ --calibration_points calibration_points.txt
    ```
    *   **Optional Arguments for Fine-Tuning:**
        *   `--feature <type>`: Choose feature type (`'torchreid'` or `'color_patch'`). Default is `'torchreid'`.
        *   `--match_threshold <value>`: Adjust the matching similarity threshold (e.g., `0.5`).
        *   `--color_weight <value>`: Weight for color histogram features in matching (e.g., `0.5`).
        *   `--patch_weight <value>`: Weight for image patch features in matching (e.g., `0.5`).
        *   `--spatial_weight <value>`: Weight for spatial constraints from homography in matching (e.g., `0.2`).
        *   Example with optional arguments:
            ```bash
            python src/main.py \
                --model models/best.pt \
                --broadcast videos/broadcast.mp4 \
                --tacticam videos/tacticam.mp4 \
                --output_dir outputs/ \
                --calibration_points calibration_points.txt \
                --feature torchreid \
                --match_threshold 0.6 \
                --spatial_weight 0.3
            ```

## Features

-   **Player Detection:** Utilizes YOLOv11 for robust and accurate player detection.
-   **Enhanced Player Tracking:** Employs a custom SimpleSORT tracker integrated with a basic Kalman filter for smooth and consistent intra-video ID assignment.
-   **Advanced Cross-View Player Re-Identification:** Leverages TorchReID (OSNet) for extracting highly discriminative deep appearance embeddings, crucial for accurate cross-view matching.
-   **Spatial Constraint Integration:** Incorporates camera calibration (homography) to add spatial proximity as a factor in the re-identification process, improving accuracy.
-   **Consistent Global ID Assignment:** Assigns a unique, consistent ID to each player across both video streams.
-   **Configurable Matching Parameters:** Provides command-line arguments to fine-tune the influence of appearance features, image patches, spatial constraints, and matching thresholds.

## Output

After a successful run, the `outputs/` directory will contain:
-   **Annotated Videos:** `broadcast_annotated.mp4` and `tacticam_annotated.mp4` showing detected players with their assigned track IDs.
-   **Side-by-Side Video:** `side_by_side.mp4` for visual comparison of tracking and re-identification across views.
-   **Player ID Mapping File:** `player_id_mapping.csv` detailing the consistent global player IDs for matched players (broadcast ID, tacticam ID, global player ID, and first appearance frames).
-   **Player Feature Similarity Matrix:** `similarity_matrix.png` visualizing the pairwise feature distances between players from both views.
-   **Player Matches on Ground Plane:** `ground_plane_matches.png` (PNG) providing an intuitive projection of matched players onto a common ground plane, visually confirming spatial alignment.
-   **Calibration Points:** `calibration_points.txt` (if generated) storing the manually selected points for camera calibration.

## Dependencies

All Python dependencies are listed in `requirements.txt` and are automatically installed when running `./run.sh`. Key libraries include:
-   `ultralytics`: For YOLOv11 model.
-   `torch`: Deep learning framework.
-   `opencv-python`: For video processing and image manipulation.
-   `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`: For numerical operations, optimization, feature processing, plotting, and data handling.
-   `torchreid`: For advanced re-identification feature extraction. 