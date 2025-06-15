import cv2
import matplotlib.pyplot as plt
import numpy as np
from reid import project_points  # Import project_points from reid.py

def get_player_color(player_id):
    """
    Generates a consistent unique color for each player ID.
    Colors are generated based on a hash of the player_id.
    Returns: BGR color tuple (0-255).
    """
    np.random.seed(player_id % 256) # Use player_id for consistent random color
    color = np.random.randint(0, 256, size=3).tolist()
    return tuple(color)

def draw_boxes(frame, boxes, ids=None, color=(0,255,0)):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if ids is not None:
            # Use global ID as label
            cv2.putText(frame, f'ID:{ids[i]}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def save_video(frames, path, fps=30):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

def make_side_by_side(frames_a, frames_b, out_path, fps=30):
    h, w = frames_a[0].shape[:2]
    combined_frames = [np.hstack((a, b)) for a, b in zip(frames_a, frames_b)]
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w*2, h))
    for f in combined_frames:
        out.write(f)
    out.release()

def plot_similarity_matrix(sim_matrix, out_path, title='Player Feature Similarity Matrix'):
    plt.figure(figsize=(8,6))
    plt.imshow(sim_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Tacticam Players')
    plt.ylabel('Broadcast Players')
    plt.savefig(out_path)
    plt.close()

def plot_ground_plane_matches(unique_ids_a, unique_ids_b, rep_bboxes_a, rep_bboxes_b, matches, H_b_to_t, H_t_to_b, output_path, global_id_map):
    plt.figure(figsize=(12, 8))
    plt.title('Player Matches on Ground Plane')
    plt.xlabel('X-coordinate (pixels in Broadcast View)')
    plt.ylabel('Y-coordinate (pixels in Broadcast View)')
    plt.grid(True)

    # Convert bboxes to centers for plotting
    centers_a_all = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in rep_bboxes_a])
    centers_b_all = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in rep_bboxes_b])

    # Project all tacticam player centers to broadcast view
    projected_centers_b_to_a = []
    if H_t_to_b is not None and len(centers_b_all) > 0:
        projected_centers_b_to_a = project_points(centers_b_all, H_t_to_b)
    
    # Plot all broadcast players
    if len(centers_a_all) > 0:
        plt.scatter(centers_a_all[:, 0], centers_a_all[:, 1], color='blue', marker='o', label='Broadcast Players (Unmatched)', alpha=0.6)

    # Plot all tacticam players (projected)
    if len(projected_centers_b_to_a) > 0:
        projected_centers_b_to_a_np = np.array(projected_centers_b_to_a)
        plt.scatter(projected_centers_b_to_a_np[:, 0], projected_centers_b_to_a_np[:, 1], 
                    color='orange', marker='x', label='Tacticam Players (Projected, Unmatched)', alpha=0.6)

    # Plot matched players and draw lines between them
    matched_broadcast_ids = set()
    matched_tacticam_ids = set()

    for idx_a, idx_b in matches:
        broadcast_id = unique_ids_a[idx_a]
        tacticam_id = unique_ids_b[idx_b]

        # Get global player ID from mapping
        global_player_id = None
        if broadcast_id in global_id_map:
            global_player_id = global_id_map[broadcast_id]
        elif tacticam_id in global_id_map:
            global_player_id = global_id_map[tacticam_id]

        # Get original bbox centers
        center_a = centers_a_all[idx_a]
        center_b_original = centers_b_all[idx_b]
        
        # Project tacticam center to broadcast view for drawing line
        projected_center_b = project_points([center_b_original], H_t_to_b)[0]

        # Draw line between matched players
        plt.plot([center_a[0], projected_center_b[0]], 
                 [center_a[1], projected_center_b[1]], 
                 color='red', linestyle='-', linewidth=1, alpha=0.8)
        
        # Plot matched points with different markers/colors
        plt.scatter(center_a[0], center_a[1], color='red', marker='o', s=100, edgecolors='black', linewidths=1.5)
        plt.scatter(projected_center_b[0], projected_center_b[1], color='red', marker='x', s=100, edgecolors='black', linewidths=1.5)

        # Annotate with global player ID
        if global_player_id is not None:
            plt.text(center_a[0] + 5, center_a[1] + 5, f'ID:{global_player_id}', fontsize=9, color='red', weight='bold')
            plt.text(projected_center_b[0] + 5, projected_center_b[1] + 5, f'ID:{global_player_id}', fontsize=9, color='red', weight='bold')
        
        matched_broadcast_ids.add(broadcast_id)
        matched_tacticam_ids.add(tacticam_id)

    plt.legend()
    plt.savefig(output_path)
    plt.close()

# NOTE: plot_ground_plane_matches also needs access to `project_points` from reid.py. 
# Ensure it's imported or passed correctly if not global.
# For simplicity, we are assuming plot_ground_plane_matches is within utils.py and can import from reid.py.
# If `project_points` cannot be imported, it needs to be passed as argument or defined locally (less ideal). 
# It's better to ensure reid.py functions are accessible where needed. 

# NOTE: The `project_points` function is assumed to exist and be imported from reid.py.
# If it's not available, it needs to be passed as an argument or defined locally. 