import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

def euclidean_distance(feature1, feature2):
    """
    Computes the Euclidean distance between two feature vectors.
    """
    return np.linalg.norm(feature1 - feature2)

def weighted_feature_distance(feature1, feature2, color_weight=0.5, patch_weight=0.5):
    """
    Computes a weighted distance between two combined color and patch feature vectors.
    Assumes feature vectors are concatenated as [color_feature (96 elements), patch_feature (3072 elements)].
    Both feature components are expected to be normalized to 0-1 range.
    """
    color_dim = 32 * 3 # 32 bins per channel * 3 channels
    color1 = feature1[:color_dim]
    patch1 = feature1[color_dim:]

    color2 = feature2[:color_dim]
    patch2 = feature2[color_dim:]

    color_dist = euclidean_distance(color1, color2)
    patch_dist = euclidean_distance(patch1, patch2)

    return color_weight * color_dist + patch_weight * patch_dist

def compute_homography(src_pts, dst_pts):
    """
    Computes the homography matrix from source and destination points.
    src_pts: list of [x, y] points in the source image.
    dst_pts: list of [x, y] corresponding points in the destination image.
    Returns: 3x3 homography matrix.
    """
    src_pts = np.array(src_pts).reshape(-1, 1, 2).astype(np.float32)
    dst_pts = np.array(dst_pts).reshape(-1, 1, 2).astype(np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

def project_points(points, H):
    """
    Projects points using a homography matrix.
    points: list of [x, y] points.
    H: 3x3 homography matrix.
    Returns: list of projected [x, y] points.
    """
    points_np = np.array(points).reshape(-1, 1, 2).astype(np.float32)
    projected_points = cv2.perspectiveTransform(points_np, H)
    return projected_points.reshape(-1, 2).tolist()

def aggregate_features(tracked_features_list_of_lists):
    """
    Aggregate features over multiple frames for each unique player (track_id).
    Args:
        tracked_features_list_of_lists: list of lists, where each inner list contains (track_id, feature_vector) tuples for a frame
    Returns:
        tuple: (list of unique track_ids, list of aggregated_feature_vectors)
    """
    all_features_by_track_id = {}
    for frame_tracked_features in tracked_features_list_of_lists:
        for track_id, feature in frame_tracked_features:
            if track_id not in all_features_by_track_id:
                all_features_by_track_id[track_id] = []
            all_features_by_track_id[track_id].append(feature)

    unique_track_ids = sorted(all_features_by_track_id.keys())
    aggregated_feature_vectors = []

    for track_id in unique_track_ids:
        features_for_this_track = np.array(all_features_by_track_id[track_id])
        if len(features_for_this_track) > 0:
            # For lightweight features, use mean aggregation
            aggregated_feature_vectors.append(np.mean(features_for_this_track, axis=0))
        else:
            # This case should ideally not happen if tracks are confirmed with features
            # If it happens, we need a default feature vector size (32*3 for color hist + 32*32*3 for patch)
            aggregated_feature_vectors.append(np.zeros((32*3) + (32*32*3)))

    return unique_track_ids, np.array(aggregated_feature_vectors)

def match_players(features_a, features_b, bboxes_a, bboxes_b, H_broadcast_to_tacticam, H_tacticam_to_broadcast, threshold, spatial_weight, color_weight, patch_weight):
    """
    Matches players between two sets of features using the Hungarian algorithm (linear_sum_assignment).
    Incorporates spatial constraints using homography.

    features_a, features_b: list of aggregated feature vectors
    bboxes_a, bboxes_b: list of representative bounding boxes (e.g., from first frame or mean bbox) corresponding to features.
    H_broadcast_to_tacticam: Homography matrix to project points from broadcast to tacticam view.
    H_tacticam_to_broadcast: Homography matrix to project points from tacticam to broadcast view.
    threshold: Matching threshold for appearance cost (lower is better).
    spatial_weight: Weight for spatial distance in the combined cost.
    color_weight: Weight for color feature in combined distance.
    patch_weight: Weight for patch feature in combined distance.
    Returns: list of (idx_a, idx_b) pairs representing optimal matches.
    """
    if len(features_a) == 0 or len(features_b) == 0:
        return []

    # Calculate combined cost matrix using weighted_feature_distance
    appearance_cost_matrix = np.zeros((len(features_a), len(features_b)))
    for i, feat_a in enumerate(features_a):
        for j, feat_b in enumerate(features_b):
            appearance_cost_matrix[i, j] = weighted_feature_distance(feat_a, feat_b, color_weight=color_weight, patch_weight=patch_weight)

    # Calculate spatial cost matrix
    spatial_cost_matrix = np.zeros_like(appearance_cost_matrix)
    if H_broadcast_to_tacticam is not None and H_tacticam_to_broadcast is not None and len(bboxes_a) > 0 and len(bboxes_b) > 0:
        # Get center points of bboxes
        centers_a = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in bboxes_a])
        centers_b = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in bboxes_b])

        # Project centers from A to B
        projected_a_to_b = project_points(centers_a, H_broadcast_to_tacticam)
        
        for i, proj_center_a in enumerate(projected_a_to_b):
            # Calculate Euclidean distance between projected point and actual points in B
            distances = np.linalg.norm(centers_b - proj_center_a, axis=1)
            # Normalize distances by max distance in current scene to scale 0-1
            max_dist = np.max(distances) if np.max(distances) > 0 else 1.0
            spatial_cost_matrix[i, :] = distances / max_dist

        # Combine appearance and spatial costs
        # Max possible weighted_feature_distance after normalization:
        # max_color_dist = sqrt(32 * 3 * 1^2) = sqrt(96) = 9.79
        # max_patch_dist = sqrt(32 * 32 * 3 * 1^2) = sqrt(3072) = 55.42
        # max_weighted_dist = 0.5 * 9.79 + 0.5 * 55.42 = 32.605
        max_appearance_dist = 33.0 # A more accurate maximum based on normalized features
        appearance_cost_matrix_normalized = appearance_cost_matrix / max_appearance_dist

        cost_matrix = (1 - spatial_weight) * appearance_cost_matrix_normalized + spatial_weight * spatial_cost_matrix
    else:
        # If no homography, only use appearance cost. Normalize it as well.
        max_appearance_dist = 33.0 # A more accurate maximum based on normalized features
        max_appearance_dist_current = np.max(appearance_cost_matrix) if np.max(appearance_cost_matrix) > 0 else 1.0
        cost_matrix = appearance_cost_matrix / max_appearance_dist

    # Apply Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        # For distance-based matching, we check if cost is *below* a threshold
        if appearance_cost_matrix[r, c] < threshold:
            matches.append((r, c))
            
    return matches 