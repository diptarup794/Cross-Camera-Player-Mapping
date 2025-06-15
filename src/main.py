import argparse
import os
import cv2
import numpy as np
from detector import YOLOv11Detector
from tracker import SimpleSORT
from feature_extractor import extract_player_feature
from reid import match_players, aggregate_features, compute_homography, project_points, weighted_feature_distance
from utils import draw_boxes, save_video, make_side_by_side, plot_similarity_matrix, plot_ground_plane_matches, get_player_color
import pandas as pd

def process_video(video_path, detector, tracker, feature_method='combined'):
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    all_tracked_data = [] # Store (frame_num, track_id, bbox_ltrb, feature)
    initial_bboxes_per_track = {} # Store first observed bbox for each track_id
    track_id_first_frame_map = {} # Store first frame number for each track_id
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame.copy()) # Store raw frame

        detections = detector.detect(frame)
        
        player_boxes_for_tracker = [d['bbox'] for d in detections if d['label'] == 'player']

        tracked_objects = tracker.update(player_boxes_for_tracker)
        
        tracked_data_in_frame = []
        for track_id, bbox_ltrb in tracked_objects:
            if track_id not in initial_bboxes_per_track: # Store initial bbox and its frame number
                initial_bboxes_per_track[track_id] = bbox_ltrb.tolist()
                track_id_first_frame_map[track_id] = frame_num

            feature = extract_player_feature(frame, bbox_ltrb, method=feature_method)
            tracked_data_in_frame.append((track_id, bbox_ltrb, feature))
        
        all_tracked_data.append(tracked_data_in_frame)
        frame_num += 1

    cap.release()
    return raw_frames, all_tracked_data, initial_bboxes_per_track, track_id_first_frame_map

def main(args):
    detector = YOLOv11Detector(args.model)
    tracker_a = SimpleSORT()
    tracker_b = SimpleSORT()

    # Process both videos
    raw_frames_a, all_tracked_data_a, initial_bboxes_a_dict, track_id_first_frame_map_a = process_video(args.broadcast, detector, tracker_a, feature_method=args.feature)
    raw_frames_b, all_tracked_data_b, initial_bboxes_b_dict, track_id_first_frame_map_b = process_video(args.tacticam, detector, tracker_b, feature_method=args.feature)

    # Extract features for aggregation
    tracked_feats_a_for_agg = [[(tid, feat) for tid, bbox, feat in frame_data] for frame_data in all_tracked_data_a]
    tracked_feats_b_for_agg = [[(tid, feat) for tid, bbox, feat in frame_data] for frame_data in all_tracked_data_b]

    # Aggregate features over frames for each player
    unique_ids_a, agg_feats_a = aggregate_features(tracked_feats_a_for_agg) 
    unique_ids_b, agg_feats_b = aggregate_features(tracked_feats_b_for_agg) 

    # Prepare representative bboxes for matching (aligned with aggregated features)
    representative_bboxes_a = [initial_bboxes_a_dict.get(uid) for uid in unique_ids_a if uid in initial_bboxes_a_dict]
    representative_bboxes_b = [initial_bboxes_b_dict.get(uid) for uid in unique_ids_b if uid in initial_bboxes_b_dict]

    # Calibration Points from outputs/calibration_points.txt
    broadcast_calibration_points = [[1743, 496], [1173, 621], [616, 747], [167, 857]]
    tacticam_calibration_points = [[404, 146], [171, 311], [8, 427], [91, 496]]

    H_b_to_t = None
    H_t_to_b = None
    if len(broadcast_calibration_points) >= 4 and len(tacticam_calibration_points) >= 4:
        H_b_to_t = compute_homography(broadcast_calibration_points, tacticam_calibration_points)
        H_t_to_b = compute_homography(tacticam_calibration_points, broadcast_calibration_points)
        print("Homography matrices computed successfully.")
    else:
        print("Warning: Not enough calibration points provided. Spatial constraints will not be used for matching.")

    # Cross-view matching with spatial constraints
    matches = match_players(agg_feats_a, agg_feats_b, 
                            representative_bboxes_a, representative_bboxes_b, 
                            H_b_to_t, H_t_to_b, 
                            spatial_weight=args.spatial_weight, 
                            threshold=args.match_threshold, 
                            color_weight=args.color_weight, 
                            patch_weight=args.patch_weight)
    mapping = []
    assigned_global_ids = {}
    global_player_id_counter = 0

    for idx_a, idx_b in matches:
        broadcast_actual_id = unique_ids_a[idx_a]
        tacticam_actual_id = unique_ids_b[idx_b]

        # Assign a consistent global player_id
        if broadcast_actual_id in assigned_global_ids:
            current_global_id = assigned_global_ids[broadcast_actual_id]
        elif tacticam_actual_id in assigned_global_ids:
            current_global_id = assigned_global_ids[tacticam_actual_id]
        else:
            current_global_id = global_player_id_counter
            global_player_id_counter += 1
        
        assigned_global_ids[broadcast_actual_id] = current_global_id
        assigned_global_ids[tacticam_actual_id] = current_global_id

        # Add frame numbers to mapping
        mapping.append({
            'broadcast_id': broadcast_actual_id, 
            'tacticam_id': tacticam_actual_id, 
            'player_id': current_global_id,
            'first_frame_broadcast': track_id_first_frame_map_a.get(broadcast_actual_id, -1),
            'first_frame_tacticam': track_id_first_frame_map_b.get(tacticam_actual_id, -1)
        })

    # Save mapping
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(mapping).to_csv(os.path.join(args.output_dir, 'player_id_mapping.csv'), index=False)

    # Prepare annotated frames for video output
    annotated_frames_a = []
    annotated_frames_b = []

    for frame_idx, frame_data_a in enumerate(all_tracked_data_a):
        frame_a_copy = raw_frames_a[frame_idx].copy()
        boxes_to_draw_a = []
        ids_to_draw_a = []
        colors_to_draw_a = []
        
        for track_id, bbox_ltrb, feature in frame_data_a:
            global_id = assigned_global_ids.get(track_id, track_id) # Use track_id if not globally assigned
            color = get_player_color(global_id)
            boxes_to_draw_a.append(bbox_ltrb)
            ids_to_draw_a.append(global_id)
            colors_to_draw_a.append(color)
        
        # Draw all boxes for the current frame
        for i in range(len(boxes_to_draw_a)):
            frame_a_copy = draw_boxes(frame_a_copy, [boxes_to_draw_a[i]], [ids_to_draw_a[i]], color=colors_to_draw_a[i])
        annotated_frames_a.append(frame_a_copy)

    for frame_idx, frame_data_b in enumerate(all_tracked_data_b):
        frame_b_copy = raw_frames_b[frame_idx].copy()
        boxes_to_draw_b = []
        ids_to_draw_b = []
        colors_to_draw_b = []
        
        for track_id, bbox_ltrb, feature in frame_data_b:
            global_id = assigned_global_ids.get(track_id, track_id) # Use track_id if not globally assigned
            color = get_player_color(global_id)
            boxes_to_draw_b.append(bbox_ltrb)
            ids_to_draw_b.append(global_id)
            colors_to_draw_b.append(color)

        # Draw all boxes for the current frame
        for i in range(len(boxes_to_draw_b)):
            frame_b_copy = draw_boxes(frame_b_copy, [boxes_to_draw_b[i]], [ids_to_draw_b[i]], color=colors_to_draw_b[i])
        annotated_frames_b.append(frame_b_copy)

    # Save annotated videos
    save_video(annotated_frames_a, os.path.join(args.output_dir, 'broadcast_annotated.mp4'))
    save_video(annotated_frames_b, os.path.join(args.output_dir, 'tacticam_annotated.mp4'))
    make_side_by_side(annotated_frames_a, annotated_frames_b, os.path.join(args.output_dir, 'side_by_side.mp4'))

    # Save similarity matrix plot (now a distance matrix)
    if len(agg_feats_a) > 0 and len(agg_feats_b) > 0:
        appearance_cost_matrix = np.zeros((len(agg_feats_a), len(agg_feats_b)))
        for i, feat_a in enumerate(agg_feats_a):
            for j, feat_b in enumerate(agg_feats_b):
                appearance_cost_matrix[i, j] = weighted_feature_distance(feat_a, feat_b, 
                                                                         color_weight=args.color_weight, 
                                                                         patch_weight=args.patch_weight)
        
        plot_similarity_matrix(appearance_cost_matrix, os.path.join(args.output_dir, 'feature_distance_matrix.png'), title="Feature Distance Matrix (Lower is Better)")
    else:
        print("Cannot plot feature distance matrix: Not enough aggregated features for comparison.")

    # Plot ground plane matches
    if len(matches) > 0 and H_b_to_t is not None and H_t_to_b is not None:
        plot_ground_plane_matches(unique_ids_a, unique_ids_b, 
                                  representative_bboxes_a, representative_bboxes_b, 
                                  matches, H_b_to_t, H_t_to_b, 
                                  os.path.join(args.output_dir, 'ground_plane_matches.png'), 
                                  assigned_global_ids)
    else:
        print("Cannot plot ground plane matches: No matches found or homography not available.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Player Re-Identification Pipeline (Lightweight)')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv11 weights')
    parser.add_argument('--broadcast', type=str, required=True, help='Path to broadcast video')
    parser.add_argument('--tacticam', type=str, required=True, help='Path to tacticam video')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--feature', type=str, default='combined', choices=['combined', 'hist', 'patch'], help='Feature extraction method')
    parser.add_argument('--color_weight', type=float, default=0.5, help='Weight for color feature in combined distance')
    parser.add_argument('--patch_weight', type=float, default=0.5, help='Weight for patch feature in combined distance')
    parser.add_argument('--match_threshold', type=float, default=10.0, help='Threshold for player matching (lower is better)')
    parser.add_argument('--spatial_weight', type=float, default=0.5, help='Weight for spatial distance in matching')
    args = parser.parse_args()
    main(args) 