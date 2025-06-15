import cv2
import numpy as np
import os
import argparse

# Global variables to store points and current image
clicked_points = []
current_image = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, current_image
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        # Draw a small circle at the clicked point
        cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Calibration Image', current_image)
        print(f"Point added: ({x}, {y})")

def collect_points_interactive(video_path, frame_number, num_points_to_collect, video_name):
    global clicked_points, current_image
    clicked_points = [] # Reset points for new image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_number} from {video_path}")
        return None

    current_image = frame.copy()
    cv2.namedWindow('Calibration Image')
    cv2.setMouseCallback('Calibration Image', mouse_callback)

    print(f"\n--- Collecting points for: {video_name} (Frame {frame_number}) ---")
    print(f"Click {num_points_to_collect} corresponding static points on the field.")
    print("Press 'q' to finish collecting points for this image.")

    while True:
        cv2.imshow('Calibration Image', current_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(clicked_points) >= num_points_to_collect: # Allow finishing early or after required points
            break

    cv2.destroyAllWindows()
    return clicked_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive tool for collecting calibration points from video frames.')
    parser.add_argument('--broadcast_video', type=str, required=True, help='Path to the broadcast video file.')
    parser.add_argument('--tacticam_video', type=str, required=True, help='Path to the tacticam video file.')
    parser.add_argument('--frame_numbers', type=str, default='100', 
                        help='Comma-separated list of frame numbers to extract (0-indexed). E.g., "100,500,1000"')
    parser.add_argument('--num_points', type=int, default=4, help='Number of corresponding points to collect PER FRAME (minimum 4).')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save the calibration points.')
    
    args = parser.parse_args()

    if args.num_points < 4:
        print("Warning: It is recommended to collect at least 4 points for homography calculation.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse frame numbers
    frame_numbers_list = [int(f) for f in args.frame_numbers.split(',')]

    all_broadcast_coords = []
    all_tacticam_coords = []

    for frame_num in frame_numbers_list:
        # Collect points for broadcast video
        broadcast_coords_for_frame = collect_points_interactive(args.broadcast_video, frame_num, args.num_points, "Broadcast Video")
        all_broadcast_coords.extend(broadcast_coords_for_frame)
        
        # Collect points for tacticam video
        tacticam_coords_for_frame = collect_points_interactive(args.tacticam_video, frame_num, args.num_points, "Tacticam Video")
        all_tacticam_coords.extend(tacticam_coords_for_frame)

    # Save aggregated coordinates to a file
    output_filepath = os.path.join(args.output_dir, 'calibration_points.txt')
    with open(output_filepath, 'w') as f:
        f.write(f"# Calibration Points (broadcast_video, tacticam_video)\n")
        f.write(f"# Frame Numbers Used: {args.frame_numbers}\n\n")
        f.write(f"broadcast_points = {all_broadcast_coords}\n")
        f.write(f"tacticam_points = {all_tacticam_coords}\n")

    print(f"\n--- Calibration Points Saved to: {output_filepath} ---")
    print("You can now use these points to compute the homography in the main pipeline.")
    print("---------------------------------------------------") 