import cv2
import os
import argparse

def extract_frame(video_path, frame_number, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame {frame_number} from {video_path}")
        cap.release()
        return None

    output_filename = os.path.join(output_dir, f"frame_{os.path.basename(video_path).split('.')[0]}_{frame_number}.png")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_filename, frame)
    print(f"Saved frame {frame_number} from {video_path} to {output_filename}")
    cap.release()
    return output_filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a specific frame from a video for calibration.')
    parser.add_argument('--broadcast_video', type=str, required=True, help='Path to the broadcast video file.')
    parser.add_argument('--tacticam_video', type=str, required=True, help='Path to the tacticam video file.')
    parser.add_argument('--frame_number', type=int, default=100, help='The frame number to extract (0-indexed).')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save the extracted frames.')
    
    args = parser.parse_args()

    print("Extracting frame from broadcast video...")
    extract_frame(args.broadcast_video, args.frame_number, args.output_dir)
    
    print("Extracting frame from tacticam video...")
    extract_frame(args.tacticam_video, args.frame_number, args.output_dir)

    print("\n--- Frame Extraction Complete ---")
    print("Please open the saved PNG images in the 'outputs/' directory.")
    print("Manually identify at least 4 corresponding static points on the field (e.g., corner flags, penalty spots).")
    print("Note down their pixel coordinates (x, y) for both images. You'll need these for camera calibration.")
    print("---------------------------------") 