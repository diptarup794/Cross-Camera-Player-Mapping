import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

# IoU function (Intersection over Union)
def iou_batch(bb_test, bb_gt):
    """
    Computes IoU between two sets of bboxes in [x1,y1,x2,y2] format
    """
    x1 = np.maximum(bb_test[0], bb_gt[:, 0])
    y1 = np.maximum(bb_test[1], bb_gt[:, 1])
    x2 = np.minimum(bb_test[2], bb_gt[:, 2])
    y2 = np.minimum(bb_test[3], bb_gt[:, 3])

    w = np.maximum(0., x2 - x1)
    h = np.maximum(0., y2 - y1)
    wh = w * h

    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + 
              (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1]) - wh)
    return o

class Track:
    """
    A single track managed by the SORT tracker with a custom Kalman filter implementation.
    """
    def __init__(self, track_id, bbox, feature=None):
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature # Currently unused by SimpleSORT internal logic
        self.hits = 1
        self.no_losses = 0
        self.trace = deque(maxlen=30) # For visualization of past positions

        # Custom Kalman filter implementation
        # State vector: [x, y, w, h, dx, dy, dw, dh]
        # x, y: center of bbox
        # w, h: width, height
        self.x = np.zeros((8, 1)) # State vector
        self.P = np.eye(8) * 1000. # Covariance matrix

        # State transition matrix (linear motion model)
        self.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        
        # Measurement function (maps state to measurement: [x, y, w, h])
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32)

        # Measurement noise covariance matrix (R)
        self.R = np.eye(4) * 10. # Increased noise for bbox measurements
        self.R[2:, 2:] *= 10. # Especially for width/height

        # Process noise covariance matrix (Q)
        self.Q = np.eye(8) * 0.1 # Small noise for state prediction
        self.Q[4:, 4:] *= 0.01 # Even smaller for velocities
        
        # Initialize state from bbox
        self.x[:4] = self._convert_bbox_to_z(bbox)

    def _convert_bbox_to_z(self, bbox):
        """
        Converts bbox [x1, y1, x2, y2] to measurement vector [x_center, y_center, width, height]
        """
        return np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.,
                         bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float32).reshape((4, 1))

    def _convert_x_to_bbox(self, x):
        """
        Converts state vector x to bbox [x1, y1, x2, y2]
        """
        return np.array([x[0, 0] - x[2, 0] / 2., x[1, 0] - x[3, 0] / 2.,
                         x[0, 0] + x[2, 0] / 2., x[1, 0] + x[3, 0] / 2.], dtype=np.float32).flatten()

    def predict(self):
        """
        Predicts the next bounding box position and updates the Kalman filter.
        """
        self.x = np.dot(self.F, self.x) # Predict state
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q # Predict covariance
        self.bbox = self._convert_x_to_bbox(self.x)
        self.trace.append(self.bbox)

    def update(self, bbox):
        """
        Updates the Kalman filter with the new observed bounding box.
        """
        z = self._convert_bbox_to_z(bbox) # Measurement

        y = z - np.dot(self.H, self.x) # Innovation
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R # Innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman gain

        self.x = self.x + np.dot(K, y) # Update state
        self.P = self.P - np.dot(np.dot(K, self.H), self.P) # Update covariance

        self.bbox = self._convert_x_to_bbox(self.x)
        self.hits += 1
        self.no_losses = 0 # Reset consecutive losses
        self.trace.append(self.bbox)

class SimpleSORT:
    """
    Enhanced SORT tracker with custom Kalman filter for more stable tracking.
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0

    def update(self, detections):
        """
        Updates the tracker with new detections.
        detections: list of [x1, y1, x2, y2] bounding boxes.
        Returns: list of (track_id, bbox) for confirmed tracks.
        """
        # Predict new locations of existing tracks
        for i, track in enumerate(self.tracks):
            track.predict()

        # Get predicted bounding boxes from tracks
        trks = np.array([t.bbox for t in self.tracks if t.no_losses < self.max_age])

        if len(trks) == 0: # No tracks to match
            matched = np.empty((0, 2), dtype=int)
            unmatched_detections = np.arange(len(detections))
            unmatched_tracks = np.empty((0, ), dtype=int)
        else:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(detections), len(trks)), dtype=np.float32)
            for d, det in enumerate(detections):
                for t, trk in enumerate(trks):
                    iou_matrix[d, t] = iou_batch(det, np.array([trk]))[0] # Compare one det to one trk

            # Convert IoU to cost (1 - IoU) for Hungarian algorithm
            cost_matrix = 1 - iou_matrix

            # Apply Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched = []
            unmatched_detections = []
            for d, det in enumerate(detections):
                if d not in row_ind: # If detection is not in row_ind, it's unmatched
                    unmatched_detections.append(d)

            unmatched_tracks = []
            for t, trk in enumerate(trks):
                if t not in col_ind: # If track is not in col_ind, it's unmatched
                    unmatched_tracks.append(t)

            # Filter out matches below IoU threshold
            matches_above_threshold = []
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] > self.iou_threshold:
                    matches_above_threshold.append((r, c))
                else:
                    unmatched_detections.append(r) # If below threshold, detection is unmatched
                    unmatched_tracks.append(c) # And track is unmatched
            matched = np.array(matches_above_threshold)
            
        # Update matched tracks
        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
            self.tracks[track_idx].no_losses = 0
            
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].no_losses += 1
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self.track_id_count += 1
            self.tracks.append(Track(self.track_id_count, detections[det_idx]))

        # Return confirmed tracks
        confirmed_tracks = []
        for t in self.tracks:
            if t.hits >= self.min_hits and t.no_losses < self.max_age:
                confirmed_tracks.append((t.track_id, t.bbox))
        
        # Clean up old tracks (optional, but good for memory)
        self.tracks = [t for t in self.tracks if t.no_losses < self.max_age]

        return confirmed_tracks 