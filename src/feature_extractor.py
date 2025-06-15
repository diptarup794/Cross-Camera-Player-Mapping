import cv2
import numpy as np

def get_dominant_color(image_bgr):
    """
    Extracts the dominant color from a BGR image.
    Returns: A 3-element numpy array representing the dominant BGR color.
    """
    # Reshape the image to be a list of pixels
    pixels = image_bgr.reshape(-1, 3)

    # Convert to float32 for k-means
    pixels = np.float32(pixels)

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    K = 1  # We want only the dominant color
    _, labels, palette = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # The palette contains the dominant colors. For K=1, it's just one color.
    dominant_color = palette[0]
    return dominant_color

def extract_color_histogram(image_bgr, bbox, hist_bins=32):
    """
    Extracts a color histogram from the cropped player image.
    image_bgr: Input image in BGR format.
    bbox: [x1, y1, x2, y2] bounding box.
    hist_bins: Number of bins for each color channel.
    Returns: 1D numpy array (feature vector) of concatenated normalized histograms.
    """
    x1, y1, x2, y2 = map(int, bbox)
    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return np.zeros(hist_bins * 3) # Return zero vector for empty crop

    hist = []
    for i in range(3): # For each B, G, R channel
        h = cv2.calcHist([crop], [i], None, [hist_bins], [0, 256])
        h = cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten() # Normalize to 0-1
        hist.append(h)
    return np.concatenate(hist)

def extract_image_patch(image, bbox, target_size=(32, 32)):
    """
    Extracts a cropped and resized image patch from the player's bounding box.
    bbox: [x1, y1, x2, y2]
    target_size: Desired output size of the patch (width, height)
    Returns: 1D numpy array of the flattened and resized image patch.
    """
    x1, y1, x2, y2 = map(int, bbox)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return np.zeros(target_size[0] * target_size[1] * 3) # Return zero vector for empty crop

    # Resize the image patch
    resized_patch = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    # Flatten and normalize the pixel values
    flattened_patch = resized_patch.flatten().astype(np.float32) / 255.0
    return flattened_patch

def extract_player_feature(image, bbox, method='combined'): 
    """
    Extracts features from a player image using the specified method.
    Args:
        image: Input image (BGR format).
        bbox: Bounding box [x1, y1, x2, y2].
        method: Feature extraction method ('combined', 'hist', or 'patch').
    Returns:
        Feature vector as numpy array.
    """
    x1, y1, x2, y2 = map(int, bbox)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        # Default size if method is 'hist': 32*3 = 96. If 'patch': 32*32*3 = 3072. If 'combined': 96 + 3072 = 3168
        if method == 'hist':
            return np.zeros(32 * 3)
        elif method == 'patch':
            return np.zeros(32 * 32 * 3)
        else: # combined
            return np.zeros((32 * 3) + (32 * 32 * 3)) 

    if method == 'combined':
        hist_feature = extract_color_histogram(image, bbox) # Use original image and bbox for cropping
        patch_feature = extract_image_patch(image, bbox)  
        return np.concatenate((hist_feature, patch_feature))
    elif method == 'hist':
        return extract_color_histogram(image, bbox)
    elif method == 'patch':
        return extract_image_patch(image, bbox)  
    else:
        raise ValueError(f'Unknown feature extraction method: {method}') 