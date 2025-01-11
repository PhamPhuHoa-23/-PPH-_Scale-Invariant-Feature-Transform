import numpy as np
from typing import List, Tuple, Optional
import cv2

def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to image"""
    # Ensure kernel size is odd and large enough
    ksize = 2 * int(4 * sigma + 0.5) + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def downsample_image(image: np.ndarray) -> np.ndarray:
    """Downsample image by factor of 2 using nearest neighbor"""
    return image[::2, ::2]

def compute_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient magnitude and orientation
    Returns:
        Tuple of (magnitude, orientation in radians)
    """
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx)
    
    return magnitude, orientation

def rotate_image(image: np.ndarray, angle: float, center: Tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Rotate image around center point"""
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (cols, rows))

# Helper functions for SIFT descriptor

def unpack_octave(keypoint: dict) -> Tuple[int, int, float]:
    """
    Compute octave, layer, and scale from a keypoint
    Returns:
        octave: Octave index
        layer: Layer index
        scale: Scale factor
    """
    octave = keypoint['octave']
    layer = int(round(keypoint['interval']))
    scale = keypoint['sigma']
    return octave, layer, scale

def convert_keypoints_to_input_size(keypoints: List[dict]) -> List[dict]:
    """Convert keypoint coordinates back to input image size"""
    converted_keypoints = []
    for kp in keypoints:
        new_kp = kp.copy()
        new_kp['x'] *= 0.5
        new_kp['y'] *= 0.5
        new_kp['sigma'] *= 0.5
        converted_keypoints.append(new_kp)
    return converted_keypoints

def normalize_histogram(hist: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Normalize histogram so it sums to 1"""
    hist_sum = np.sum(hist)
    if hist_sum > eps:
        hist = hist / hist_sum
    return hist