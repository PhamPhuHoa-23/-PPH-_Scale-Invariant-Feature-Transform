import numpy as np
from typing import List
from src.utils import *

class OrientationAssigner:
    def __init__(self,
                 num_bins: int = 36,
                 peak_ratio: float = 0.8,
                 scale_multiplier: float = 3,
                 radius_factor: float = 3):
        """
        Initialize orientation parameters
        Args:
            num_bins: Number of orientation histogram bins
            peak_ratio: Only keep peaks > peak_ratio * max_peak
            scale_multiplier: Used to determine Gaussian window size
            radius_factor: Used to determine region radius
        """
        self.num_bins = num_bins
        self.peak_ratio = peak_ratio
        self.scale_multiplier = scale_multiplier
        self.radius_factor = radius_factor

    def compute_orientation(self, keypoint: dict, gaussian_image: np.ndarray) -> List[dict]:
        """
        Compute orientations for each keypoint
        Args:
            keypoint: Keypoint information
            gaussian_image: Corresponding Gaussian blurred image
        Returns:
            List of keypoints (can be multiple if there are auxiliary peaks)
        """
        keypoints_with_orientations = []
        image_shape = gaussian_image.shape

        # Compute window size based on scale
        scale = keypoint['sigma'] * self.scale_multiplier
        radius = int(round(self.radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)

        # Initialize histogram
        raw_histogram = np.zeros(self.num_bins)
        smooth_histogram = np.zeros(self.num_bins)

        # Compute contribution of each pixel
        for i in range(-radius, radius + 1):
            region_y = int(round(keypoint['y'])) + i
            if region_y > 0 and region_y < image_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(round(keypoint['x'])) + j
                    if region_x > 0 and region_x < image_shape[1] - 1:
                        # Compute gradient
                        dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                        dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                        
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        
                        # Compute weight and bin
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                        bin_index = int(round(gradient_orientation * self.num_bins / 360.0)) % self.num_bins
                        raw_histogram[bin_index] += weight * gradient_magnitude

        # Smooth histogram
        for n in range(self.num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] + 
                                 4 * (raw_histogram[(n - 1) % self.num_bins] + 
                                     raw_histogram[(n + 1) % self.num_bins]) + 
                                 raw_histogram[(n - 2) % self.num_bins] + 
                                 raw_histogram[(n + 2) % self.num_bins]) / 16.

        # Find peaks in smoothed histogram
        orientation_max = np.max(smooth_histogram)
        orientation_peaks = self._find_peaks(smooth_histogram)
        
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= self.peak_ratio * orientation_max:
                # Fit parabola to 3 points centered on peak
                left_value = smooth_histogram[(peak_index - 1) % self.num_bins]
                right_value = smooth_histogram[(peak_index + 1) % self.num_bins]
                
                # Quadratic interpolation for peak position
                interpolated_peak_index = (peak_index + 
                    0.5 * (left_value - right_value) / 
                    (left_value - 2 * peak_value + right_value)) % self.num_bins
                
                # Convert to degrees
                orientation = 360. - (interpolated_peak_index * 360. / self.num_bins)
                if abs(orientation - 360.) < float_tolerance:
                    orientation = 0
                
                # Create new keypoint with this orientation
                new_keypoint = keypoint.copy()
                new_keypoint['orientation'] = orientation
                keypoints_with_orientations.append(new_keypoint)

        return keypoints_with_orientations

    def _find_peaks(self, histogram: np.ndarray) -> List[int]:
        """Find peaks in histogram"""
        peaks = []
        for i in range(len(histogram)):
            if (histogram[i] > histogram[(i - 1) % len(histogram)] and 
                histogram[i] > histogram[(i + 1) % len(histogram)]):
                peaks.append(i)
        return peaks

# Helper constant
float_tolerance = 1e-7