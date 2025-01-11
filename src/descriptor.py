import numpy as np
from typing import List
from src.utils import *

class DescriptorExtractor:
    def __init__(self,
                 window_width: int = 4,
                 num_bins: int = 8,
                 scale_multiplier: float = 3,
                 descriptor_max_value: float = 0.2):
        """
        Initialize descriptor parameters
        Args:
            window_width: Width of descriptor window (4 means 4x4 windows)
            num_bins: Number of bins in each histogram
            scale_multiplier: Used for window size
            descriptor_max_value: Value to clip descriptor elements
        """
        self.window_width = window_width
        self.num_bins = num_bins
        self.scale_multiplier = scale_multiplier
        self.descriptor_max_value = descriptor_max_value

    def compute_descriptors(self, keypoints: List[dict], gaussian_images: List[List[np.ndarray]]) -> np.ndarray:
        """
        Compute descriptors for each keypoint
        Args:
            keypoints: List of keypoints
            gaussian_images: Gaussian pyramid
        Returns:
            Array of descriptors (N x 128)
        """
        descriptors = []

        for keypoint in keypoints:
            octave = keypoint['octave']
            layer = int(round(keypoint['interval']))
            gaussian_image = gaussian_images[octave][layer]
            
            point = np.round(np.array([keypoint['x'], keypoint['y']])).astype('int')
            bins_per_degree = self.num_bins / 360.
            angle = 360. - keypoint['orientation']
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            
            # Radius of descriptor window
            weight_scale = keypoint['sigma'] * self.scale_multiplier
            radius = self.window_width * 0.5 * np.sqrt(2) * (weight_scale * 1.5)
            radius = int(min(radius, np.sqrt(gaussian_image.shape[0]**2 + gaussian_image.shape[1]**2)))
            
            weight_multiplier = -0.5 / ((0.5 * self.window_width) ** 2)
            
            # Initialize arrays
            hist_tensor = np.zeros((self.window_width + 2, self.window_width + 2, self.num_bins))
            hist_width = self.scale_multiplier * 0.5 * weight_scale

            # Collect samples in region around keypoint
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    # Rotate coordinates
                    rot_j = j * cos_angle - i * sin_angle
                    rot_i = j * sin_angle + i * cos_angle
                    
                    # Get position in descriptor array
                    window_r = (rot_i / hist_width) + self.window_width/2 - 0.5
                    window_c = (rot_j / hist_width) + self.window_width/2 - 0.5
                    
                    if (window_r > -1 and window_r < self.window_width and 
                        window_c > -1 and window_c < self.window_width):
                        # Get image coordinates
                        img_r = int(round(point[1] + i))
                        img_c = int(round(point[0] + j))
                        
                        if (img_r > 0 and img_r < gaussian_image.shape[0]-1 and 
                            img_c > 0 and img_c < gaussian_image.shape[1]-1):
                            dx = gaussian_image[img_r, img_c+1] - gaussian_image[img_r, img_c-1]
                            dy = gaussian_image[img_r-1, img_c] - gaussian_image[img_r+1, img_c]
                            
                            gradient_magnitude = np.sqrt(dx*dx + dy*dy)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            
                            # Weight by gaussian
                            weight = np.exp(weight_multiplier * 
                                          ((rot_i/hist_width)**2 + (rot_j/hist_width)**2))
                            
                            # Interpolate into histogram
                            self._add_sample_to_histogram(hist_tensor, window_r, window_c,
                                                        gradient_magnitude * weight,
                                                        gradient_orientation - angle,
                                                        bins_per_degree)

            # Get descriptor vector from histogram
            descriptor_vector = self._get_descriptor_vector(hist_tensor)
            
            # Normalize and threshold
            descriptor_vector = self._normalize_descriptor(descriptor_vector)
            
            descriptors.append(descriptor_vector)

        return np.array(descriptors)

    def _add_sample_to_histogram(self, hist_tensor: np.ndarray,
                               r: float, c: float,
                               magnitude: float,
                               orientation: float,
                               bins_per_degree: float):
        """Add weighted sample to 3D histogram"""
        r_floor, c_floor = np.floor([r, c]).astype(int)
        orientation_bin = orientation * bins_per_degree
        orientation_floor = int(np.floor(orientation_bin))
        
        r_frac = r - r_floor
        c_frac = c - c_floor
        o_frac = orientation_bin - orientation_floor
        
        if orientation_floor < 0:
            orientation_floor += self.num_bins
        if orientation_floor >= self.num_bins:
            orientation_floor -= self.num_bins

        # Trilinear interpolation
        for dr in range(2):
            r_weight = magnitude * (1 - r_frac if dr == 0 else r_frac)
            r_idx = r_floor + dr
            
            if r_idx >= 0 and r_idx < hist_tensor.shape[0]:
                for dc in range(2):
                    c_weight = r_weight * (1 - c_frac if dc == 0 else c_frac)
                    c_idx = c_floor + dc
                    
                    if c_idx >= 0 and c_idx < hist_tensor.shape[1]:
                        for do in range(2):
                            o_weight = c_weight * (1 - o_frac if do == 0 else o_frac)
                            o_idx = (orientation_floor + do) % self.num_bins
                            hist_tensor[r_idx, c_idx, o_idx] += o_weight

    def _get_descriptor_vector(self, hist_tensor: np.ndarray) -> np.ndarray:
        """Convert 3D histogram to descriptor vector"""
        return hist_tensor[1:-1, 1:-1, :].flatten()

    def _normalize_descriptor(self, descriptor: np.ndarray) -> np.ndarray:
        """Normalize descriptor and threshold large values"""
        # Normalize
        norm = np.linalg.norm(descriptor)
        if norm > float_tolerance:
            descriptor = descriptor / norm
        
        # Threshold
        descriptor = np.minimum(descriptor, self.descriptor_max_value)
        
        # Normalize again
        norm = np.linalg.norm(descriptor)
        if norm > float_tolerance:
            descriptor = descriptor / norm
            
        return descriptor

# Helper constant
float_tolerance = 1e-7 