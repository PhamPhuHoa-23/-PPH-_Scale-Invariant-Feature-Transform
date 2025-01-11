import numpy as np
from typing import List, Tuple
from src.utils import *
from src.scale_space import ScaleSpace

class KeypointDetector:
    def __init__(self, 
                 contrast_threshold: float = 0.04,
                 edge_threshold: float = 10.0,
                 border_width: int = 5):
        """
        Initialize detector parameters
        Args:
            contrast_threshold: Threshold for low contrast keypoints
            edge_threshold: Threshold for edge response
            border_width: How many pixels to ignore around the image border
        """
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.border_width = border_width

    def find_keypoints(self, gaussian_images: List[List[np.ndarray]], 
                      dog_images: List[List[np.ndarray]], 
                      num_intervals: int,
                      sigma: float) -> List[dict]:
        """
        Find scale-space extrema in the image pyramid
        """
        threshold = np.floor(0.5 * self.contrast_threshold / num_intervals * 255)
        keypoints = []

        for octave_index, dog_images_in_octave in enumerate(dog_images):
            for image_index in range(1, len(dog_images_in_octave) - 1):
                # Get 3 adjacent DoG images
                first_image = dog_images_in_octave[image_index - 1]
                second_image = dog_images_in_octave[image_index]
                third_image = dog_images_in_octave[image_index + 1]

                # Find extrema in 3x3x3 cube
                for i in range(self.border_width, first_image.shape[0] - self.border_width):
                    for j in range(self.border_width, first_image.shape[1] - self.border_width):
                        if self._is_extremum(first_image[i-1:i+2, j-1:j+2],
                                          second_image[i-1:i+2, j-1:j+2],
                                          third_image[i-1:i+2, j-1:j+2],
                                          threshold):
                            # Locate extremum with sub-pixel accuracy
                            localization_result = self._localize_extremum(
                                i, j, image_index, octave_index,
                                num_intervals, dog_images_in_octave,
                                sigma, threshold)
                            
                            if localization_result is not None:
                                keypoint = localization_result
                                keypoints.append(keypoint)

        return self._remove_duplicate_keypoints(keypoints)

    def _is_extremum(self, first_sub: np.ndarray, 
                    second_sub: np.ndarray, 
                    third_sub: np.ndarray,
                    threshold: float) -> bool:
        """
        Check if center pixel is extremum in 3x3x3 neighborhood
        """
        center_pixel_value = second_sub[1, 1]
        if abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                return np.all(center_pixel_value >= first_sub) and \
                       np.all(center_pixel_value >= third_sub) and \
                       np.all(center_pixel_value >= second_sub[0, :]) and \
                       np.all(center_pixel_value >= second_sub[2, :]) and \
                       center_pixel_value >= second_sub[1, 0] and \
                       center_pixel_value >= second_sub[1, 2]
            elif center_pixel_value < 0:
                return np.all(center_pixel_value <= first_sub) and \
                       np.all(center_pixel_value <= third_sub) and \
                       np.all(center_pixel_value <= second_sub[0, :]) and \
                       np.all(center_pixel_value <= second_sub[2, :]) and \
                       center_pixel_value <= second_sub[1, 0] and \
                       center_pixel_value <= second_sub[1, 2]
        return False

    def _localize_extremum(self, i: int, j: int, 
                          image_index: int,
                          octave_index: int, 
                          num_intervals: int,
                          dog_images_in_octave: List[np.ndarray],
                          sigma: float,
                          contrast_threshold: float,
                          eigenvalue_ratio: float = 10,
                          num_attempts: int = 5) -> dict:
        """
        Iteratively refine pixel positions of scale-space extrema via quadratic fit
        """
        extremum_is_outside_image = False
        image_shape = dog_images_in_octave[0].shape
        
        for _ in range(num_attempts):
            # Get 3x3x3 cube around the point
            first_image = dog_images_in_octave[image_index - 1]
            second_image = dog_images_in_octave[image_index]
            third_image = dog_images_in_octave[image_index + 1]

            pixel_cube = np.stack([
                first_image[i-1:i+2, j-1:j+2],
                second_image[i-1:i+2, j-1:j+2],
                third_image[i-1:i+2, j-1:j+2]
            ]).astype('float32') / 255.

            gradient = self._compute_gradient_at_center_pixel(pixel_cube)
            hessian = self._compute_hessian_at_center_pixel(pixel_cube)

            try:
                offset = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                return None

            if abs(offset[0]) < 0.5 and abs(offset[1]) < 0.5 and abs(offset[2]) < 0.5:
                break

            j += int(round(offset[0]))
            i += int(round(offset[1]))
            image_index += int(round(offset[2]))

            # Ensure we stay within image bounds
            if i < self.border_width or i >= image_shape[0] - self.border_width or \
               j < self.border_width or j >= image_shape[1] - self.border_width or \
               image_index < 1 or image_index > len(dog_images_in_octave) - 2:
                extremum_is_outside_image = True
                break

        if extremum_is_outside_image:
            return None

        # Reject unstable extrema
        function_value = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, offset)
        
        if abs(function_value) * num_intervals < contrast_threshold:
            return None

        # Principal curvature test
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        
        if xy_hessian_det <= 0 or \
           (xy_hessian_trace ** 2) / xy_hessian_det >= (eigenvalue_ratio + 1) ** 2 / eigenvalue_ratio:
            return None

        # Return keypoint
        keypoint = {
            'x': (j + offset[0]) * (2 ** octave_index),
            'y': (i + offset[1]) * (2 ** octave_index),
            'octave': octave_index,
            'interval': image_index + offset[2],
            'sigma': sigma * (2 ** ((image_index + offset[2]) / num_intervals)) * (2 ** octave_index)
        }

        return keypoint

    def _compute_gradient_at_center_pixel(self, pixel_cube: np.ndarray) -> np.ndarray:
        """
        Compute gradient at center pixel [1, 1, 1] of 3x3x3 array
        """
        dx = 0.5 * (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0])
        dy = 0.5 * (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1])
        ds = 0.5 * (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1])
        return np.array([dx, dy, ds])

    def _compute_hessian_at_center_pixel(self, pixel_cube: np.ndarray) -> np.ndarray:
        """
        Compute Hessian at center pixel [1, 1, 1] of 3x3x3 array
        """
        center_pixel_value = pixel_cube[1, 1, 1]
        dxx = pixel_cube[1, 1, 2] - 2 * center_pixel_value + pixel_cube[1, 1, 0]
        dyy = pixel_cube[1, 2, 1] - 2 * center_pixel_value + pixel_cube[1, 0, 1]
        dss = pixel_cube[2, 1, 1] - 2 * center_pixel_value + pixel_cube[0, 1, 1]

        dxy = 0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - 
                      pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0])
        dxs = 0.25 * (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - 
                      pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0])
        dys = 0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - 
                      pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1])

        return np.array([[dxx, dxy, dxs],
                        [dxy, dyy, dys],
                        [dxs, dys, dss]])

    def _remove_duplicate_keypoints(self, keypoints: List[dict]) -> List[dict]:
        """
        Sort keypoints and remove any duplicates
        """
        if len(keypoints) < 2:
            return keypoints

        keypoints = sorted(keypoints, key=lambda x: (x['x'], x['y'], x['sigma']))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique = unique_keypoints[-1]
            if last_unique['x'] != next_keypoint['x'] or \
               last_unique['y'] != next_keypoint['y'] or \
               last_unique['sigma'] != next_keypoint['sigma']:
                unique_keypoints.append(next_keypoint)

        return unique_keypoints