import numpy as np
from typing import List, Tuple
from src.utils import *

class ScaleSpace:
    def __init__(self,
                 num_octaves: int = None, 
                 num_intervals: int = 3,
                 assumed_blur: float = 0.5,
                 sigma: float = 1.6):
        """
        Initialize Scale Space parameters
        Args:
            num_octaves: Number of octaves 
            num_intervals: Number of scales per octave (s)
            assumed_blur: Initial blur in the input image
            sigma: Initial sigma
        """
        self.num_intervals = num_intervals
        self.assumed_blur = assumed_blur
        self.sigma = sigma
        self.num_octaves = num_octaves
        # Number of images per octave = intervals + 3 để detect extrema
        self.num_images_per_octave = num_intervals + 3
        self.k = 2 ** (1.0 / num_intervals)

    def generate_gaussian_pyramid(self, image: np.ndarray) -> List[List[np.ndarray]]:
        """
        Generate Gaussian Pyramid
        Args:   
            image: Input image (grayscale)
        Returns:
            List of lists where inner list represents an octave containing Gaussian blurred images
        """
        if self.num_octaves is None:
            self.num_octaves = int(round(np.log(min(image.shape)) / np.log(2) - 1))

        # Generate base image
        base_image = self._create_initial_image(image)

        # Generate list of sigmas
        gaussian_kernels = self._generate_gaussian_kernels()
        
        # Build pyramid
        gaussian_images = []
        for octave_idx in range(self.num_octaves):
            gaussian_images_in_octave = [base_image]
            for gaussian_kernel in gaussian_kernels[1:]:
                image = gaussian_blur(base_image, gaussian_kernel)
                gaussian_images_in_octave.append(image)
            gaussian_images.append(gaussian_images_in_octave)
            
            # Prepare base for next octave
            if octave_idx < self.num_octaves - 1:
                base_image = gaussian_images_in_octave[-3]
                base_image = downsample_image(base_image)

        return gaussian_images

    def generate_DoG_pyramid(self, gaussian_pyramid: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        Generate Difference of Gaussian pyramid
        Args:
            gaussian_pyramid: Gaussian pyramid from generate_gaussian_pyramid
        Returns:
            List of lists where each inner list represents an octave containing DoG images 
        """
        dog_images = []

        for gaussian_images_in_octave in gaussian_pyramid:
            dog_images_in_octave = []
            for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                dog_images_in_octave.append(second_image - first_image)
            dog_images.append(dog_images_in_octave)

        return dog_images

    def _create_initial_image(self, image: np.ndarray) -> np.ndarray:
        """
        Create initial image by upsampling and blurring
        Args:
            image: Input image
        Returns:
            Processed initial image
        """
        # Double image size
        image = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

        # Calculate sigma difference
        sigma_diff = np.sqrt(max((self.sigma ** 2) - ((2 * self.assumed_blur) ** 2), 0.01))
        return gaussian_blur(image, sigma_diff)

    def _generate_gaussian_kernels(self) -> np.ndarray:
        """
        Generate list of gaussian kernels to go from one blur scale to the next
        Returns:
            1D array of gaussian kernel sigmas
        """
        gaussian_kernels = np.zeros(self.num_images_per_octave)
        gaussian_kernels[0] = self.sigma

        for image_index in range(1, self.num_images_per_octave):
            sigma_previous = self.k ** (image_index - 1) * self.sigma
            sigma_total = self.k * sigma_previous
            gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)

        return gaussian_kernels