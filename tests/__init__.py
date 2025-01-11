import numpy as np
import cv2
from typing import Tuple, Optional

def generate_gaussian_kernel(
        sigma: float,
        kernel_size: Optional[int] = None) -> np.ndarray:
    """
    Generate 2D Gaussian kernel
    Args:
        sigma: Standard deviation of Gaussian
        kernel_size: Size of kernel(Optional, calculated from sigma if None)
    Returns:
        2D numpy array of Gaussian kernel
    """
    if kernel_size is None:
        # Apply 3-sigma method (99.7% of information)
        kernel_size = 2 * int(np.ceil(3*sigma)) + 1

    if kernel_size % 2 == 0:
        # Kernal size should be odd
        kernel_size += 1
    
    # Create 1D coordinate arrays
    ax = np.arange(-(kernel_size//2),kernel_size//2+1)
    # Create 2D coordinate arrays using meshgid
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2)) / (2*sigma**2)

    # Normalize
    kernel = kernel / (2*np.pi*sigma**2)

    # Normalize to sum to 1
    kernel = kernel / kernel.sum()
    return kernel

def compute_gradient(
        image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient magnitude and orientation using central diffirence
    Args:
        image: Input image
    Returns:
        Tuple of (magnitude, orientation) arrays
    """
    # Central diffirence for x direction
    dx = np.zeros_like(image)
    dx[:,1:-1] = image[:,2:] - image[:,:-2] / 2

    dx[:,0] = image[:,1] - image[:,0]
    dx[:,-1] = image[:,-1] - image[:,-2]

    # Central diffirence for y direction
    dy = np.zeros_like(image)
    dy[1:-1,:] = (image[2:,:] - image[:-2,:]) / 2
    dy[0,:] = image[1,:] - image[0,:]
    dy[-1,:] = image[-1,:] - image[-2,:]

    # Compute magnitude and orientation
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy,dx)

    return magnitude, orientation

def gaussian_blur(
        image: np.ndarray,
        sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to image
    Args:
        image: Input image
        sigma: Blur sigma
    Returns:
        Blurred image
    """
    kernel = generate_gaussian_kernel(sigma=sigma)
    kernel_size = kernel.shape[0]
    padding_size = kernel_size // 2

    # Pad the image to maintain its shape (reflect)
    padding_image = np.pad(image, pad_width=padding_size, mode="reflect")
    # Apply convolution
    out_h, out_w = image.shape
    output = np.zeros_like(image)

    # Convolution
    for i in range(kernel_size):
        for j in range(kernel_size):
            output += padding_image[i:i+out_h, j:j+out_w] * kernel[i,j]

    return output

def downsample_image(
        image: np.ndarray) -> np.ndarray:
    """
    Downsample image by factor of 2
    Args:
        image: Input image
    Returns:
        Downsampled image
    """
    return image[::2, ::2]

def rotate_image(
        image: np.ndarray,
        angle: float,
        center: Tuple[int, int]) -> np.ndarray:
    """
    Rotate image around center point
    Args:
        image: Input image
        angle: Rotation angel in degress
        center: Center point (x, y)
    Returns:
        Rotated image
    """
    # Get image height, width
    h, w = image.shape

    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)

    # Perform rotation
    rotated_image= cv2.warpAffine(image, M, (w,h))

    return rotated_image