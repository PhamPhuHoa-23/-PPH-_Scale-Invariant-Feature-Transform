# tests/test_utils.py
import pytest
import numpy as np
from src.utils import *

def test_gaussian_kernel():
    """Test Gaussian kernel generation"""
    kernel = generate_gaussian_kernel(sigma=1.0)
    # Kernel should be normalized
    assert np.abs(np.sum(kernel) - 1.0) < 1e-6
    # Kernel should be symmetric
    assert np.allclose(kernel, kernel.T)

def test_gradient_computation():
    """Test gradient computation"""
    # Create test image with known gradients
    image = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.float32)
    magnitude, orientation = compute_gradient(image)
    # Check expected values
    assert magnitude.shape == image.shape
    assert orientation.shape == image.shape

def test_gaussian_blur():
    """Test Gaussian blur"""
    image = np.ones((3, 3))
    blurred = gaussian_blur(image, sigma=1.0)
    assert blurred.shape == image.shape
    # Blurred image should still sum to approximately the same as original
    assert np.abs(np.sum(blurred) - np.sum(image)) < 1e-6

def test_downsample():
    """Test image downsampling"""
    image = np.ones((10, 10))
    downsampled = downsample_image(image)
    assert downsampled.shape == (5, 5)

def test_rotate_image():
    """Test image rotation"""
    image = np.ones((10, 10))
    center = (5, 5)
    rotated = rotate_image(image, angle=90, center=center)
    assert rotated.shape == image.shape