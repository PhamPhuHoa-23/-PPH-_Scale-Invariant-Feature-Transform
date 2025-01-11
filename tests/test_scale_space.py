import pytest
import numpy as np
from src.scale_space import ScaleSpace

def test_gaussian_pyramid_shape():
    """Test if Gaussian pyramid has correct shapes"""
    # Create dummy image
    image = np.random.rand(100, 100)
    scale_space = ScaleSpace(num_octaves=3, scales_per_octave=3)
    gaussian_pyramid = scale_space.generate_gaussian_pyramid(image)
    
    # Test pyramid structure
    assert len(gaussian_pyramid) == 3  # num_octaves
    assert len(gaussian_pyramid[0]) == 3  # scales_per_octave
    assert gaussian_pyramid[2][0].shape == (50, 50)  # Check downsampling

def test_DoG_pyramid_shape():
    """Test if DoG pyramid has correct shapes"""
    image = np.random.rand(100, 100)
    scale_space = ScaleSpace()
    gaussian_pyramid = scale_space.generate_gaussian_pyramid(image)
    DoG_pyramid = scale_space.generate_DoG_pyramid(gaussian_pyramid)
    
    # Test DoG structure
    assert len(DoG_pyramid) == len(gaussian_pyramid)
    assert len(DoG_pyramid[0]) == len(gaussian_pyramid[0]) - 1

def test_initial_image():
    """Test initial image processing"""
    image = np.ones((50, 50))
    scale_space = ScaleSpace()
    initial_image = scale_space._create_initial_image(image)
    
    assert initial_image.shape == (100, 100)  # Test doubling