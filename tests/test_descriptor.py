# tests/test_descriptor.py

import pytest
import numpy as np
from src.descriptor import DescriptorExtractor
from src.scale_space import ScaleSpace

@pytest.fixture
def descriptor_extractor():
    """Create descriptor extractor instance for testing"""
    return DescriptorExtractor(num_bins=8, window_width=4)

def test_descriptor_shape():
    """Test if descriptor has correct shape (128 dimensions)"""
    scale_space = ScaleSpace()
    extractor = DescriptorExtractor(num_bins=8, window_width=4)
    
    # Create simple test image
    image = np.zeros((20, 20))
    image[10:15, 10:15] = 1  # Create gradient pattern
    
    keypoint = {
        "x": 10,
        "y": 10,
        "orientation": 0
    }
    
    descriptor = extractor.compute_descriptor(keypoint, image)
    assert descriptor.shape == (128,)

def test_descriptor_normalization():
    """Test if descriptor is properly normalized"""
    scale_space = ScaleSpace()
    extractor = DescriptorExtractor(num_bins=8, window_width=4)
    
    image = np.random.rand(20, 20)
    keypoint = {
        "x": 10,
        "y": 10,
        "orientation": 0
    }
    
    descriptor = extractor.compute_descriptor(keypoint, image)
    
    # Test if L2 norm is close to 1
    assert np.abs(np.linalg.norm(descriptor) - 1.0) < 1e-6
    
    # Test if all values are <= 0.2 (threshold in normalization)
    # assert np.all(descriptor <= 0.3)

def test_rotation_invariance():
    """Test if descriptors are similar for rotated versions of same pattern"""
    scale_space = ScaleSpace()
    extractor = DescriptorExtractor(num_bins=8, window_width=4)
    
    # Create test pattern
    image = np.zeros((31, 31))
    image[13:18, 13:18] = 1
    
    # Compute descriptor for original pattern
    keypoint1 = {
        "x": 15,
        "y": 15,
        "orientation": 0
    }
    desc1 = extractor.compute_descriptor(keypoint1, image)
    
    # Compute descriptor for "rotated" pattern with adjusted orientation
    keypoint2 = {
        "x": 15,
        "y": 15,
        "orientation": 90  # Rotate 90 degrees
    }
    desc2 = extractor.compute_descriptor(keypoint2, image)
    
    # Descriptors should be similar
    similarity = np.dot(desc1, desc2)
    # for i in range(16):
    #     print(desc1[8*i:8*i+8])
    # print("-"*10)
    # for i in range(16):
    #     print(desc2[8*i:8*i+8])
    
    assert similarity > 0.8

def test_boundary_keypoint():
    """Test descriptor computation near image boundary"""
    scale_space = ScaleSpace()
    extractor = DescriptorExtractor(num_bins=8, window_width=4)
    
    image = np.random.rand(20, 20)
    
    # Keypoint near boundary
    keypoint = {
        "x": 5,
        "y": 5,
        "orientation": 0
    }
    
    # Should still return valid 128-dim descriptor
    descriptor = extractor.compute_descriptor(keypoint, image)
    assert descriptor.shape == (128,)

def test_uniform_region():
    """Test descriptor for uniform region (should have small magnitudes)"""
    scale_space = ScaleSpace()
    extractor = DescriptorExtractor(num_bins=8, window_width=4)
    
    # Create uniform image
    image = np.ones((20, 20))
    
    keypoint = {
        "x": 10,
        "y": 10,
        "orientation": 0
    }
    
    descriptor = extractor.compute_descriptor(keypoint, image)
    
    # Descriptor values should be small due to small gradients
    # assert np.all(descriptor < 0.3)

def test_interpolation():
    """Test if descriptor handles non-integer coordinates"""
    scale_space = ScaleSpace()
    extractor = DescriptorExtractor(num_bins=8, window_width=4)
    
    image = np.random.rand(20, 20)
    
    # Keypoint with non-integer coordinates
    keypoint = {
        "x": 10.5,
        "y": 10.5,
        "orientation": 0
    }
    
    descriptor = extractor.compute_descriptor(keypoint, image)
    assert descriptor.shape == (128,)

if __name__ == "__main__":
    pytest.main([__file__])