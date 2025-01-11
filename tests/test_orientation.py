# tests/test_orientation.py

import pytest
import numpy as np
from src.orientation import OrientationAssigner
from src.scale_space import ScaleSpace

@pytest.fixture
def orientation_assigner():
    """Create orientation assigner instance for testing"""
    scale_space = ScaleSpace()  # Use default params
    return OrientationAssigner(scale_space=scale_space)

def test_create_orientation_histogram():
    """Test histogram creation from a simple gradient pattern"""
    scale_space = ScaleSpace()
    assigner = OrientationAssigner(scale_space)
    
    # Create a simple 7x7 test image with known gradient
    image = np.zeros((7, 7))
    image[3, 4] = 1  # Create horizontal gradient at center
    image[3, 2] = -1
    
    hist = assigner._create_orientation_histogram(
        image=image,
        x=3,  # center
        y=3,
        radius=3
    )
    
    # Should have peak at 0/180 degrees (horizontal gradient)
    peak_bin = np.argmax(hist)
    peak_angle = peak_bin * (360.0/assigner.num_bins)
    assert peak_angle == 0 or abs(peak_angle - 180) < 1e-6

def test_compute_orientation_single_peak():
    """Test orientation computation with single clear peak"""
    scale_space = ScaleSpace()
    assigner = OrientationAssigner(scale_space)
    
    # Create test image with single strong gradient direction
    image = np.zeros((10, 10))
    image[5, 6:8] = 1  # horizontal gradient
    image[5, 3:5] = -1
    
    keypoint = {
        "x": 5,
        "y": 5,
        "scale_idx": 0
    }
    
    keypoints = assigner.compute_orientation(keypoint, image)
    
    # Should return single keypoint with orientation near 0 or 180
    assert len(keypoints) == 1
    orientation = keypoints[0]["orientation"]
    assert orientation == 0 or abs(orientation - 180) < 1e-6

def test_compute_orientation_multiple_peaks():
    """Test orientation computation with multiple significant peaks"""
    scale_space = ScaleSpace()
    assigner = OrientationAssigner(scale_space)
    
    # Create test image with two strong gradient directions
    image = np.zeros((10, 10))
    # Horizontal gradient
    image[5, 6:8] = 1
    image[5, 3:5] = -1
    # Vertical gradient
    image[6:8, 5] = 1
    image[3:5, 5] = -1
    
    keypoint = {
        "x": 5,
        "y": 5,
        "scale_idx": 0
    }
    
    keypoints = assigner.compute_orientation(keypoint, image)
    
    # Should return multiple keypoints due to multiple strong peaks
    assert len(keypoints) > 1

def test_orientation_histogram_empty():
    """Test histogram with uniform image - should have peaks at boundaries"""
    scale_space = ScaleSpace()
    assigner = OrientationAssigner(scale_space)
    
    # Uniform image
    image = np.ones((10, 10))
    
    hist = assigner._create_orientation_histogram(
        image=image,
        x=5,
        y=5,
        radius=3
    )
    
    # Should have peak at 0/180 degrees due to boundary gradients
    peak_bin = np.argmax(hist)
    peak_angle = peak_bin * (360.0/assigner.num_bins)
    assert peak_angle == 0 or abs(peak_angle - 180) < 1e-6

def test_orientation_histogram_bounds():
    """Test if histogram bins are properly bounded [0,360)"""
    scale_space = ScaleSpace()
    assigner = OrientationAssigner(scale_space)
    
    image = np.random.rand(15, 15)
    
    hist = assigner._create_orientation_histogram(
        image=image,
        x=7,
        y=7,
        radius=3
    )
    
    # Test if we have correct number of bins
    assert len(hist) == assigner.num_bins
    
    # Each bin should represent 360/num_bins degrees
    bin_size = 360.0/assigner.num_bins
    assert abs(bin_size * len(hist) - 360) < 1e-6

if __name__ == "__main__":
    pytest.main([__file__])