# tests/test_keypoints.py

import pytest
import numpy as np
from src.keypoints import KeypointDetector

def test_find_scale_space_extrema():
    """Test extrema detection in scale space"""
    # Create mock DoG octave with known extrema
    DoG_octave = np.zeros((3, 5, 5))  # 3 images, 5x5 each
    
    # Create a local maximum
    DoG_octave[0:3, 1:4, 1:4] = 0.5  # surrounding points
    DoG_octave[1, 2, 2] = 1.0  # center point
    
    detector = KeypointDetector()
    extrema = detector._find_scale_space_extrema(DoG_octave)
    
    # Should find one extrema point at (2,2,1)
    assert len(extrema) == 1
    assert extrema[0] == (2, 2, 1)

def test_empty_extrema():
    """Test with no extrema points"""
    DoG_octave = np.zeros((3, 5, 5))  # All zeros = no extrema
    detector = KeypointDetector()
    extrema = detector._find_scale_space_extrema(DoG_octave)
    assert len(extrema) == 0

def test_refine_keypoint_location():
    """Test keypoint refinement"""
    # Create mock DoG octave
    DoG_octave = np.random.rand(3, 10, 10)
    detector = KeypointDetector()
    
    # Test valid keypoint
    keypoint = (5, 5, 1)  # Center point
    refined = detector._refine_keypoints_location(keypoint, DoG_octave)
    
    # Should return a dictionary with refined coordinates
    assert isinstance(refined, (dict, type(None)))
    if refined is not None:
        assert "x" in refined
        assert "y" in refined
        assert "scale_idx" in refined
        assert "contrast" in refined

def test_edge_cases():
    """Test boundary conditions and invalid inputs"""
    DoG_octave = np.random.rand(3, 10, 10)
    detector = KeypointDetector()
    
    # Test boundary keypoints
    edge_cases = [
        (0, 5, 1),  # Left edge
        (9, 5, 1),  # Right edge
        (5, 0, 1),  # Top edge
        (5, 9, 1),  # Bottom edge
        (5, 5, 0),  # First scale
        (5, 5, 2)   # Last scale
    ]
    
    for keypoint in edge_cases:
        refined = detector._refine_keypoints_location(keypoint, DoG_octave)
        assert refined is None  # Should reject boundary keypoints

def test_full_pipeline():
    """Test complete keypoint detection pipeline"""
    # Create mock DoG pyramid
    DoG_pyramid = [
        [np.random.rand(10, 10) for _ in range(3)]  # One octave with 3 scales
    ]
    
    detector = KeypointDetector()
    keypoints = detector.find_keypoints(DoG_pyramid)
    
    # Check keypoint format
    for keypoint in keypoints:
        if keypoint is not None:
            assert isinstance(keypoint, dict)
            assert all(k in keypoint for k in ["x", "y", "scale_idx", "contrast"])

def test_contrast_threshold():
    """Test contrast threshold filtering"""
    detector = KeypointDetector(contrast_threshold=0.5)
    DoG_octave = np.ones((3, 10, 10)) * 0.1  # All low contrast
    
    keypoint = (5, 5, 1)
    refined = detector._refine_keypoints_location(keypoint, DoG_octave)
    
    # Should reject low contrast keypoint
    assert refined is None

if __name__ == "__main__":
    pytest.main([__file__])