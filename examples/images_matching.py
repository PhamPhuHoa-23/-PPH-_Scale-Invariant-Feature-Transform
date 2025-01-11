import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from src.scale_space import ScaleSpace
from src.keypoints import KeypointDetector
from src.orientation import OrientationAssigner  
from src.descriptor import DescriptorExtractor

def save_descriptors_txt(filename: str, keypoints: List[dict], descriptors: np.ndarray):
    """
    Save keypoints and descriptors to readable text file
    Args:
        filename: Output file path (.txt)
        keypoints: List of keypoint dictionaries 
        descriptors: Numpy array of descriptors
    """
    with open(filename, 'w') as f:
        # Write total number of keypoints
        f.write(f"Total keypoints: {len(keypoints)}\n")
        f.write("-" * 50 + "\n")
        
        # For each keypoint and its descriptor
        for idx, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            # Write keypoint info
            f.write(f"Keypoint {idx}:\n")
            f.write(f"x: {kp['x']}, y: {kp['y']}\n")
            f.write(f"octave: {kp['octave']}\n")
            f.write(f"interval: {kp['interval']}\n")
            f.write(f"sigma: {kp['sigma']}\n")
            if 'orientation' in kp:
                f.write(f"orientation: {kp['orientation']}\n")
            
            # Write descriptor values 
            f.write("Descriptor:\n")
            desc_str = ", ".join(str(int(x)) for x in desc)  
            f.write(f"{desc_str}\n")
            
            f.write("-" * 50 + "\n")

def compute_sift(image: np.ndarray):
    """Compute SIFT keypoints and descriptors"""
    # Initialize components  
    scale_space = ScaleSpace(num_intervals=3)
    keypoint_detector = KeypointDetector(contrast_threshold=0.01)
    orientation_assigner = OrientationAssigner()
    descriptor_extractor = DescriptorExtractor()
    
    # Compute keypoints
    gaussian_pyramid = scale_space.generate_gaussian_pyramid(image)
    dog_pyramid = scale_space.generate_DoG_pyramid(gaussian_pyramid)
    keypoints = keypoint_detector.find_keypoints(gaussian_images=gaussian_pyramid, 
                                                dog_images=dog_pyramid, 
                                                num_intervals=3, 
                                                sigma=1.6)
    
    # Compute orientations & descriptors
    descriptors = []
    kps = []
    for kp in keypoints:
        octave = kp['octave']
        layer = int(round(kp['interval']))
        gaussian_image = gaussian_pyramid[octave][layer]

        oriented_kps = orientation_assigner.compute_orientation(kp, gaussian_image)
        for oriented_kp in oriented_kps:
            desc = descriptor_extractor.compute_descriptors([oriented_kp], gaussian_pyramid)[0]
            descriptors.append(desc)
            kps.append(oriented_kp)
            
    return kps, np.array(descriptors)

def match_descriptors(desc1: np.ndarray, 
                     desc2: np.ndarray,
                     threshold: float = 0.9) -> List[Tuple[int, int]]:
    """
    Match descriptors using ratio test 
    Returns list of (idx1, idx2) tuples for matching keypoints
    """
    matches = []
    
    # For each descriptor in first image  
    for i in range(len(desc1)):
        # Compute distances to all descriptors in second image
        distances = np.linalg.norm(desc2 - desc1[i], axis=1)
        
        # Get indices of two closest matches
        idx = np.argpartition(distances, 2)[:2]
        best1, best2 = distances[idx[0]], distances[idx[1]]
        
        # Apply ratio test
        if best1 < threshold * best2: matches.append((i, idx[0]))
            
    return matches

def plot_matches(img1: np.ndarray,
                kp1: List[dict], 
                img2: np.ndarray,
                kp2: List[dict], 
                matches: List[Tuple[int, int]]):
    """Plot matching keypoints between images"""
    # Create figure
    fig, ax = plt.subplots(figsize=(20,10))
    
    # Combine images horizontally
    h1, w1 = img1.shape
    h2, w2 = img2.shape 
    vis = np.zeros((max(h1, h2), w1 + w2), np.float32)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    ax.imshow(vis, cmap='gray')
    
    # Plot matches
    for idx1, idx2 in matches:
        x1, y1 = kp1[idx1]['x'], kp1[idx1]['y'] 
        x2, y2 = kp2[idx2]['x'], kp2[idx2]['y']
        
        # Draw line between matches
        ax.plot([x1, x2 + w1], [y1, y2], 'r-')
        
        # Draw keypoints  
        ax.plot(x1, y1, 'r.')
        ax.plot(x2 + w1, y2, 'r.')
        
    plt.axis('off') 
    plt.show()

def main():
    # Get input image names from user
    img1_name = input("Enter name of first image in 'images' folder: ")
    img2_name = input("Enter name of second image in 'images' folder: ")

    # Load images
    img1_path = f'./examples/images/{img1_name}'
    img2_path = f'./examples/images/{img2_name}'
    
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize images to same size
    size = (100, 100)
    img1 = cv2.resize(img1, size, interpolation=cv2.INTER_LINEAR) 
    # img2 = cv2.resize(img2, size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to float32 
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32) 
    
    # Compute SIFT
    print("Computing SIFT for image 1...")
    kp1, desc1 = compute_sift(img1) 
    print(f"Found {len(kp1)} keypoints in image 1")
    print(f"Found {len(desc1)}")
    print(desc1)
    
    print("Computing SIFT for image 2...")
    kp2, desc2 = compute_sift(img2)
    print(f"Found {len(kp2)} keypoints in image 2")
    print(f"Found {len(desc2)}")
    print(desc2)
    
    # Save descriptors for image 1
    save_descriptors_txt('descriptors_img1.txt', kp1, desc1)
    
    # Save descriptors for image 2  
    save_descriptors_txt('descriptors_img2.txt', kp2, desc2)

    # Match descriptors 
    print("Matching descriptors...")
    matches = match_descriptors(desc1, desc2)
    print(f"Found {len(matches)} matches")
    print(matches)
    
    # Plot results
    scale_space = ScaleSpace(num_intervals=3)
    plot_matches(scale_space._create_initial_image(img1), kp1, scale_space._create_initial_image(img2), kp2, matches)
    
if __name__ == "__main__":
    main()