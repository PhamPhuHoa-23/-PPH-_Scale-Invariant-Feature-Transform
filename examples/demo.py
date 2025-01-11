import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List
from src.scale_space import ScaleSpace
from src.keypoints import KeypointDetector
from src.orientation import OrientationAssigner
from src.descriptor import DescriptorExtractor

# def plot_keypoints(image: np.ndarray,
#                    keypoints: List[dict],
#                    title: str = "Keypoints"):
#     """Helper function to plot keypoints"""
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image, cmap='gray')
#     for kp in keypoints:
#         plt.plot(kp['x'], kp['y'], 'r.')
#         if 'orientation' in kp:
#             # Plot orientation (optional)
#             angle = np.deg2rad(kp['orientation'])
#             length = 20
#             dx = length * np.cos(angle)
#             dy = length * np.sin(angle)
#             plt.arrow(kp['x'], kp['y'], dx, dy, 
#                      head_width=3, head_length=5, fc='r', ec='r')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()
def plot_keypoints(image: np.ndarray, keypoints: List[dict], title: str = "Keypoints"):
    """
    Plot keypoints with colored circles and orientation
    Args:
        image: Input image
        keypoints: List of keypoint dictionaries with x, y, scale_idx, orientation
        title: Plot title
    """
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    # Show image
    plt.imshow(image, cmap='gray')
    
    initial_sigma = 1.6
    k = 2 ** (1.0/5)  # với scales_per_octave = 3
   
    # Plot each keypoint
    for kp in keypoints:
        x, y = kp['x'], kp['y']
        scale_idx = kp.get('scale_idx', 0)
        octave_idx = kp.get('octave_idx', 0)
        
        # Tính sigma cho scale này
        sigma = initial_sigma * (k ** scale_idx) * (2 ** octave_idx)
        
        # Radius tương đương với OpenCV
        radius = sigma * 2
        
        # Tạo circle patch
        circle = patches.Circle(
            (x, y), 
            radius=radius,
            fill=True,
            color='red',
            alpha=0.5
        )
        ax.add_patch(circle)
        
        # Vẽ orientation nếu có
        if 'orientation' in kp:
            orientation = np.deg2rad(kp['orientation'])
            dx = np.cos(orientation) * radius
            dy = np.sin(orientation) * radius
            plt.arrow(x, y, dx, dy, 
                     head_width=radius/2, 
                     head_length=radius/2, 
                     fc='yellow', 
                     ec='yellow',
                     alpha=0.8)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_gaussian_pyramid(gaussian_pyramid: List[List[np.ndarray]]):
    """Helper function to visualize Gaussian pyramid""" 
    plt.figure(figsize=(15, 5))
    for octave_idx, octave in enumerate(gaussian_pyramid):
        for scale_idx, image in enumerate(octave):
            plt.subplot(len(gaussian_pyramid), len(octave), 
                       octave_idx * len(octave) + scale_idx + 1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.title(f'Octave {octave_idx}, Scale {scale_idx}')
    plt.tight_layout()
    plt.show()

def main():
    # 1. Load image
    image_path = "examples/images/cat.jpg"
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = gray.astype(np.float32) / 255.0
    gray = gray.astype(np.float32)

    # 2. Initialize SIFT components
    scale_space = ScaleSpace(num_octaves=int(round(np.log(min(gray.shape*2)) / np.log(2) - 2)), scales_per_octave=5, sigma=1.6)
    keypoint_detector = KeypointDetector(contrast_threshold=0.04, edge_threshold=10, max_offset=0.5)
    orientation_assigner = OrientationAssigner(scale_space=scale_space, num_bins=36)
    descriptor_extractor = DescriptorExtractor(num_bins=8, window_width=4)

    # 3. Build Gaussian and DoG pyramid
    print("----- Building Gaussian Pyramid -----")
    gaussian_pyramid = scale_space.generate_gaussian_pyramid(image=gray)
    visualize_gaussian_pyramid(gaussian_pyramid=gaussian_pyramid)

    print("----- Building DoG pyramid -----")
    dog_pyramid = scale_space.generate_DoG_pyramid(gaussian_pyramid=gaussian_pyramid)
    visualize_gaussian_pyramid(gaussian_pyramid=dog_pyramid)
    print(dog_pyramid)
    # 4. Detect keypoints
    print("----- Detecting keypoints -----")
    keypoints = keypoint_detector.find_keypoints(DoG_pyramid=dog_pyramid)
    print(f"Found {len(keypoints)} initial keypoints")
    plot_keypoints(image=scale_space._create_initial_image(gray), keypoints=keypoints, title="Initial keypoints")

    # 5. Assign orientations
    print("----- Assigning orientations -----")
    oriented_keypoints = []

    for octave_idx, octave in enumerate(gaussian_pyramid):
        for kp in keypoints:
            if kp["scale_idx"] < len(octave):
                oriented_kps = orientation_assigner.compute_orientation(keypoint=kp, gaussian_image=octave[int(kp["scale_idx"])])
            oriented_keypoints.extend(oriented_kps)
    
    print(f"Found {len(oriented_keypoints)} oriented keypoints")
    plot_keypoints(image=scale_space._create_initial_image(gray), keypoints=oriented_keypoints, title="Oriented keypoints")

    # 6. Extract descriptors
    descriptors = []
    for kp in oriented_keypoints:
        octave_idx = kp.get("octave_idx", 0)
        desc = descriptor_extractor.compute_descriptor(keypoint=kp, gaussian_image=gaussian_pyramid[0][int(kp["scale_idx"])])
        descriptors.append(desc)
    descriptors = np.array(descriptors)

    print(f"Computed {len(descriptors)} descriptors")

    # 7. Compare with OpenCV SIFT (optional)
    sift = cv2.SIFT_create()
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = gray.astype(np.float32) / 255.0
    kp_cv, desc_cv = sift.detectAndCompute(gray, None)
    print(f"OpenCV SIFT found {len(kp_cv)} keypoints")
    
    # Plot OpenCV results
    cv_image = gray.copy()
    cv_image = cv2.drawKeypoints(cv_image, kp_cv, None, 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv_image, cmap='gray')
    plt.title("OpenCV SIFT Keypoints")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()