import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pygco import cut_simple

def mrf_image_segmentation(image_path, num_labels=3, smoothness_cost=2):
    """
    Performs image segmentation using Markov Random Fields (MRF) and Graph Cuts.

    Parameters:
        image_path (str): Path to the image file.
        num_labels (int): Number of segments (labels) to partition the image into.
        smoothness_cost (int): Penalty for assigning different labels to neighboring pixels.

    Returns:
        segmented_image (numpy.ndarray): The segmented image with each region assigned a unique color.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at the specified path: {image_path}")
    
    # Convert to RGB and reshape for clustering
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image_rgb.shape
    pixels = image_rgb.reshape(-1, c)

    # K-means clustering to initialize labels
    k = num_labels
    _, labels, centers = cv2.kmeans(
        data=np.float32(pixels),
        K=k,
        bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
        attempts=10,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )
    labels = labels.flatten()

    # Compute unary potentials (data cost)
    unary_cost = np.zeros((pixels.shape[0], k))
    for i in range(k):
        cluster_center = centers[i]
        distances = np.linalg.norm(pixels - cluster_center, axis=1)
        unary_cost[:, i] = distances
    
    # Pairwise potentials (smoothness cost)
    pairwise_cost = smoothness_cost * (np.ones((k, k)) - np.eye(k))
    
    # Reshape labels and compute pairwise edges
    labels_reshaped = labels.reshape(h, w)
    edges = []
    for i in range(h):
        for j in range(w):
            if i < h - 1:  # Down
                edges.append((i * w + j, (i + 1) * w + j))
            if j < w - 1:  # Right
                edges.append((i * w + j, i * w + j + 1))

    # Perform graph cut optimization
    optimized_labels = cut_simple(unary_cost, pairwise_cost, edges, h * w, k)
    optimized_labels_reshaped = optimized_labels.reshape(h, w)
    
    # Generate segmented image
    segmented_image = np.zeros_like(image_rgb)
    for label in range(k):
        color = centers[label].astype(np.uint8)
        segmented_image[optimized_labels_reshaped == label] = color

    return segmented_image

def process_folder(input_folder, output_folder, num_labels=3, smoothness_cost=10):
    """
    Processes all images in a folder using MRF segmentation and saves the segmented images.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save segmented images.
        num_labels (int): Number of segments (labels).
        smoothness_cost (int): Penalty for smoothness.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"segmented_{image_file}")
        
        print(f"Processing: {input_path}")
        
        try:
            # Perform MRF-based segmentation
            segmented_image = mrf_image_segmentation(input_path, num_labels=num_labels, smoothness_cost=smoothness_cost)
            
            # Save the segmented image
            segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, segmented_image_bgr)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

def main(input_folder, output_folder, num_labels=3, smoothness_cost=10):
    """
    Main function to process a folder of images.

    Parameters:
        input_folder (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        num_labels (int): Number of segments (labels).
        smoothness_cost (int): Penalty for smoothness.
    """
    process_folder(input_folder, output_folder, num_labels=num_labels, smoothness_cost=smoothness_cost)

# Example usage
main(
    input_folder=r"C:\Users\Thinkpad\Desktop\input_images",  # Replace with your input folder path
    output_folder=r"C:\Users\Thinkpad\Desktop\output_images",  # Replace with your output folder path
    num_labels=3,
    smoothness_cost=10
)
