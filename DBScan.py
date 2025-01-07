import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def dbscan_image_segmentation(image_path, eps=0.5, min_samples=5):
    """
    Performs DBSCAN clustering on an image for segmentation.

    Parameters:
        image_path (str): Path to the image.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        segmented_image (numpy.ndarray): Image segmented using DBSCAN.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    # Convert the image to RGB format (from BGR, which is the default in OpenCV)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image into a 2D array of pixels and normalize pixel values
    pixels = image_rgb.reshape(-1, 3)
    
    # Standardize pixel values
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(pixels_scaled)
    
    # Reshape labels to match image dimensions
    labels_reshaped = labels.reshape(image_rgb.shape[:2])
    
    # Generate segmented image by mapping each cluster to a color
    unique_labels = np.unique(labels)
    segmented_image = np.zeros_like(image_rgb)
    for label in unique_labels:
        if label == -1:
            # Assign black color for noise
            color = [0, 0, 0]
        else:
            # Generate a random color for each cluster
            color = np.random.randint(0, 255, size=3)
        segmented_image[labels_reshaped == label] = color
    
    # Return the segmented image
    return segmented_image

def process_folder(input_folder, output_folder, eps=0.5, min_samples=5):
    """
    Processes all images in a folder using DBSCAN clustering and saves the segmented images.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save segmented images.
        eps (float): DBSCAN epsilon parameter.
        min_samples (int): DBSCAN minimum samples parameter.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"segmented_{image_file}")
        
        print(f"Processing: {input_path}")
        
        # Perform DBSCAN segmentation
        segmented_image = dbscan_image_segmentation(input_path, eps=eps, min_samples=min_samples)
        
        # Save the segmented image
        segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, segmented_image_bgr)
        print(f"Saved: {output_path}")

def main(input_folder, output_folder, eps=0.5, min_samples=5):
    """
    Main function to process a folder of images.

    Parameters:
        input_folder (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        eps (float): DBSCAN epsilon parameter.
        min_samples (int): DBSCAN minimum samples parameter.
    """
    process_folder(input_folder, output_folder, eps=eps, min_samples=min_samples)

# Example usage
main(
    input_folder=r"C:\Users\Thinkpad\Desktop\input_images",  # Replace with your input folder path
    output_folder=r"C:\Users\Thinkpad\Desktop\output_images",  # Replace with your output folder path
    eps=0.5,
    min_samples=10
)
