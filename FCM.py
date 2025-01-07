import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color


# Fuzzy C-Means Implementation
class FCM:
    def __init__(self, n_clusters=3, max_iter=300, m=2.0, error=1e-5):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m  # Fuzziness parameter
        self.error = error

    def fit(self, data):
        n_samples = data.shape[0]

        # Initialize membership matrix randomly
        self.u = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples).T

        # Iterate
        for iteration in range(self.max_iter):
            # Calculate cluster centers
            self.centers = self._compute_centers(data)

            # Update membership matrix
            u_new = self._update_membership(data)

            # Check for convergence
            if np.linalg.norm(self.u - u_new) < self.error:
                break

            self.u = u_new

    def _compute_centers(self, data):
        # Compute cluster centers
        um = self.u ** self.m
        return (um @ data) / np.sum(um, axis=1, keepdims=True)

    def _update_membership(self, data):
        # Update membership matrix
        distances = np.linalg.norm(data[:, np.newaxis] - self.centers, axis=2) + 1e-10
        inv_distances = 1.0 / distances
        power = 2 / (self.m - 1)
        normalized = (inv_distances.T / np.sum(inv_distances, axis=1)).T
        return normalized ** power


# Load and preprocess the image
def load_image(image_path):
    image = io.imread(image_path)
    if len(image.shape) > 2:  # Convert to grayscale if RGB
        image = color.rgb2gray(image)
    return image


# Reshape the image for clustering
def reshape_image(image):
    return image.flatten().reshape(-1, 1)


# Segment the image
def segment_image(image_shape, membership_matrix):
    cluster_indices = np.argmax(membership_matrix, axis=0)
    segmented_image = cluster_indices.reshape(image_shape)
    return segmented_image


# Visualize and save the results
def save_segmented_image(segmented_image, output_path):
    plt.imsave(output_path, segmented_image, cmap="jet")


# Process a folder of images
def process_folder(input_folder, output_folder, n_clusters=3):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"segmented_{image_file}")

        print(f"Processing: {input_path}")

        # Load image
        original_image = load_image(input_path)

        # Reshape for clustering
        reshaped_image = reshape_image(original_image)

        # Perform Fuzzy C-Means
        fcm = FCM(n_clusters=n_clusters)
        fcm.fit(reshaped_image)

        # Segment the image
        segmented_image = segment_image(original_image.shape, fcm.u)

        # Save the segmented image
        save_segmented_image(segmented_image, output_path)
        print(f"Saved: {output_path}")


# Main function
def main(input_folder, output_folder, n_clusters=3):
    process_folder(input_folder, output_folder, n_clusters)


# Run the main function
main(
    input_folder=r"C:\Users\Thinkpad\Desktop\input_images",  # Replace with your input folder path
    output_folder=r"C:\Users\Thinkpad\Desktop\output_images",  # Replace with your output folder path
    n_clusters=3
)
