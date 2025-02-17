import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re

### Apply Gaussian blur, followed by Laplacian operator
def apply_gaussian_laplace(image, kernel_size=5, sigma=1.0):

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F) 

    return np.abs(laplacian)

### Extract the distance in cm from the filename
def extract_distance_from_filename(filename):

    match = re.search(r'd(\d+)cm', filename)

    if match:
        return int(match.group(1))
    
    return None

### Process all images in the folder and calculate the 90th percentile of filtered images
def process_images_in_folder(folder_path, kernel_size=5, sigma=1.0, percentile=90):

    distances = []
    percentiles = []

    for filename in sorted(os.listdir(folder_path)):

        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Filter for image files
            distance = extract_distance_from_filename(filename)
            
            if distance is not None:

                filepath = os.path.join(folder_path, filename)
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    continue
                
                filtered_image = apply_gaussian_laplace(image, kernel_size, sigma)
                percentile_value = np.percentile(filtered_image, percentile)

                distances.append(distance)
                percentiles.append(percentile_value)

    # Sort distances and percentiles in aascending order of distances
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    percentiles = np.array(percentiles)[sorted_indices]

    return distances, percentiles

### Plot the 90th percentiles against d_v - viewing distances
def plot_percentiles(distances, percentiles):

    plt.figure(figsize=(10, 6))
    plt.plot(distances, percentiles, marker='o', linestyle='-', color='blue')

    plt.xlabel('Distance (cm)')
    plt.ylabel('90th Percentile of Gaussian laplace Filtered Image')
    plt.title('Sharpness vs. Viewing Distance')

    # plt.xlim(0, 200)  # Set x-axis range from 0 to 200
    plt.xlim(15, 180)  # Set x-axis range from 0 to 200

    plt.grid(which='major', linestyle='-', linewidth=0.75, color='gray')
    plt.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
    plt.minorticks_on()  # Enable minor ticks for finer gridlines
    plt.show()

# Main function
folder_path = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Camera_Calibration\FOV_DoF\all_images"
distances, percentiles = process_images_in_folder(folder_path)

if distances.size > 0 and percentiles.size > 0:
    plot_percentiles(distances, percentiles)
else:
    print("No valid images found in the folder.")
