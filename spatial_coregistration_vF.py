### White box and data analysis - spatial co-registration routine (Bongiorno et al.)

# whitebox
import pygame
import sys
import os
import seabreeze.spectrometers as sb
import PyCapture2
import time

# data_analysis
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import glob
import numpy as np
import re
import pygame
import seabreeze.spectrometers as sb
import PyCapture2
import time
import sys

# Import functions
from spectrum_handler import get_spectrum
from image_handler import print_build_info, print_camera_info, enable_embedded_timestamp, grab_images
from file_handler import get_timestamp, generate_filename, save_image, save_spectrum, \
                         set_working_directory_to_script_location, create_new_session_folder, format_path
from main_v3 import capture_spectrum_and_image, set_exposure_time

#################### Parameters ######################

# box_size = 60 
# box_size = 30 
# box_size = 100
# box_size = 750 
# box_size = 450
# box_size = 750 
# box_size = 100       # pixels
# box_size = 50
# box_size = 60
# box_size = 150
# box_size = 500
# box_size = 1000
# box_size = 40
box_size = 120
# box_size = 160

# d_v = 15
# d_v = 30
d_v = 60
# d_v = 90
# d_v = 120
# d_v = 150

distance = f"{d_v}cm"

zoom = "100"
# zoom = "125"

other_notes = ""
# other_notes = "_redundant"

metadata = f"boxsize{box_size}_d{distance}_zoom{zoom}{other_notes}"

# Beep noise
# beep_filepath_raw = r"C:\Users\lilan\Documents\Thesis\project_Lenovo\beep-01a.wav"
beep_filepath_raw = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\HardwareTest\beep-01a.wav"
beep_filepath = format_path(beep_filepath_raw)

time.sleep(5)

######################################################

# Set the working directory to the program file's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

### Set up data storage directory
parent_dir = "../spatial_coregistration"
data_dir = f"{parent_dir}/data"
os.makedirs(parent_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
#distance = input("Enter the distance between the device and the screen, in mm: ")
current_session_dir = create_new_session_folder(data_dir, metadata) #, distance)

### Initialise spectrometer ###
# Find spectrometer
devices = sb.list_devices()
if not devices:
    raise RuntimeError("No spectrometers found. Exiting...")
    exit() # Might not be necessary if error is raised

# Select spectrometer on 0th index
spec = sb.Spectrometer(devices[0])
print(f"Spectrometer is: {spec}.\n")

# Set integration time to 1 second
integration_time = 1000 # milliseconds
spec.integration_time_micros(integration_time * 1000) # microseconds

### Initialise camera ###
# Print PyCapture 2 Library Information
print_build_info()

# Ensure sufficient cameras are found
bus = PyCapture2.BusManager()
num_cams = bus.getNumOfCameras()
print('Number of cameras detected: ', num_cams)
if not num_cams:
    raise RuntimeError("Insufficient number of cameras. Exiting...")
    # exit()

# Select camera on 0th index
c = PyCapture2.Camera()
uid = bus.getCameraFromIndex(0)
c.connect(uid)
print_camera_info(c)

# Set exposure time to 20 ms
# set_exposure_time(c, 10000) # old setup

set_exposure_time(c, 10000) # new setup, 150cm
# set_exposure_time(c, 20000) # new setup, 150cm
# set_exposure_time(c, 30000) # new setup, 150cm
# set_exposure_time(c, 50000) # new setup


# Enable camera embedded timestamp
enable_embedded_timestamp(c, True)
c.startCapture()


input("Press ENTER to start acquisition. It will start in 20 seconds.")
time.sleep(5)


### Initialise pygame
pygame.init()
pygame.mixer.init()

# Get the screen's width and height for fullscreen mode
screen_info = pygame.display.Info()
screen_width, screen_height = screen_info.current_w, screen_info.current_h

# Set up the screen in fullscreen mode
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption('Fullscreen Box Fill with Edge Alignment')

# Define box properties
# box_size = 50       # Box is 30x30 pixels
step = box_size     # Move by 60 pixels at a time
x, y = 0, 0    # Start at top-left corner

# Define the clock for controlling the frame rate
clock = pygame.time.Clock()

# Fill the screen with black initially
screen.fill((0, 0, 0))
pygame.display.flip()

 # Delay to ensure camera doesn't capture mid-motion
# time.sleep(25)
pygame.time.delay(25*1000)

# Run spectrum and image capture once
current_pair = 0
n_spectra = 10 # Number of spectra to capture for averaging
capture_spectrum_and_image(spec, c, integration_time, current_pair, current_session_dir, n_spectra, averaging=True)
first_box = True

# Generate beep
pygame.mixer.Sound(beep_filepath).play()
# time.sleep(30)
pygame.time.delay(30*1000)
pygame.mixer.Sound(beep_filepath).play()


def move_box(x, y):
    ### Move the box across the screen, filling each row, and align to edges
    # Move the box to the right
    x += step

    # Check if the box is past the right edge
    if x + box_size > screen_width:
        x = 0
        y += step

    # Check if the box is past the bottom edge
    if y + box_size > screen_height:
        y = 0

    return x, y

# Main game loop
while True:
    # Event handling (for quitting the game and exiting fullscreen with ESC)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # Press ESC to exit fullscreen
                pygame.quit()
                sys.exit()

    if not first_box:
        # Move the box across the screen in a filling pattern
        x, y = move_box(x, y)
    else:
        first_box = False

    # Fill the screen with black
    screen.fill((0, 0, 0))

    # Draw the white box at the new position
    pygame.draw.rect(screen, (255, 255, 255), (x, y, box_size, box_size))

    # Update the display
    pygame.display.flip()

    # Delay to ensure camera doesn't capture during motion
    # time.sleep(0.5)
    pygame.time.delay(500)


    # Capture image and spectrum
    current_pair += 1
    n_spectra = 1
    capture_spectrum_and_image(spec, c, integration_time, current_pair, current_session_dir, n_spectra)
    # time.sleep(0.01)
    pygame.time.delay(10)


    # Check if we've reached the bottom-right position
    # if x == screen_width - box_size and y == screen_height - box_size:
    if x + 2*box_size > screen_width and y + 2*box_size > screen_height:
        print("Reached the bottom-right corner. Exiting.")
        pygame.time.wait(1000)  # Pause briefly before quitting
        pygame.quit()
        # sys.exit()
        break

    # Control the frame rate
    clock.tick(5)  # Adjust frame rate as needed

c.stopCapture()
c.disconnect()
spec.close()

# Save folder path of Stage 1: White box
whitebox_folder_path = current_session_dir

################## DATA ANALYSIS #####################

threshold_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for k in range(len(threshold_range)):

    #################### Parameters ############################################################################

    threshold = threshold_range[k]
    print("\nIndex:", k, "\nThreshold:", threshold,"\n")

    other_notes = ""
    # other_notes = "_redundant"

    metadata = f"boxsize{box_size}_d{distance}_threshold{threshold}_zoom{zoom}{other_notes}"

    folder_path_raw = whitebox_folder_path

    # Set the working directory to the program file's location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ### Set up data storage directory
    # Normalised sensitivity
    S_xy_dir = f"{parent_dir}/S_xy"
    os.makedirs(S_xy_dir, exist_ok=True)
    S_xy_session_dir = create_new_session_folder(S_xy_dir, metadata) #, distance)
    print(S_xy_session_dir)

    # Normalised sensitivity corrected for angle variance
    A_xy_dir = f"{parent_dir}/A_xy"
    os.makedirs(A_xy_dir, exist_ok=True)
    A_xy_session_dir = create_new_session_folder(A_xy_dir, metadata) #, distance)
    print(A_xy_session_dir)

    # Get angle variance data
    anglevar_dir = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\project_Lenovo_v2\angle_var\best_data"
    anglevar_df_path = f"{anglevar_dir}/anglevar_{d_v}cm_df_clean.csv"

    # Load the angle variance data
    anglevar_df = pd.read_csv(anglevar_df_path)

    # Extract columns from the dataframe
    angles = anglevar_df["Angle"].values
    normalised_intensities = anglevar_df["Normalised Intensity"].values

    # Polynomial fitting to 5th order
    polynomial_coefficients = np.polyfit(angles, normalised_intensities, 5)
    polynomial = np.poly1d(polynomial_coefficients)

    pixel_length = 0.233 / 10  # Convert mm to cm


    ### Function Definitions
    def extract_csv_files(folder_path):

        # Get all CSV files in the specified folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        # Return the list of CSV file paths
        return csv_files

    def plot_and_analyse_spectra(csv_file, plot_Flag, plot_save_loc, pair_index, box_size):
        # Read the CSV file
        data = pd.read_csv(csv_file, header=None)
        
        # Extract wavelengths and intensities
        wavelengths = data.iloc[0]  # First row: wavelengths
        intensities = data.iloc[1]  # Second row: intensities

        # Create a mask for wavelengths between 400 and 1000 nm
        mask = (wavelengths >= 400) & (wavelengths <= 1000)
        
        # Filter wavelengths and intensities based on the mask
        filtered_wavelengths = wavelengths[mask]
        filtered_intensities = intensities[mask]

        # Apply Savitzky-Golay filter to smooth intensities
        smooth_intensities = savgol_filter(filtered_intensities, window_length=7, polyorder=2)
        # smoothed_intensities = savgol_filter(intensities, window_length=7, polyorder=2)

        if plot_Flag:

            # Plot the original and smoothed data
            plt.figure(figsize=(10, 6))
            plt.plot(filtered_wavelengths, filtered_intensities, label="Original spectra", color="blue", alpha=0.5)
            plt.plot(filtered_wavelengths, smooth_intensities, label="Smooth spectra", color="red")
            # plt.plot(wavelengths, intensities, label="Original Spectra", color="blue", alpha=0.5)
            # plt.plot(wavelengths, smoothed_intensities, label="Smoothed Spectra", color="red")       
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity")
            plt.title("Spectra plot with Savitzky-Golay filtering")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{plot_save_loc}/spectra_pair{pair_index}_boxsize{box_size}.jpg", dpi=300, bbox_inches='tight')
            #plt.show()
            plt.close()

        # Calculate mean intensity
        mean_intensity = smooth_intensities.mean()
        return mean_intensity

    def extract_pair_number(filename):
        # Use regex to find "pairXXXX" and extract the numeric part
        match = re.search(r"pair(\d{4})", filename)
        if match:
            return int(match.group(1))
        return None

    def analyse_all_csv(folder_path, box_size):

        # Extract CSV files from the specified folder
        csv_files = extract_csv_files(f"{folder_path}/spectra") 
        print(len(csv_files))

        # Determine maximum index to create the numpy array of appropriate size
        max_index = max(extract_pair_number(f) for f in csv_files if extract_pair_number(f) is not None)
        mean_intensities = np.zeros(max_index + 1)  # Create array to hold mean intensities
        
        # Perform analysis on each CSV file independently
        for file_path in csv_files:

            pair_index = extract_pair_number(file_path)
            
            if pair_index is not None:

                if pair_index % 25 == 0:
                    mean_intensity = plot_and_analyse_spectra(file_path, True, S_xy_session_dir, pair_index, box_size)
                    print(pair_index)
                else:
                    mean_intensity = plot_and_analyse_spectra(file_path, False, S_xy_session_dir, pair_index, box_size)
                
                mean_intensities[pair_index] = mean_intensity
        
        return mean_intensities


    folder_path = format_path(folder_path_raw)
    mean_intensities = analyse_all_csv(folder_path, box_size)
    print("Mean intensities array:", mean_intensities)
    print("Length of mean intensities array:",len(mean_intensities))
    print(mean_intensities[0])

    # Function to convert mean_intensities to mean_intensities corrected for background measurement. 
    corrected_mean_intensities = mean_intensities[1:] - mean_intensities[0]

    def find_normalised_sensitivity(mean_i):

        s_min = mean_i.min()
        print("s_min:", s_min)
        s_max = mean_i.max()
        print("s_max:", s_max)

        S_xy = np.zeros(len(mean_i))

        for j in range(len(mean_i)):
            S_xy[j] = (mean_i[j] - s_min)/(s_max - s_min)
            # print(S_xy[j])

        return S_xy

    S_xy = find_normalised_sensitivity(corrected_mean_intensities)
    print("S_xy:", S_xy)
    np.savetxt(f"{S_xy_session_dir}/S_xy.csv", S_xy, delimiter=",", fmt="%.3f", header="Normalised sensitivity matrix S_xy")

    # Function to reshape into white box grid
    def find_num_boxes(box_size):

        # Initialise Pygame
        pygame.init()

        # Set up fullscreen display to get screen dimensions
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        screen_width_pixels, screen_height_pixels = screen.get_size()

        # Calculate how many boxes fit horizontally and vertically
        num_boxes_x = screen_width_pixels // box_size
        num_boxes_y = screen_height_pixels // box_size

        print("Number of boxes (width):", num_boxes_x)
        print("Number of boxes (height):", num_boxes_y)

        # Calculate the centre of the screen in pixels
        centre_x = (screen_width_pixels - 1) / 2
        centre_y = (screen_height_pixels - 1) / 2

        print(f"Centre of screen (x, y): ({centre_x}, {centre_y})")

        # Quit Pygame
        pygame.quit()

        return num_boxes_x, num_boxes_y, centre_x, centre_y


    num_boxes_x, num_boxes_y, centre_x, centre_y = find_num_boxes(box_size)
    # print(w,h)

    # Reshape S_xy based on box size
    S_xy_reshaped = S_xy.reshape(num_boxes_y, num_boxes_x)
    print("Reshaped S_xy:", S_xy_reshaped)
    np.savetxt(f"{S_xy_session_dir}/S_xy_reshaped.csv", S_xy_reshaped, delimiter=",", fmt="%.3f", header="Normalised sensitivity matrix S_xy_reshaped")

    # Plot the matrix
    plt.imshow(S_xy_reshaped, cmap='viridis', aspect='auto')
    plt.colorbar(label="Intensity")  # Add a colorbar to indicate values

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Matrix Plot")

    # Save the plot as an image
    plt.savefig(f"{S_xy_session_dir}/S_xy_heatmap.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # Show the plot
    #plt.show()

    # Binarising matrix
    S_xy_binary = np.where(S_xy_reshaped >= threshold, 1, 0)
    print("Binarised S_xy:\n", S_xy_binary)
    np.savetxt(f"{S_xy_session_dir}/S_xy_binary_threshold{threshold}.csv", S_xy, delimiter=",", fmt="%.3f", header="Binarised Sensitivity Matrix S_xy")

    # Plot the matrix using imshow
    plt.imshow(S_xy_binary, cmap='viridis', aspect='auto')
    plt.colorbar(label="Intensity")  # Add a colourbar to indicate values

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the plot as an image
    plt.savefig(f"{S_xy_session_dir}/S_xy_binary_threshold{threshold}.jpg", dpi=300, bbox_inches='tight')
    plt.close()


    ############### Initialise the output matrix V_xy
    V_xy = np.zeros((num_boxes_y, num_boxes_x))
    theta_xy = np.zeros((num_boxes_y, num_boxes_x))
    d_0_xy = np.zeros((num_boxes_y, num_boxes_x))

    # Calculate the normalised intensity for each box
    for i in range(num_boxes_y):
        for j in range(num_boxes_x):

            # Calculate the centre of the current box in pixels
            box_centre_x = j * box_size + box_size / 2
            box_centre_y = i * box_size + box_size / 2

            # Calculate the distance d_0 in cm
            d_0 = np.sqrt(((box_centre_x - centre_x) * pixel_length) ** 2 +
                        ((box_centre_y - centre_y) * pixel_length) ** 2)
            d_0_xy[i, j] = d_0

            # Calculate the viewing angle theta (in degrees)
            theta = np.degrees(np.arctan(d_0 / d_v))
            theta_xy[i, j] = theta

            # Interpolate the normalised intensity using the cubic spline
            if 0 <= theta <= 90:  # Ensure theta is within the interpolation range

                # V_xy[i, j] = spline(theta)
                V_xy[i, j] = polynomial(theta)

                # Correct for any error in spline/polynomial
                if V_xy[i, j] > 1:
                    V_xy[i, j] = 1 
            else:
                V_xy[i, j] = 0  # Assign a default value if theta is out of range

    # Output the result
    print("Normalised Intensity Matrix (V_xy):")
    print(V_xy)

    print("Theta Matrix (theta_xy):")
    print(theta_xy)

    print("d_0 Matrix (d_0_xy):")
    print(d_0_xy)

    # Save V_xy, theta_xy, and d_0_xy to file
    # np.savetxt("V_xy.csv", V_xy, delimiter=",", fmt="%.3f", header="Normalised Intensity Matrix")
    np.savetxt(f"{A_xy_session_dir}/V_xy.csv", V_xy, delimiter=",", fmt="%.3f", header="Angle Variance Matrix")
    np.savetxt(f"{A_xy_session_dir}/theta_xy.csv", theta_xy, delimiter=",", fmt="%.3f", header="Theta Matrix")
    np.savetxt(f"{A_xy_session_dir}/d_0_xy.csv", d_0_xy, delimiter=",", fmt="%.3f", header="d_0 Matrix")

    # Save V_xy, theta_xy and d_0_xy as a heatmap
    # plt.figure(figsize=(10, 8))
    plt.imshow(V_xy, cmap='viridis', origin='upper')
    plt.colorbar(label='Normalised Intensity')
    plt.title('Heatmap of Angle Variance (V_xy)')
    plt.xlabel('Box Index (Horizontal)')
    plt.ylabel('Box Index (Vertical)')
    plt.savefig(f"{A_xy_session_dir}/V_xy_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # plt.figure(figsize=(10, 8))
    plt.imshow(theta_xy, cmap='viridis', origin='upper')
    plt.colorbar(label='Normalised Intensity')
    plt.title('Heatmap of Theta (theta_xy)')
    plt.xlabel('Box Index (Horizontal)')
    plt.ylabel('Box Index (Vertical)')
    plt.savefig(f"{A_xy_session_dir}/theta_xy_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # plt.figure(figsize=(10, 8))
    plt.imshow(d_0_xy, cmap='viridis', origin='upper')
    plt.colorbar(label='Normalised Intensity')
    plt.title('Heatmap of d_0 (d_0_xy)')
    plt.xlabel('Box Index (Horizontal)')
    plt.ylabel('Box Index (Vertical)')
    plt.savefig(f"{A_xy_session_dir}/d_0_xy_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
        
    ############# Determine A_xy, normalised sensitivity matrix corrected for angle variance
    A_xy = np.multiply(S_xy_reshaped, V_xy)

    # Save to file
    np.savetxt(f"{A_xy_session_dir}/A_xy.csv", A_xy, delimiter=",", fmt="%.3f", header="Sensitivity Matrix corrected for Angle Variance, A_xy")
    # plt.figure(figsize=(10, 8))
    plt.imshow(A_xy, cmap='viridis', origin='upper')
    plt.colorbar(label='Normalised Intensity')
    plt.title('Heatmap of Corrected Sensitivity Matrix (A_xy)')
    plt.xlabel('Box Index (Horizontal)')
    plt.ylabel('Box Index (Vertical)')
    plt.savefig(f"{A_xy_session_dir}/A_xy_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


    ############ Binarised A_xy
    # Binarising matrix
    A_xy_binary = np.where(A_xy >= threshold, 1, 0)
    print("Binarised A_xy:\n", A_xy_binary)
    np.savetxt(f"{A_xy_session_dir}/A_xy_binary_threshold{threshold}.csv", A_xy, delimiter=",", fmt="%.3f", header="Binarised Corrected Sensitivity Matrix A_xy")

    # Plot the matrix using imshow
    plt.imshow(A_xy_binary, cmap='viridis', aspect='auto')
    plt.colorbar(label="Intensity")  # Add a colourbar to indicate values

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Matrix Plot")

    # Save the plot as an image
    plt.savefig(f"{A_xy_session_dir}/FOV_sensitivity_binary{box_size}.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # Recreate grid using pygame and take an image with the camera
    ### Initialise spectrometer ###
    # Find spectrometer
    devices = sb.list_devices()
    if not devices:
        raise RuntimeError("No spectrometers found. Exiting...")
        exit() # Might not be necessary if error is raised

    # Select spectrometer on 0th index
    spec = sb.Spectrometer(devices[0])
    print(f"Spectrometer is: {spec}.\n")

    # Set integration time to 1 second
    integration_time = 1000 # milliseconds
    spec.integration_time_micros(integration_time * 1000) # microseconds

    # Initialise camera
    # Print PyCapture 2 Library Information
    print_build_info()

    # Ensure sufficient cameras are found
    bus = PyCapture2.BusManager()
    num_cams = bus.getNumOfCameras()
    print('Number of cameras detected: ', num_cams)
    if not num_cams:
        raise RuntimeError("Insufficient number of cameras. Exiting...")
        # exit()

    # Select camera on 0th index
    c = PyCapture2.Camera()
    uid = bus.getCameraFromIndex(0)
    c.connect(uid)
    print_camera_info(c)

    # Enable camera embedded timestamp
    enable_embedded_timestamp(c, True)
    c.startCapture()

    # Put a delay
    # input("Ensure the camera is set still at the same position at which calibration was measured. \
    #     \nMove the mouse out of the screen. \
    #     \nTurn the lights off. After the first beep is played, turn the lights back on.")
    print("Ensure the camera is set still at the same position at which calibration was measured. \
        \nMove the mouse out of the screen. \
        \nTurn the lights off. After the first beep is played, turn the lights back on.")
    time.sleep(5)

    # Initialise Pygame
    pygame.init()
    pygame.mixer.init()

    # Set up fullscreen mode and get screen dimensions
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_width, screen_height = screen.get_size()
    pygame.display.set_caption("Display Matrix as Grid")

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        print("Capturing S_xy_binary...\n")

        # Clear the screen to black
        screen.fill((0, 0, 0))

        # Draw the grid based on the matrix values
        for i in range(num_boxes_y):
            for j in range(num_boxes_x):
                if S_xy_binary[i,j] == 1:
                    pygame.draw.rect(screen, (255, 255, 255), (j * box_size, i * box_size, box_size, box_size))

        # Update the display
        pygame.display.flip()

        time.sleep(5)

        # Take picture in dark
        current_pair = 1
        n_spectra = 1 # Number of spectra to capture for averaging
        capture_spectrum_and_image(spec, c, integration_time, current_pair, S_xy_session_dir, n_spectra)

        time.sleep(5)

        # # Generate beep
        # pygame.mixer.Sound(beep_filepath).play()

        # time.sleep(10)

        # # Take picture in light
        # current_pair += 1
        # capture_spectrum_and_image(spec, c, integration_time, current_pair, S_xy_session_dir, n_spectra)

        # time.sleep(5)

        # current_pair += 1
        # integration_time = 200
        # capture_spectrum_and_image(spec, c, integration_time, current_pair, S_xy_session_dir, n_spectra)

        # # Generate beep
        # pygame.mixer.Sound(beep_filepath).play()

        print("Capturing A_xy_binary...\n")

        # Clear the screen to black
        screen.fill((0, 0, 0))

        # Draw the grid based on the matrix values
        for i in range(num_boxes_y):
            for j in range(num_boxes_x):
                if A_xy_binary[i,j] == 1:
                    pygame.draw.rect(screen, (255, 255, 255), (j * box_size, i * box_size, box_size, box_size))

        # Update the display
        pygame.display.flip()

        time.sleep(5)

        # Take picture in dark
        current_pair = 1
        n_spectra = 1 # Number of spectra to capture for averaging
        capture_spectrum_and_image(spec, c, integration_time, current_pair, A_xy_session_dir, n_spectra)

        time.sleep(5)

        print("Final capture recorded")

        # Check if at last entry
        if k == len(threshold_range) - 1:

            num_sounds = 30
            sound_i = 1

            while sound_i <= num_sounds:

                pygame.mixer.Sound(beep_filepath).play()    
                sound_i += 1
                time.sleep(2)

        pygame.time.wait(1000)  # Pause briefly before quitting
        pygame.quit()
        pygame.mixer.quit()
        running = False
        # sys.exit()

    # Quit Pygame
    # pygame.quit()
    # pygame.mixer.quit()
    c.stopCapture()
    c.disconnect()
    spec.close()

    time.sleep(3)
