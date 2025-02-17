import pygame
import numpy as np
import pandas as pd
import time
import seabreeze.spectrometers as sb
import os
import PyCapture2

# Import functions
from spectrum_handler import get_spectrum, autoexposure
from image_handler import print_build_info, print_camera_info, enable_embedded_timestamp, grab_images
from file_handler import get_timestamp, generate_filename, save_image, save_spectrum, \
                         set_working_directory_to_script_location, create_new_session_folder, format_path
from main_v3 import capture_spectrum_and_image, set_exposure_time

# Set viewing distance
distance = 15
# distance = 30
# distance = 60
# distance = 90
# distance = 120
# distance = 150

other_notes = ""
# other_notes = "_redundant"

metadata = f"d{distance}cm{other_notes}"


# Beep noise
beep_filepath = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\HardwareTest\beep-01a.wav"


# Set the working directory to the program file's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

### Set up data storage directory
data_dir = "../angle_var"
os.makedirs(data_dir, exist_ok=True)

current_session_dir = create_new_session_folder(data_dir, metadata) #, distance)

# Subdirectories
autoexposure_dir    = f"{current_session_dir}/autoexposure"
anglevar_df_dir       = f"{current_session_dir}/anglevar_df"

# Create directories
os.makedirs(autoexposure_dir, exist_ok=True)
os.makedirs(anglevar_df_dir, exist_ok=True)

### Initialise spectrometer ###
# Find spectrometer
devices = sb.list_devices()
if not devices:
    raise RuntimeError("No spectrometers found. Exiting...")
    exit() # Might not be necessary if error is raised

# Select spectrometer on 0th index
spec = sb.Spectrometer(devices[0])
print(f"Spectrometer is: {spec}.\n")


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

# Set exposure time to 10 ms #20 ms
set_exposure_time(c, 10000)
# set_exposure_time(c, 20000)

# Enable camera embedded timestamp
enable_embedded_timestamp(c, True)
c.startCapture()


# Message to ensure spectrometer is set
input("\nPlace the spectrometer facing the white screen directly. \
       \n\nPress ENTER to start the autoexposure process.")


### Initialise pygame
pygame.init()
pygame.mixer.init()

# Get the screen's width and height for fullscreen mode
screen_info = pygame.display.Info()
screen_width, screen_height = screen_info.current_w, screen_info.current_h

# Set up the screen in fullscreen mode
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption('Fullscreen Box Fill with Edge Alignment')


# Fill the screen with white initially
screen.fill((255, 255, 255))
pygame.display.flip()

# Define the clock for controlling the frame rate
clock = pygame.time.Clock()

# Generate beep
pygame.mixer.Sound(beep_filepath).play()

 # Delay to ensure camera doesn't capture mid-motion
pygame.time.delay(15*1000)

# Generate beep
pygame.mixer.Sound(beep_filepath).play()

# Autoexposure algorithm for spectrometer to determine integration time at 0 degrees
optimal_integration_time, auto_df = autoexposure(spec, target_percentage=0.85, min_integration_time=10, max_integration_time=1000, step=10, press_enter=False)
print(f"\nOptimal integration time: {optimal_integration_time} ms\n")

# Save dataframe to file
auto_df.to_csv(f"{autoexposure_dir}/autoexposure_data.csv", index=False)

# Generate beep
pygame.mixer.Sound(beep_filepath).play()

# Quit Pygame
pygame.quit()



# Now take measurements
# Degrees: 0, 10, 20, 30, 40, 50, 60, ...
# Can be a flexible size - use an input maybe to check whether or not to continue?
# Get image-spectrum capture. Get 10 spectra for each angle, and average. 
# Input number of next degree, otherwise enter 'q' to terminate
# Store as dataframe, with theta vs mean_intensity
# Always start with 0 degrees
take_measurements = True
measurement_count = 0
angle = None
n_spectra = 10 # Number of spectra to capture for averaging
# left_lim = 400
# right_lim = 1000
# wavelengths = spectrometer.wavelengths()
# mask = (wavelengths >= left_lim) & (wavelengths <= right_lim)

# Initialise a dataframe
# Headers: Angle, Mean Spectrum, Mean Intensity, Normalised Intensity
headers = ["Angle", "Mean Spectrum", "Mean Intensity", "Normalised Intensity"]
anglevar_df = pd.DataFrame(columns=headers)
print(anglevar_df)

# Acquire angle variance measurements
while take_measurements:

    # First measurement 
    if measurement_count == 0:
        input("Place the spectrometer directly facing the screen (angle = 0 degrees). Press ENTER to continue.\n")
        angle = 0
    else:
        input(f"Place the spectrometer at an angle of {angle} degrees towards the screen. Press ENTER to continue.\n")

    ### Initialise pygame
    pygame.init()
    pygame.mixer.init()

    # Get the screen's width and height for fullscreen mode
    screen_info = pygame.display.Info()
    screen_width, screen_height = screen_info.current_w, screen_info.current_h

    # Set up the screen in fullscreen mode
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption('Fullscreen Box Fill with Edge Alignment')


    # Fill the screen with white initially
    screen.fill((255, 255, 255))
    pygame.display.flip()

    # Define the clock for controlling the frame rate
    clock = pygame.time.Clock()

    # Generate beep
    pygame.mixer.Sound(beep_filepath).play()

    # Delay to ensure camera doesn't capture mid-motion
    pygame.time.delay(15*1000)

    # Generate beep
    pygame.mixer.Sound(beep_filepath).play()
    
    # Take image-spectrum capture, with 10 captures averaged to obtain the spectrum
    mean_spectrum = capture_spectrum_and_image(spec, c, optimal_integration_time, measurement_count, current_session_dir, n_spectra, mask_Flag=True, averaging=True)
    print(mean_spectrum)
    print("Dimensions:", mean_spectrum.shape)

    # Generate beep
    pygame.mixer.Sound(beep_filepath).play()

    # Quit Pygame
    pygame.quit()

    # Get mean intensity
    mean_intensity = np.mean(mean_spectrum[1,:])

    # Update dataframe
    anglevar_df.loc[measurement_count] = [angle, mean_spectrum, mean_intensity, None]

    # Get next measurement value from user
    next_measurement = input("Enter the next angle to take measurement, otherwise enter 'q' to quit.\n")

    # Check if the input is 'q'
    if next_measurement.lower() == 'q':
        print("\nExiting measurement process.\n")
        take_measurements = False
        break

    # Check if the input is an integer between 0 and 90
    try:
        angle = int(next_measurement)
        if 0 <= angle <= 90:
            print(f"\nValid angle entered: {angle}\n")
            # Update count
            measurement_count += 1

        else:
            print("\nError: Angle must be an integer between 0 and 90.\n")
    except ValueError:
        print("\nError: Invalid input. Enter an integer between 0 and 90, or 'q' to quit.\n")


# Print dataframe
print(anglevar_df)
    
# Sort data frame based on increasing angle, i.e. based on angle column
anglevar_df = anglevar_df.sort_values(by="Angle", ascending=True)
anglevar_df.reset_index(drop=True, inplace=True)    # reset index

# Get the mean intensity at angle 0 degrees
I_0 = anglevar_df.loc[anglevar_df["Angle"] == 0, "Mean Intensity"].values[0]

# Determine the normalised intensity
anglevar_df["Normalised Intensity"] = anglevar_df["Mean Intensity"] / I_0

print(anglevar_df)

# Store the dataframe
anglevar_df.to_csv(f"{anglevar_df_dir}/anglevar_{distance}cm_df.csv", index=False)

# Afterwards, do calculation based on box size (variable)
# Assumption is that point spectrometer is pointing to the centre of the screen
# Calculate expected intensity adjusted for angle variance based on box position and box size, based on centre of box
