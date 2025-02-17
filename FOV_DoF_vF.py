import pygame
import time
import seabreeze.spectrometers as sb
import os
import PyCapture2
import sys

# Import functions
from spectrum_handler import get_spectrum, autoexposure
from image_handler import print_build_info, print_camera_info, enable_embedded_timestamp, grab_images
from file_handler import get_timestamp, generate_filename, save_image, save_spectrum, \
                         set_working_directory_to_script_location, create_new_session_folder, format_path
from main_v3 import capture_spectrum_and_image, set_exposure_time

# Box size in pixels
box_size = 60

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
parent_dir = "../camera_calibration"
os.makedirs(parent_dir, exist_ok=True)

# Subdirectories
FOV_DoF_dir = f"{parent_dir}/FOV_DoF"

# Create directories
os.makedirs(FOV_DoF_dir, exist_ok=True)

current_session_dir = create_new_session_folder(FOV_DoF_dir, metadata) #, distance)


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
exposure_time = 20000
# set_exposure_time(c, 10000)
# set_exposure_time(c, 20000)
set_exposure_time(c, exposure_time)

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
pygame.display.set_caption('Chessboard Pattern')


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
auto_df.to_csv(f"{current_session_dir}/autoexposure_data.csv", index=False)

# Generate beep
pygame.mixer.Sound(beep_filepath).play()

# Quit Pygame
pygame.quit()


def draw_chessboard(screen, square_size):
    ### Draws a chessboard pattern on the given screen.

    # Get the dimensions of the screen
    width, height = screen.get_size()

    # Calculate the number of rows and columns
    rows = height // square_size
    cols = width // square_size

    # Draw the squares
    for row in range(rows):
        for col in range(cols):
            
            # Alternate the colour based on the row and column index
            if (row + col) % 2 == 0:
                colour = (255, 255, 255)  # White
            else:
                colour = (0, 0, 0)  # Black

            # Calculate the position and size of the square
            x = col * square_size
            y = row * square_size
            pygame.draw.rect(screen, colour, pygame.Rect(x, y, square_size, square_size))


# Main loop
n_capture = 10
pair_index = 1
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw the chessboard
    screen.fill((0, 0, 0))  # Clear the screen
    draw_chessboard(screen, box_size)

    time.sleep(5)

    capture_spectrum_and_image(spec, c, optimal_integration_time, pair_index, current_session_dir, exposure_time=exposure_time)

    # Update the display
    pygame.display.flip()

    # Update parameters
    n_capture -= 1
    pair_index += 1

    if n_capture == 0:
        running = False

pygame.quit()
sys.exit()
