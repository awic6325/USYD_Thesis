# Packages
import time
import PyCapture2
import os

# Import functions
from spectrum_handler import autoexposure, get_spectralon_or_dark_current_reading, get_spectrum
from image_handler import print_build_info, print_camera_info, enable_embedded_timestamp, grab_images
from file_handler import get_timestamp, generate_filename, save_image, save_spectrum, \
                         set_working_directory_to_script_location, create_new_session_folder


def set_exposure_time(cam, exposure_time_us):

    # Disable auto-exposure
    cam.setProperty(type=PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, autoManualMode=False)

    # Set exposure time in microseconds
    exposure_property = PyCapture2.Property()
    exposure_property.type = PyCapture2.PROPERTY_TYPE.SHUTTER
    exposure_property.autoManualMode = False
    exposure_property.absControl = True  # Set in absolute mode
    exposure_property.absValue = exposure_time_us / 1000  # convert microseconds to milliseconds
    cam.setProperty(exposure_property)

    # Enable auto gain
    cam.setProperty(type=PyCapture2.PROPERTY_TYPE.GAIN, autoManualMode=True)

    print("Camera configured with manual exposure and auto gain.")
    print(f"Exposure time set to: {exposure_time_us} µs.")


# Set the working directory to the program file's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

# Enable camera embedded timestamp
enable_embedded_timestamp(c, True)

########## Set exposure time to 5 ms # TODO: Move to image_handler.py, import from there
# set_exposure_time(c, 30000)
set_exposure_time(c, 20000)

###
c.startCapture()

# Request the user to enter metadata to describe current experiment
metadata = input("\nPlease enter metadata to describe the current experiment:\n")

### Set up data storage directory
data_dir = "../camera_calibration"
os.makedirs(data_dir, exist_ok=True)
current_session_dir = create_new_session_folder(data_dir, metadata)

# num_images = 25
# num_images = 50
# num_images = 75
num_images = 100
save_dir = current_session_dir

input("\nPress ENTER to start the data acquisition of camera calibration images.\n")
time.sleep(2)

for i in range(num_images):

    pair_index = i+1
    print(f"Capturing image {pair_index}...\n")

    # Image capture
    img_time = get_timestamp()
    img_data = grab_images(c, 1)

    # Image storage
    img_filename = generate_filename("image", pair_index, img_time)
    save_image(img_data, save_dir, img_filename)

    time.sleep(0.5)

c.stopCapture()
c.disconnect()