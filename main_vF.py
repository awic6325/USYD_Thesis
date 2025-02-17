### Synchronous data acquisition of image-spectrum pairs

# Packages
import seabreeze.spectrometers as sb
import numpy as np
import matplotlib.pyplot as plt
import time
import PyCapture2
import datetime
import os
import keyboard
import threading
import pandas as pd
import pygame

# Import functions
from spectrum_handler import autoexposure, get_spectralon_or_dark_current_reading, get_spectrum, get_mask
from image_handler import print_build_info, print_camera_info, enable_embedded_timestamp, grab_images
from file_handler import get_timestamp, generate_filename, save_image, save_spectrum, \
                         set_working_directory_to_script_location, create_new_session_folder

# Set the working directory to the program file's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Function to run spectrum and image capture
def capture_spectrum_and_image(spectrometer, camera, opt_int_time, pair_index, save_dir, n_spectra=1, mask_Flag=False, averaging=False, exposure_time=1200):

    # Subdirectory to store images and spectra
    images_dir    = f"{save_dir}/images"
    spectra_dir      = f"{save_dir}/spectra"

    # Create folder
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(spectra_dir, exist_ok=True)
    
    # Image capture
    img_time = get_timestamp()
    img_data = grab_images(camera, 1, exposure_time)

    # Image storage
    img_filename = generate_filename("image", pair_index, img_time)
    save_image(img_data, images_dir, img_filename)

    if averaging:
        if mask_Flag:
            _, _, mask, mask_length = get_mask(spectrometer)
            spectra = np.zeros((n_spectra, mask_length))
        else:
            # Initialise vector to average over
            spectrum_length = spectrometer.pixels
            spectra = np.zeros((n_spectra, spectrum_length))

        # Loop enough times for spectra
        for i in range(n_spectra):

            # Spectrum capture
            spec_time = get_timestamp()
            spec_data = get_spectrum(spectrometer, opt_int_time, mask_Flag)

            spectra[i] = spec_data[1,:]
            
            print(i)

        # Get mean spectrum
        mean_intensities = np.mean(spectra, axis=0)
        wavelengths = spectrometer.wavelengths()
        if mask_Flag:
            wavelengths = wavelengths[mask]
        mean_spectrum = np.vstack((wavelengths, mean_intensities))

        # Spectrum storage
        spec_filename = generate_filename("spectrum", pair_index, spec_time)
        save_spectrum(mean_spectrum, spectra_dir, spec_filename)

        return mean_spectrum

    else:
        # Loop enough times for spectra
        for i in range(n_spectra):

            # Spectrum capture
            spec_time = get_timestamp()
            spec_data = get_spectrum(spectrometer, opt_int_time, mask_Flag)

            # Spectrum storage
            spec_filename = generate_filename("spectrum", pair_index, spec_time)
            save_spectrum(spec_data, spectra_dir, spec_filename)

        return None

### Flag to control the loop ###
stop_loop = False

def listen_for_exit():
    global stop_loop
    while True:
        if input() == 'q':
            stop_loop = True
            break

def run_until_key_press(spectrometer, camera, opt_int_time, save_dir, exposure_time=1200):

    pygame.mixer.init()
    beep_filepath = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\HardwareTest\beep-01a.wav"

    global stop_loop
    print("Press 'q' to stop the loop.")

    # Start the listener thread
    listener_thread = threading.Thread(target=listen_for_exit)
    listener_thread.start()

    current_pair = 1

    while not stop_loop:
        capture_spectrum_and_image(spectrometer, camera, opt_int_time, current_pair, save_dir, n_spectra=1, mask_Flag=False, averaging=False, exposure_time=exposure_time)
        current_pair += 1

        if (current_pair % 10) == 0:
            print(f"Number of image-spectrum pairs captured so far: {current_pair}") 

        if (current_pair % 50) == 0:
        # if (current_pair % 30) == 0:
            pygame.mixer.Sound(beep_filepath).play()

        time.sleep(0.01)  # Short delay to prevent busy waiting

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

# Function to run the whole process
def daq_pipeline():

    # Request user to enter metadata to describe current experiment
    metadata = input("\nPlease enter metadata to describe the current experiment:\n")
    # metadata = ""

    ### Set up data storage directory
    data_dir = "../device_data"
    os.makedirs(data_dir, exist_ok=True)
    current_session_dir = create_new_session_folder(data_dir, metadata)
    print(current_session_dir)

    # Subdirectories to store autoexposure and R_spectralon
    # Path
    autoexposure_dir    = f"{current_session_dir}/autoexposure"
    spectralon_dir      = f"{current_session_dir}/spectralon"

    # Create folder
    os.makedirs(autoexposure_dir, exist_ok=True)
    os.makedirs(spectralon_dir, exist_ok=True)

    ### Initialise spectrometer ###
    # Find spectrometer
    devices = sb.list_devices()
    if not devices:
        raise RuntimeError("No spectrometers found. Exiting...")
        exit() # not necessary
    
    # Select spectrometer on 0th index
    spec = sb.Spectrometer(devices[0])
    print(f"\nSpectrometer is: {spec}.\n")

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

    ###### TODO: UNCOMMENT #################################
    # Set exposure time to 20000 µs (20 ms)
    # cam_exposure_time = 5000
    # cam_exposure_time = 2000
    # cam_exposure_time = 3000
    # cam_exposure_time = 2500
    # cam_exposure_time = 2000
    # cam_exposure_time = 1500
    cam_exposure_time = 1200
    # cam_exposure_time = 1000

    set_exposure_time(c, cam_exposure_time)

    ###### TODO: COMMENT OUT #################################
    # c.setProperty(type=PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, autoManualMode=True)1


    ### Run autoexposure
    # Get optimal integration time for current illumination conditions
    optimal_integration_time, auto_df = autoexposure(spec)
    print(f"\nOptimal integration time: {optimal_integration_time} ms\n")

    # Save dataframe to file
    auto_df.to_csv(f"{autoexposure_dir}/autoexposure_data.csv", index=False)


    ###### TODO: COMMENT OUT #################################
    # TODO: Minimise spectrometer integration time somehow. 
    # - Use brighter illumination condition?
    # - Find way to make sure spectrometer is pointed at Spectralon panel. Look at FlyCap2?
    # - Set autoexposure algorithm limits to [10, 200] instead of max 250?
    # optimal_integration_time = 10


    # Set optimal integration time
    spec.integration_time_micros(optimal_integration_time * 1000)

    # Get dark current value - not necessary
    # dark_current = get_spectralon_or_dark_current_reading(spec, optimal_integration_time, False)

    # Get Spectralon reading beforehand, and save to file
    R_Spectralon_before = get_spectralon_or_dark_current_reading(spec, optimal_integration_time, False, True)
    save_spectrum(R_Spectralon_before, spectralon_dir, "R_spectralon_before")

    # Subdirectory to store synchronous data capture
    daq_dir = f"{current_session_dir}/daq_{optimal_integration_time}ms"
    os.makedirs(daq_dir, exist_ok=True)

    input("\nPress ENTER to start the synchronous data acquisition of image-spectrum pairs.\n")
    time.sleep(2)

    ### Start synchronous data capture
    c.startCapture()
    run_until_key_press(spec, c, optimal_integration_time, daq_dir, exposure_time=cam_exposure_time)
    
    # Get Spectralon reading after
    R_Spectralon_after = get_spectralon_or_dark_current_reading(spec, optimal_integration_time, False, True)
    save_spectrum(R_Spectralon_after, spectralon_dir, "R_spectralon_after")

    # Close camera
    c.stopCapture()
    enable_embedded_timestamp(c, False)
    c.disconnect()
    

# Run Main program
if __name__ == "__main__":

    daq_pipeline()