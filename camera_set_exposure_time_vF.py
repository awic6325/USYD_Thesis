import PyCapture2

from image_handler import print_build_info, print_camera_info, enable_embedded_timestamp, grab_images

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

    print("Camera is configured with manual exposure and auto gain.")
    print(f"Exposure time set to: {exposure_time_us} µs.")

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
# set_exposure_time(c, 10000)
set_exposure_time(c, 20000)
# set_exposure_time(c, 30000)