import PyCapture2

######################################### CAMERA #########################################
def print_build_info():
    lib_ver = PyCapture2.getLibraryVersion()
    print('PyCapture2 library version: %d %d %d %d' % (lib_ver[0], lib_ver[1], lib_ver[2], lib_ver[3]))
    print()

def print_camera_info(cam):
    cam_info = cam.getCameraInfo()
    print('\n*** CAMERA INFORMATION ***\n')
    print('Serial number - %d' % cam_info.serialNumber)
    print('Camera model - %s' % cam_info.modelName)
    print('Camera vendor - %s' % cam_info.vendorName)
    print('Sensor - %s' % cam_info.sensorInfo)
    print('Resolution - %s' % cam_info.sensorResolution)
    print('Firmware version - %s' % cam_info.firmwareVersion)
    print('Firmware build time - %s' % cam_info.firmwareBuildTime)
    print()

def enable_embedded_timestamp(cam, enable_timestamp):
    embedded_info = cam.getEmbeddedImageInfo()
    if embedded_info.available.timestamp:
        cam.setEmbeddedImageInfo(timestamp = enable_timestamp)
        if enable_timestamp :
            print('\nTimeStamp is enabled.\n')
        else:
            print('\nTimeStamp is disabled.\n')

# Rework this function later to be simpler
def grab_images(cam, num_images_to_grab, exposure_time_us=5000):
    image = None
    prev_ts = None
    for i in range(num_images_to_grab):
        try:
            image = cam.retrieveBuffer()
        except PyCapture2.Fc2error as fc2Err:
            print('Error retrieving buffer : %s' % fc2Err)
            continue

        ts = image.getTimeStamp()
        if prev_ts:
            diff = (ts.cycleSeconds - prev_ts.cycleSeconds) * 8000 + (ts.cycleCount - prev_ts.cycleCount)
            print('Timestamp [ %d %d ] - %d' % (ts.cycleSeconds, ts.cycleCount, diff))
        prev_ts = ts

    applied_exposure = cam.getProperty(PyCapture2.PROPERTY_TYPE.SHUTTER).absValue * 1000
    
    if abs(applied_exposure - exposure_time_us) < 50:
        # print(f"Exposure set and verified successfully: {applied_exposure} µs.")
        # return True
        pass
    else:
        print(f"Warning: Applied exposure ({applied_exposure} µs) differs from requested ({exposure_time_us} µs).")
        # return False

    newimg = image.convert(PyCapture2.PIXEL_FORMAT.BGR)
    return newimg


