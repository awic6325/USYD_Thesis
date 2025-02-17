import cv2
import numpy as np
import glob
import os

def process_directory(root_dir, mtx, dist):

    # Walk through all files and subdirectories within the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):

        # Filter for .png files in the current directory
        png_files = [f for f in filenames if f.lower().endswith('.png')]
        
        if png_files:

            # Create a new adjacent folder for undistorted images
            undistorted_dir = f"{dirpath}_undistorted"
            os.makedirs(undistorted_dir, exist_ok=True)
            
            # Process each .png file in the current directory
            for file_name in png_files:
                image_path = os.path.join(dirpath, file_name)
                undistorted_image = undistort_image(image_path, mtx, dist)
                
                # Save the undistorted image in the new directory
                undistorted_image_path = os.path.join(undistorted_dir, file_name)
                cv2.imwrite(undistorted_image_path, undistorted_image)
                print(f"Saved undistorted image: {undistorted_image_path}")

# def calibrate_camera(chessboard_images_path, chessboard_size=(8, 6), square_size=25):
def calibrate_camera(chessboard_images_path, output_path, chessboard_size=(8, 6), square_size=35):

    # Define the chessboard object points with real-world scale
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get all images
    images = glob.glob(chessboard_images_path)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            # If found, add object points and image points
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw and display the corners for verification
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(100)

            # Save image with detected corners
            base_name = os.path.basename(fname)
            output_file = os.path.join(output_path, base_name)
            cv2.imwrite(output_file, img)
    
    cv2.destroyAllWindows()
    
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if ret:
        print("Camera calibrated successfully")
        print("Camera Matrix:\n", mtx)
        print("Distortion Coefficients:\n", dist)
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        print("Reprojection Error:", mean_error)
        
        return mtx, dist, mean_error
    else:
        print("Camera calibration failed")
        return None, None, None


def undistort_image(image_path, mtx, dist, show_image=False):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Get optimal camera matrix - better undistortion
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # Undistort the image
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
    
    # Crop the image, remove warped/black regions
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    
    if show_image:
        # Display the result
        cv2.imshow("Original Image", img)
        cv2.imshow("Undistorted Image", undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return undistorted_img



### n = 100
path = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\project_v2\code_v2\camera_calibration\20250103_session22_20mm_19x13_n100_20ms"
path_to_chessboard_imgs = f"{path}/*.png"
path_to_test_img = f"{path}/image_pair0001_20250103_155446_827.png"
output_path = f"{path}/corners_drawn"
os.makedirs(output_path, exist_ok=True)

mtx, dist, mean_error = calibrate_camera(path_to_chessboard_imgs, output_path, chessboard_size=(19, 13), square_size=20)   # A3 sheet
undistorted_image = undistort_image(path_to_test_img, mtx, dist, show_image=True)
