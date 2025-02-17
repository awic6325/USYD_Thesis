### Determines hyperspectral 3D maps (absolute reflectance)
### Printing suppressed to speed up computation

# Packages
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import ast
import math
import os
import time


def reproject_points(shared_dir, export_dir, colmap_point_cloud_file, ellipse_params_df_file, s_world, Z_c_min=15, Z_c_max=180, savgol_w=11, gauss_weight=0.6, interpolant_kind='cubic', metadata="", visualisation=False, printing=False):

    # Start the timer
    start_time = time.time()

    # Unload subdirectories and files
    images_dir = f"{shared_dir}\images_undistorted"
    spectra_dir = f"{shared_dir}\spectra_savgol_w{savgol_w}_p2_corrected"
    cameras_df_file = f"{export_dir}\cameras_df.csv"
    images_data_df_file = f"{export_dir}\images_data_df.csv"
    images_points2D_df_file = f"{export_dir}\images_points2D_df.csv"
    points3D_df_file = f"{export_dir}\points3D_df.csv"

    # Read in CSV files into dataframes
    ellipse_params_df   = pd.read_csv(ellipse_params_df_file)
    cameras_df          = pd.read_csv(cameras_df_file)
    images_data_df      = pd.read_csv(images_data_df_file)
    images_points2D_df  = pd.read_csv(images_points2D_df_file)
    points3D_df         = pd.read_csv(points3D_df_file)

    # Pre-index dataframes as dictionaries
    cameras_dict = cameras_df.set_index('CAMERA_ID').to_dict('index')
    images_data_dict = images_data_df.set_index('IMAGE_ID').to_dict('index')
    points3D_dict = points3D_df.set_index('POINT3D_ID').to_dict('index')

    # Get image width and height in pixels from cameras_df
    image_width = cameras_df['WIDTH'].iloc[0]
    image_height = cameras_df['HEIGHT'].iloc[0]

    if printing:
        # Print info
        print(ellipse_params_df.dtypes)
        print(cameras_df.dtypes)
        print(images_data_df.dtypes)
        print(images_points2D_df.dtypes)
        print(points3D_df.dtypes)

        print("cameras_dict:", cameras_dict)
        print("images_data_dict:", images_data_dict)
        # print("points3D_dict:", points3D_dict)

        print("Image Width:", image_width)
        print("Image Height:", image_height)

    # Function to get extrinsic matrix
    def get_extrinsic_matrix(qw, qx, qy, qz, tx, ty, tz):
        rotation = R.from_quat([qx, qy, qz, qw])
        R_matrix = rotation.as_matrix()
        t_vector = np.array([tx, ty, tz]).reshape(3, 1)
        # extrinsic_matrix = np.hstack((R_matrix, t_vector))
        extrinsic_matrix_top = np.hstack((R_matrix, t_vector))
        extrinsic_matrix_bottom = np.array([0, 0, 0, 1])
        extrinsic_matrix = np.vstack((extrinsic_matrix_top, extrinsic_matrix_bottom))
        return extrinsic_matrix

    # Function to project 3D points to 2D image plane
    def project_point(intrinsic_matrix, extrinsic_matrix, point_world):
        point_camera = extrinsic_matrix @ np.append(point_world, 1)  # [X, Y, Z, 1] in camera frame

        # print("Point Camera:", point_camera)

        u, v, w = intrinsic_matrix @ point_camera
        # print("u,v,w:",u,v,w)
        return np.array([u/w, v/w]), point_camera[0:3]  # Return [u, v] and depth Z_C

    # Ellipse interpolation
    def interpolate_all_ellipse_parameters(ellipse_df, distance, interpolant_kind='cubic'):
        
        ### Interpolate all ellipse parameters at a given distance
        ### Parameters:
        #    ellipse_df - DataFrame containing ellipse parameters, i.e. 'Distance', 'Centre_X', 'Centre_Y', 'Major_Axis', 'Minor_Axis', and 'Angle'
        #    distance - target distance for interpolation
        # Returns: Interpolated ellipse parameters at the given distance, as dict.

        # Ensure the df is sorted by distance
        ellipse_df = ellipse_df.sort_values('Distance')

        # Columns to interpolate (exclude filename)
        columns_to_interpolate = ['Centre_X', 'Centre_Y', 'Major_Axis', 'Minor_Axis', 'Angle']

        # Create interpolation functions for each column
        interpolation_functions = {
            # col: interp1d(ellipse_df['Distance'], ellipse_df[col], kind='linear', fill_value="extrapolate")
            # col: interp1d(ellipse_df['Distance'], ellipse_df[col], kind='quadratic', fill_value="extrapolate")
            # col: interp1d(ellipse_df['Distance'], ellipse_df[col], kind='cubic', fill_value="extrapolate")
            col: interp1d(ellipse_df['Distance'], ellipse_df[col], kind=interpolant_kind, fill_value="extrapolate")
            for col in columns_to_interpolate
        }
        
        # Interpolate all parameters
        interpolated_values = {col: interpolation_functions[col](distance) for col in columns_to_interpolate}
        interpolated_values['Distance'] = distance  # Add the given distance to the result
        
        return interpolated_values


    # Function to get weighting 
    def gaussian_weighting(interpolated_ellipse, x, y, boundary_weighting=0.6):
        
        ### Determines if a point (x, y) lies within the ellipse, then calculates its Gaussian weighting

        # Extract ellipse parameters
        cx = interpolated_ellipse['Centre_X']
        cy = interpolated_ellipse['Centre_Y']
        a = interpolated_ellipse['Major_Axis'] / 2  # Semi-major axis
        b = interpolated_ellipse['Minor_Axis'] / 2  # Semi-minor axis
        angle = math.radians(interpolated_ellipse['Angle'])  # Convert angle to radians

        # Translate the point relative to the ellipse centre
        dx = x - cx
        dy = y - cy

        # Rotate the point into the ellipse's frame of reference
        rotated_x = dx * math.cos(angle) + dy * math.sin(angle)
        rotated_y = -dx * math.sin(angle) + dy * math.cos(angle)

        # Normalise the rotated point by the ellipse axes
        normalised_distance = (rotated_x / a)**2 + (rotated_y / b)**2

        # Check if the point lies within the ellipse
        if normalised_distance > 1:
            return None  # Outside the ellipse

        # Calculate the Gaussian weighting
        d = math.sqrt(normalised_distance)  # Distance from the centre in normalised coordinates
        sigma = math.sqrt(-1 / (2 * math.log(boundary_weighting)))  # Standard deviation for w(1) = 0.6
        weighting = math.exp(-d ** 2 / (2 * sigma ** 2))

        return weighting

    def get_corresponding_spectral_filename(image_filepath, spectra_dir):
        ### Extract the corresponding spectral CSV filename for a given image file based on the pairXXXX index.

        # Extract the filename from the image filepath
        image_filename = os.path.basename(image_filepath)
        
        # Extract the pairXXXX id from the image filename
        pair_index = None
        for part in image_filename.split('_'):
            if part.startswith('pair') and len(part) == 8:  
                pair_index = part
                break
        
        if not pair_index:
            raise ValueError(f"could not find 'pairXXXX' in the image filename: {image_filename}")

        # Search for a spectral file containing the same pairXXXX id
        spectral_filename = None
        for file in os.listdir(spectra_dir):
            if pair_index in file and file.endswith('.csv'):
                spectral_filename = file
                break
        
        if not spectral_filename:
            raise FileNotFoundError(f"No spectral file containing '{pair_index}' found in directory '{spectra_dir}'.")

        # Return just the filename
        return spectral_filename

    # Initialise a list to store rows
    rows = []

    # Precompute extrinsic and intrinsic matrices
    extrinsic_matrices = {}
    intrinsic_matrices = {}

    for _, row in images_data_df.iterrows():

        image_id = row['IMAGE_ID']

        # Compute extrinsic matrix
        qw, qx, qy, qz = row[['QW', 'QX', 'QY', 'QZ']]
        tx, ty, tz = row[['TX', 'TY', 'TZ']]
        extrinsic_matrices[image_id] = get_extrinsic_matrix(qw, qx, qy, qz, tx, ty, tz)
        
        # Compute intrinsic matrix
        # params = ast.literal_eval(cameras_dict[image_id]['PARAMS'])
        # fx, u0, v0 = params[0], params[1], params[2]
        # intrinsic_matrices[image_id] = np.array([
        #     [fx, 0, u0, 0],
        #     [0, fx, v0, 0],
        #     [0, 0, 1, 0]
        # ])

        camera_row = cameras_dict[image_id]  # Directly get the row as a dictionary
        params_str = camera_row['PARAMS']
       
        params_array = ast.literal_eval(params_str)
        
        fx = params_array[0]
        fy = fx
        u0 = params_array[1]
        v0 = params_array[2]

        # Print the values

        # Assemble the intrinsic matrix
        intrinsic_matrices[image_id] = np.array([
            [fx, 0, u0, 0],
            [0, fy, v0, 0],
            [0, 0, 1, 0]
        ])

        if printing:
            print("Image_ID:", image_id)
            print("params_str:", params_str)
            print("params_str type:", type(params_str))
            print("params_array:", params_array)
            print("params_array type:", type(params_array))
            print(f"fx: {fx}, fy: {fy}, u0: {u0}, v0: {v0}")


    ### ITERATE through points3D_df
    # Iterate through all rows in the points3D_df

    # Test first n rows
    # for _, row in points3D_df.head(100).iterrows():

    # # Run across all rows
    # for _, row in points3D_df.iterrows():
    #     point3d_id = row['POINT3D_ID']
    #     x, y, z = row['X'], row['Y'], row['Z']  # raw world coordinates
    #     r, g, b = row['R'], row['G'], row['B']
    #     track = row['TRACK']

    # Extract columns as np arrays
    point3d_ids = points3D_df['POINT3D_ID'].to_numpy()
    coords = points3D_df[['X', 'Y', 'Z']].to_numpy()
    colours = points3D_df[['R', 'G', 'B']].to_numpy()
    points3D_df['TRACK'] = points3D_df['TRACK'].apply(ast.literal_eval)
    tracks = points3D_df['TRACK'].to_numpy()

    # Iterate using np arrays
    for i in range(len(point3d_ids)):
        point3d_id = point3d_ids[i]
        x, y, z = coords[i]
        r, g, b = colours[i]
        track = tracks[i]

        # # Parse the TRACK[] column (if it's a string representation of a list)
        # if isinstance(track, str):
        #     track = ast.literal_eval(track)  # Convert string to list of tuples
        #     print("\n\nTrack:", track)


        # Get scaled world coordinates
        X_w = s_world * x
        Y_w = s_world * y
        Z_w = s_world * z

        if printing:
            # Print POINT3D_ID
            print(f"POINT3D_ID: {point3d_id}")
            print("Raw world coords:", x, y, z)
            print("Scaled world coords:", X_w, Y_w, Z_w)

        point_world = np.array([x, y, z]) #### NOTE Needs to be unscaled for reprojection to work correctly

        # Iterate through TRACK[] tuples to extract IMAGE_IDs
        # Build the new TRACK[] with placeholder values
        new_track = []
        for image_id, point2d_idx in track:

            # print(f"\nIMAGE_ID: {image_id}, POINT2D_IDX: {point2d_idx}")

            # Determine extrinsic matrix
            # images_data_row = images_data_df[images_data_df['IMAGE_ID'] == image_id]
            # qw, qx, qy, qz = images_data_row[['QW', 'QX', 'QY', 'QZ']].iloc[0]
            # print("\nQ:",qw,qx,qy,qz)
            # tx, ty, tz = images_data_row[['TX', 'TY', 'TZ']].iloc[0]
            # print("T:",tx,ty,tz)

            # images_data_row = images_data_dict[image_id]
            # qw, qx, qy, qz = images_data_row['QW'], images_data_row['QX'], images_data_row['QY'], images_data_row['QZ']
            # print("\nQ:", qw, qx, qy, qz)

            # tx, ty, tz = images_data_row['TX'], images_data_row['TY'], images_data_row['TZ']
            # print("T:", tx, ty, tz)

            # extrinsic_matrix = get_extrinsic_matrix(qw,qx,qy,qz,tx,ty,tz)
            # print("Extrinsic Matrix:\n", extrinsic_matrix)

            extrinsic_matrix = extrinsic_matrices[image_id]
            # print("Extrinsic Matrix:\n", extrinsic_matrix)


            # Determine intrinsic matrix
            # camera_row = cameras_df[cameras_df['CAMERA_ID'] == image_id]
            # params_str = camera_row['PARAMS'].iloc[0]

            # camera_row = cameras_dict[image_id]  # Directly get the row as a dictionary
            # params_str = camera_row['PARAMS']
            # print("params_str:", params_str)
            # print("params_str type:", type(params_str))
            # params_array = ast.literal_eval(params_str)
            # print("params_array:", params_array)
            # print("params_array type:", type(params_array))
            # fx = params_array[0]
            # fy = fx
            # u0 = params_array[1]
            # v0 = params_array[2]

            # # Print the values
            # print(f"fx: {fx}, fy: {fy}, u0: {u0}, v0: {v0}")

            # # Assemble the intrinsic matrix
            # intrinsic_matrix = np.array([
            #     [fx, 0, u0, 0],
            #     [0, fy, v0, 0],
            #     [0, 0, 1, 0]
            # ])
            # print("Intrinsic Matrix:\n", intrinsic_matrix)

            intrinsic_matrix = intrinsic_matrices[image_id]
            # print("Intrinsic Matrix:\n", intrinsic_matrix)

            ### Determine pixel coordinates and camera coordinates ###
            (u,v), point_camera = project_point(intrinsic_matrix, extrinsic_matrix, point_world)

            Z_c = s_world * point_camera[2]

            # Get image filepath for img_id
            # images_data_row = images_data_df[images_data_df['IMAGE_ID'] == image_id]
            # img_id_filename = images_data_row['IMAGE_NAME'].iloc[0]
            # # print(img_id_filename)
            # img_id_filepath = f"{images_dir}\{img_id_filename}"
            # # print(img_id_filepath)

            # Get image filename directly from images_data_dict
            img_id_filename = images_data_dict[image_id]['IMAGE_NAME']
            img_id_filepath = f"{images_dir}\\{img_id_filename}"

            if visualisation:
                ### VISUALISATION ###

                # NOTE Show pixel coordinate on image
                img = cv2.imread(img_id_filepath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.scatter([u], [v], color='red', s=50, label=f"Pixel: ({u}, {v})")
                plt.title("Image with pixel marker")
                plt.xlabel("Width (pixels)")
                plt.ylabel("Height (pixels)")
                plt.legend(loc="upper right")
                plt.show()


                # Show corresponding world coordinate in 3D model
                point_cloud = o3d.io.read_point_cloud(colmap_point_cloud_file)
                point_cloud.scale(s_world, center=(0, 0, 0))

                # Create a small sphere at the scaled coordinates for visualisation
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=2)
                marker.translate((X_w, Y_w, Z_w))
                marker.paint_uniform_color([1, 0, 0])  # Red color for the marker

                # Add the marker to the point cloud
                geometries = [point_cloud, marker]

                # NOTE Visualise the point cloud with the marker
                o3d.visualization.draw_geometries(geometries, window_name="Point Cloud with scaled coords")


            # Determine ellipse at depth of point in image
            interpolated_ellipse = interpolate_all_ellipse_parameters(ellipse_params_df, Z_c, interpolant_kind)

            

            if 0 <= u < image_width and 0 <= v < image_height and Z_c_min <= Z_c <= Z_c_max:

                weighting = gaussian_weighting(interpolated_ellipse, u, v, boundary_weighting=gauss_weight)

                if weighting is not None:
                    # if printing:
                    #     print(f"Point ({u}, {v}) lies within the ellipse with a weighting of {weighting:.3f}")
                    pass
                else:
                    # if printing:
                    #     print(f"Point ({u}, {v}) lies outside the ellipse.")
                    continue
            
            else:
                # if printing:
                #     print(f"Point ({u}, {v}) lies outside the image bounds or depth bounds.")
                continue
            

            # Get the spectrum filename that corresponds to the current image
            spectrum_filename = get_corresponding_spectral_filename(img_id_filepath, spectra_dir)

            if printing:
                print("Extrinsic Matrix:\n", extrinsic_matrix)
                print("Intrinsic Matrix:\n", intrinsic_matrix)
                print("(u,v):",u,v)
                print("Point Camera:", point_camera)
                print("Depth:", point_camera[2])
                print("Depth SCALED, Z_c:", Z_c)
                print(img_id_filename)
                print(img_id_filepath)
                print(f"Ellipse parameters at {interpolated_ellipse['Distance']} cm:")
                for param, value in interpolated_ellipse.items():
                    print(f"  {param}: {value:.2f}")
                print(type(interpolated_ellipse))
                print(f"Corresponding spectral filename: {spectrum_filename}")


            # Add the tuple to the new TRACK[]
            # new_track.append((image_id, weighting, img_id_filename, spectrum_filename))
            new_track.append((image_id, weighting, u, v, Z_c, img_id_filename, spectrum_filename))

        # In the loop, append rows as dictionaries
        rows.append({
            'POINT3D_ID': point3d_id,
            'X': x,
            'Y': y,
            'Z': z,
            'X_scaled': X_w,
            'Y_scaled': Y_w,
            'Z_scaled': Z_w,
            'R': r,
            'G': g,
            'B': b,
            'TRACK': new_track
        })

    # After the loop, create the dataframe
    hyperspectral_pc_df = pd.DataFrame(rows)

    # Save dataframe
    hyperspectral_pc_df.to_csv(f"{export_dir}\hyperspectral_pc_df{metadata}.csv", index=False) # NOTE

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return hyperspectral_pc_df, elapsed_time

# elapsed_time = reproject_points(shared_dir, export_dir, colmap_point_cloud_file, ellipse_params_df_file, s_world, Z_c_min, Z_c_max, savgol_w=7, gauss_weight=0.6, metadata="", visualisation=True)
# print(f"Elapsed time: {elapsed_time:.2f} seconds")