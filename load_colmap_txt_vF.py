####### Convert colmap output from .txt to .csv format

import pandas as pd

def load_cameras_txt(filepath):
    # List to hold camera data as dictionaries for each row
    cameras_data = []

    with open(filepath, 'r') as file:
        for line in file:
            elements = line.strip().split()

            # Skip comments and empty lines
            # if len(elements) < 4:
            if len(elements) < 4 or not elements[0].isdigit():
                continue

            # Extract CAMERA_ID, MODEL, WIDTH, HEIGHT, and PARAMS[]
            camera_id = int(elements[0])
            model = elements[1]
            width = int(elements[2])
            height = int(elements[3])
            params = list(map(float, elements[4:]))

            # Append data as a dictionary to the list
            cameras_data.append({
                'CAMERA_ID': camera_id,
                'MODEL': model,
                'WIDTH': width,
                'HEIGHT': height,
                'PARAMS': params  # Stores params as a list, can unpack later if necessary
            })

    # Convert list of dictionaries to a DataFrame
    cameras_df = pd.DataFrame(cameras_data)
    return cameras_df


def load_images_txt(filepath):

    # Separate lists for metadata and points data
    metadata_records = []
    points2D_records = []

    with open(filepath, 'r') as file:
        lines = file.readlines()
        
        for i in range(4, len(lines), 2):  # Start after header,process two lines at a time
            
            # First line for image metadata
            metadata_line = lines[i].strip().split()
            image_id = int(metadata_line[0])
            qw, qx, qy, qz = map(float, metadata_line[1:5])
            tx, ty, tz = map(float, metadata_line[5:8])
            camera_id = int(metadata_line[8])
            image_name = metadata_line[9]
            
            # Append metadata as a single record
            metadata_records.append([image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name])

            # Second line for Points2D
            # points2d_line = lines[i].strip().split()
            points2d_line = lines[i + 1].strip().split()
            for j in range(0, len(points2d_line), 3):

                # Get info
                x = float(points2d_line[j])
                y = float(points2d_line[j + 1])
                point3d_id = int(points2d_line[j + 2])
                points2D_records.append([image_id, x, y, point3d_id])

    # Create DataFrames for image metadata and points data
    df_images = pd.DataFrame(metadata_records, columns=['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'IMAGE_NAME'])
    df_points2D = pd.DataFrame(points2D_records, columns=['IMAGE_ID', 'X', 'Y', 'POINT3D_ID'])

    df_combined = pd.merge(df_images, df_points2D, on='IMAGE_ID')

    return df_images, df_points2D, df_combined


def load_points3D_txt(filepath):
    # Lists to hold data for the dataframe
    point3d_ids, xs, ys, zs, rs, gs, bs, errors, tracks = [], [], [], [], [], [], [], [], []

    with open(filepath, 'r') as file:
        for line in file:
            elements = line.strip().split()

            # Skip comments and empty lines
            if len(elements) < 8 or not elements[0].isdigit():
                continue

            # Parse each line of points3D.txt
            point3d_id = int(elements[0])
            x, y, z = map(float, elements[1:4])
            r, g, b = map(int, elements[4:7])
            error = float(elements[7])

            # Read TRACK[] data as a list of (IMAGE_ID, POINT2D_IDX) tuples
            track = [(int(elements[i]), int(elements[i + 1])) for i in range(8, len(elements), 2)]
            
            # Append data to lists for DF columns
            point3d_ids.append(point3d_id)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            rs.append(r)
            gs.append(g)
            bs.append(b)
            errors.append(error)
            tracks.append(track)

    # Create DF from lists
    points3d_df = pd.DataFrame({
        'POINT3D_ID': point3d_ids,
        'X': xs,
        'Y': ys,
        'Z': zs,
        'R': rs,
        'G': gs,
        'B': bs,
        'ERROR': errors,
        'TRACK': tracks
    })

    return points3d_df



############################ FINAL THESIS DATA ############################
######### EXPERIMENT 1: COLOURED SHEETS #########
##### JAN 1st #####
# ### 100 ###
# # Main directory
# main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 100\colmap_ws\export_txt"

# # FILES
# cameras_file = f"{main_dir}\cameras.txt"
# images_file = f"{main_dir}\images.txt"
# points3D_file = f"{main_dir}\points3D.txt"

# # Load files to get dataframes
# cameras_df = load_cameras_txt(cameras_file)
# images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
# points3D_df = load_points3D_txt(points3D_file)

# # Save dataframes
# cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
# images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
# images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
# points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)

# ### 200 ###
# # Main directory
# main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 200\colmap_ws\export_txt"

# # FILES
# cameras_file = f"{main_dir}\cameras.txt"
# images_file = f"{main_dir}\images.txt"
# points3D_file = f"{main_dir}\points3D.txt"

# # Load files to get dataframes
# cameras_df = load_cameras_txt(cameras_file)
# images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
# points3D_df = load_points3D_txt(points3D_file)

# # Save dataframes
# cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
# images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
# images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
# points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)

# ### 300 ###
# # Main directory
# main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 300\colmap_ws\export_txt"

# # FILES
# cameras_file = f"{main_dir}\cameras.txt"
# images_file = f"{main_dir}\images.txt"
# points3D_file = f"{main_dir}\points3D.txt"

# # Load files to get dataframes
# cameras_df = load_cameras_txt(cameras_file)
# images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
# points3D_df = load_points3D_txt(points3D_file)

# # Save dataframes
# cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
# images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
# images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
# points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)

##### JAN 2nd #####
### 100 ###
# Main directory
main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 100\colmap_ws\export_txt"

# FILES
cameras_file = f"{main_dir}\cameras.txt"
images_file = f"{main_dir}\images.txt"
points3D_file = f"{main_dir}\points3D.txt"

# Load files to get dataframes
cameras_df = load_cameras_txt(cameras_file)
images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
points3D_df = load_points3D_txt(points3D_file)

# Save dataframes
cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)

### 200 ###
# Main directory
main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 200\colmap_ws\export_txt"

# FILES
cameras_file = f"{main_dir}\cameras.txt"
images_file = f"{main_dir}\images.txt"
points3D_file = f"{main_dir}\points3D.txt"

# Load files to get dataframes
cameras_df = load_cameras_txt(cameras_file)
images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
points3D_df = load_points3D_txt(points3D_file)

# Save dataframes
cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)

### 300 ###
# Main directory
main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 300\colmap_ws\export_txt"

# FILES
cameras_file = f"{main_dir}\cameras.txt"
images_file = f"{main_dir}\images.txt"
points3D_file = f"{main_dir}\points3D.txt"

# Load files to get dataframes
cameras_df = load_cameras_txt(cameras_file)
images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
points3D_df = load_points3D_txt(points3D_file)

# Save dataframes
cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)


######### EXPERIMENT 2: DRIVEWAY #########
##### JAN 6th #####
### 100 ###
# Main directory
main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 100\colmap_ws\export_txt"

# FILES
cameras_file = f"{main_dir}\cameras.txt"
images_file = f"{main_dir}\images.txt"
points3D_file = f"{main_dir}\points3D.txt"

# Load files to get dataframes
cameras_df = load_cameras_txt(cameras_file)
images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
points3D_df = load_points3D_txt(points3D_file)

# Save dataframes
cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)

### 200 ###
# Main directory
main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 200\colmap_ws\export_txt"

# FILES
cameras_file = f"{main_dir}\cameras.txt"
images_file = f"{main_dir}\images.txt"
points3D_file = f"{main_dir}\points3D.txt"

# Load files to get dataframes
cameras_df = load_cameras_txt(cameras_file)
images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
points3D_df = load_points3D_txt(points3D_file)

# Save dataframes
cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)

### 300 ###
# Main directory
main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 300\colmap_ws\export_txt"

# FILES
cameras_file = f"{main_dir}\cameras.txt"
images_file = f"{main_dir}\images.txt"
points3D_file = f"{main_dir}\points3D.txt"

# Load files to get dataframes
cameras_df = load_cameras_txt(cameras_file)
images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
points3D_df = load_points3D_txt(points3D_file)

# Save dataframes
cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)


######### EXPERIMENT 3: KU-RING-GAI #########
##### DEC 26th #####
### 100 ###
# Main directory
main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 100\colmap_ws\export_txt"

# FILES
cameras_file = f"{main_dir}\cameras.txt"
images_file = f"{main_dir}\images.txt"
points3D_file = f"{main_dir}\points3D.txt"

# Load files to get dataframes
cameras_df = load_cameras_txt(cameras_file)
images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
points3D_df = load_points3D_txt(points3D_file)

# Save dataframes
cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)

### 200 ###
# Main directory
main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 200\colmap_ws\export_txt"

# FILES
cameras_file = f"{main_dir}\cameras.txt"
images_file = f"{main_dir}\images.txt"
points3D_file = f"{main_dir}\points3D.txt"

# Load files to get dataframes
cameras_df = load_cameras_txt(cameras_file)
images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
points3D_df = load_points3D_txt(points3D_file)

# Save dataframes
cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)

### 300 ###
# Main directory
main_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 300\colmap_ws\export_txt"

# FILES
cameras_file = f"{main_dir}\cameras.txt"
images_file = f"{main_dir}\images.txt"
points3D_file = f"{main_dir}\points3D.txt"

# Load files to get dataframes
cameras_df = load_cameras_txt(cameras_file)
images_data_df, images_points2D_df, images_df2 = load_images_txt(images_file)
points3D_df = load_points3D_txt(points3D_file)

# Save dataframes
cameras_df.to_csv(f"{main_dir}\cameras_df.csv", index=False)
images_data_df.to_csv(f"{main_dir}\images_data_df.csv", index=False)
images_points2D_df.to_csv(f"{main_dir}\images_points2D_df.csv", index=False)
points3D_df.to_csv(f"{main_dir}\points3D_df.csv", index=False)