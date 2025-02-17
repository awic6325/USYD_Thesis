# Example of determining hyperspectral 3D map, for exp1A-i (S_xy) dataset

######### EXPERIMENT DETAILS #########
#
# Experiment 1:     Coloured Sheets
# Date:             Jan 1st
# No. DAQ Pairs:    100
# Ellipse:          A_xy
# Interpolant:      Cubic Spline

import sys
import os

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import the module from the parent directory
from reproject_points_BEST_noprint import reproject_points
from count_hyperpc_track import average_tuples_in_track

# Parameters to vary
coreg = "S_xy"
# coreg = "A_xy"

n_daq = 100
# n_daq = 200
# n_daq = 300

# Interpolant kind
kind = 'cubic'

# Thresholds
Z_c_min = 15
Z_c_max = 180

# Vary according to type of spatial co-registration
if coreg == "S_xy":

    # Metadata
    type = f"_{coreg}_{kind}"

    # Ellipse parameters
    ellipse_params_df_file = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Spatial_Coregistration\spatial_coregistration\ellipse_df\S_xy_ellipse_parameters_BEST.csv"


elif coreg == "A_xy":

    # Metadata
    type = f"_{coreg}_{kind}"

    # Ellipse parameters
    ellipse_params_df_file = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Spatial_Coregistration\spatial_coregistration\ellipse_df\A_xy_ellipse_parameters_BEST.csv"


# Vary according to number of image-spectrum pairs
if n_daq == 100:

    # World scaling factor
    s_world = 9.139701300942097

    # Experiment Data
    shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 100\daq_200ms"
    export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 100\colmap_ws\export_txt"
    colmap_point_cloud_file = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 100\colmap_ws\dense\1\meshed-poisson.ply"

elif n_daq == 200:

    # World scaling factor
    s_world = 9.505465234063715

    # Experiment Data
    shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 200\daq_200ms"
    export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 200\colmap_ws\export_txt"
    colmap_point_cloud_file = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 200\colmap_ws\dense\0\meshed-poisson.ply"

elif n_daq == 300:

    # World scaling factor
    s_world = 9.65213356275381 

    # Experiment Data
    shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 300\daq_200ms"
    export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 300\colmap_ws\export_txt"
    colmap_point_cloud_file = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 300\colmap_ws\dense\0\meshed-poisson.ply"


# Construct hyperspectral point cloud
hyperspectral_pc_df, elapsed_time = reproject_points(shared_dir, export_dir, colmap_point_cloud_file, ellipse_params_df_file, s_world, Z_c_min, Z_c_max, savgol_w=11, gauss_weight=0.6, interpolant_kind=kind, metadata=type, visualisation=False, printing=False)
print(f"Elapsed time: {elapsed_time:.2f} seconds")

average_tuples = average_tuples_in_track(hyperspectral_pc_df)
print(f"Average number of tuples in non-empty TRACK lists: {average_tuples}")


# Save info to txt file
txt_filename = f"hpc_info_{n_daq}_{coreg}_{kind}.txt"
txt_filepath = f"{export_dir}/{txt_filename}"
with open(txt_filepath, "w") as file:
    file.write(f"# DAQ: {n_daq} \
               \nCoreg: {coreg}\n \
               \nInterpolant kind: {kind} \
               \n \
               \nElapsed time: {elapsed_time:.2f} seconds. \
               \nAverage number of tuples in non-empty TRACK lists: {average_tuples}")