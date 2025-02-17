import pandas as pd
import numpy as np
from scipy.spatial import distance
import os
import pysolar.solar as pss
import pytz
import datetime as dt
import math
import numpy as np
import datetime as dt
import re
import matplotlib.pyplot as plt
import csv
import time


def compute_and_store_surface_spectra(hyperspectral_pc_df, latitude, longitude, raw_spectra_dir, output_dir, plot_dir):

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    os.makedirs(plot_dir, exist_ok=True)  # Ensure the plot directory exists

    ### Solar information
    # Inputs: raw_spectra_dir, latitude, longitude
    # Outputs: altitude, azimuth, L

    ### Extracts the timestamp from a filename in the given format
    def extract_timestamp_from_filename(filename):

        # match = re.search(r"_(\d{8}_\d{6}_\d{3})\.png$", filename)    # image dir
        match = re.search(r"_(\d{8}_\d{6}_\d{3})\.csv$", filename)      # spectra dir
        if match:
            date_time_str = match.group(1)  # Extract the matched timestamp string
            return dt.datetime.strptime(date_time_str, "%Y%m%d_%H%M%S_%f")
        return None
    
    ### Finds the mean timestamp from filenames in a given dir
    def find_mean_timestamp_from_filenames(directory_path):

        timestamps = []

        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.csv'):
                timestamp = extract_timestamp_from_filename(filename)
                if timestamp:
                    timestamps.append(timestamp)

        if not timestamps:
            raise ValueError("No valid timestamps found in the filenames")

        # Compute the mean timestamp
        mean_timestamp = sum((ts.timestamp() for ts in timestamps)) / len(timestamps)
        mean_datetime = dt.datetime.fromtimestamp(mean_timestamp)
        
        # Determine naive time
        date_time_str = mean_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
        date_time_naive = dt.datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S.%f")
        
        return date_time_naive
    
    ### Apply spherical transformation
    def get_sun_direction(azimuth, elevation):

        L1 = math.cos(elevation) * math.cos(azimuth)
        L2 = math.cos(elevation) * math.sin(azimuth)
        L3 = math.sin(elevation)

        L = np.array([L1, L2, L3])
        L_norm = L / np.linalg.norm(L)

        return L_norm

    ### Find neighbours
    # Function to find neighbours within a given radius - 2.5cm
    def find_neighbours_within_radius(df, row_index, radius=0.025):

        selected_point = df.loc[row_index, ['X_ENU', 'Y_ENU', 'Z_ENU']].values
        all_points = df[['X_ENU', 'Y_ENU', 'Z_ENU']].values
        
        distances = np.linalg.norm(all_points - selected_point, axis=1)
        neighbours = all_points[distances <= radius]

        return neighbours

    ### Fit plane
    # Function to fit a plane using least squares
    def fit_plane(points):

        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]
        
        # Create a matrix for least squares
        A = np.c_[X, Y, np.ones_like(X)]
        C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)  # Solve for plane coefficients
        
        # Normal vector of the plane: (-A, -B, 1) normalised, solves plane equation
        normal = np.array([-C[0], -C[1], 1.0])
        normal /= np.linalg.norm(normal)  # Normalise
        return normal

    # Save surface spectrum
    def save_surface_reflectance(spectrum, point3d_id):

        if spectrum is None:
            return None

        file_name = f"surface_reflectance_{point3d_id}.csv"
        file_path = os.path.join(output_dir, file_name)

        # Write to file
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(spectrum[0])
            writer.writerow(spectrum[1])

        return file_path

    # Plot surface spectrum
    def save_spectrum_plot(spectrum, point3d_id):

        if spectrum is None:
            return None

        file_name = f"surface_reflectance_plot_{point3d_id}.png"
        file_path = os.path.join(plot_dir, file_name)

        wavelengths = spectrum[0]
        intensities = spectrum[1]
        
        mask = (wavelengths >= 350) & (wavelengths <= 1000)

        plt.figure()
        plt.plot(wavelengths[mask], 100*intensities[mask], label=f"Point3D ID: {point3d_id}")
        plt.xlabel("Wavelength")
        plt.ylabel("Reflectance (%)")
        plt.title(f"Surface Reflectance for Point3D ID {point3d_id}")
        plt.legend()
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        return file_path

    # Get points - convert to numeric, drop NaN values
    coords_ENU = hyperspectral_pc_df[['X_ENU', 'Y_ENU', 'Z_ENU']].apply(pd.to_numeric, errors='coerce').dropna()

    # Columns to append to hyperspectral_pc_df
    solar_info_vals = []
    L_Sun_vals = []
    N_obs_vals = []
    N_spec_vals = []
    corr_factor_vals = []
    surface_spectra_filenames = []
    surface_plot_filenames = []

    # Find mean timestamp from raw spectra directory
    mean_timestamp = find_mean_timestamp_from_filenames(raw_spectra_dir)
    print("Mean Timestamp:", mean_timestamp)

    # Convert to Sydney timezone
    sydney = pytz.timezone('Australia/Sydney')
    date_time = sydney.localize(mean_timestamp)
    print(f"Sydney datetime: {date_time}")

    # Get solar angles
    altitude = pss.get_altitude(latitude, longitude, date_time)  # Solar elevation angle
    azimuth = pss.get_azimuth(latitude, longitude, date_time)  # Solar azimuth angle

    print(f"Solar Elevation: {altitude} deg, Solar Azimuth: {azimuth} deg")

    # Convert solar azimuth and elevation to ENU vector
    alt_rad = math.radians(altitude)
    az_rad = math.radians(azimuth)

    print(f"alt_rad: {alt_rad}, az_rad: {az_rad}")

    # Get normalised Sun vector
    L_norm = get_sun_direction(az_rad, alt_rad)
    print(f"L_Sun: {L_norm}")

    # Assume normal vector of Spectralon is perfectly vertical
    N_spec = np.array([0, 0, 1])
    print(f"N_spec: {N_spec}")

    ### Loop through all rows of hyperspectral_pc_df
    for index, row in hyperspectral_pc_df.iterrows():

        # print(f"\n~~~~~~ Index: {index} ~~~~~~\n")
        # print(f"\n~~~~~~ Row: \n{row} ~~~~~~\n")
        
        # Check if it's a hyperspectral point
        if pd.isna(row['WEIGHTED_SPECTRUM_FILENAME']) or row['WEIGHTED_SPECTRUM_FILENAME'] == '':

            # Append None and continue
            solar_info_vals.append(None)
            L_Sun_vals.append(None)
            N_obs_vals.append(None)
            N_spec_vals.append(None)
            corr_factor_vals.append(None)
            surface_spectra_filenames.append(None)
            surface_plot_filenames.append(None)

            continue

        else:

            # Find neighbours within a 2.5 cm radius
            neighbours = find_neighbours_within_radius(coords_ENU, index)

            # Fit a plane to the neighbours and get the normal vector
            N_obs = fit_plane(neighbours)
            # print(f"N_obs: {N_obs}")

            # Determine correction factor
            corr_factor = np.dot(N_spec, L_norm) / np.dot(N_obs, L_norm)
            # print(f"Correction factor: {corr_factor}")

            # Get current weighted spectrum
            weighted_spectrum_filename = row['WEIGHTED_SPECTRUM_FILENAME']
            weighted_spectrum = np.genfromtxt(weighted_spectrum_filename, delimiter=',')
            # print(weighted_spectrum)
            # print(weighted_spectrum[0])
            # print(weighted_spectrum[1])

            # Determine surface reflectance
            surface_reflectance = np.zeros_like(weighted_spectrum)
            surface_reflectance[0] = weighted_spectrum[0]
            surface_reflectance[1] = corr_factor * weighted_spectrum[1]
            # print(surface_reflectance)

            # Current 3D point
            point3D_ID = row['POINT3D_ID']
            # print(f"Point3D ID: {point3D_ID}")

            # Save and plot surface reflectance
            surface_reflectance_filename = save_surface_reflectance(surface_reflectance, point3D_ID)
            surface_plot_filename = save_spectrum_plot(surface_reflectance, point3D_ID)

            # Append
            solar_info_vals.append([latitude, longitude, date_time, altitude, azimuth])
            L_Sun_vals.append(L_norm)
            N_obs_vals.append(N_obs)
            N_spec_vals.append(N_spec)
            corr_factor_vals.append(corr_factor)
            surface_spectra_filenames.append(surface_reflectance_filename)
            surface_plot_filenames.append(surface_plot_filename)

    # Add columns to dataframe
    hyperspectral_pc_df['SOLAR_INFO'] = solar_info_vals
    hyperspectral_pc_df['L_Sun'] = L_Sun_vals
    hyperspectral_pc_df['N_obs'] = N_obs_vals
    hyperspectral_pc_df['N_spec'] = N_spec_vals
    hyperspectral_pc_df['Correction Factor'] = corr_factor_vals
    hyperspectral_pc_df['Surface Reflectance File'] = surface_spectra_filenames
    hyperspectral_pc_df['Surface Reflectance Plot'] = surface_plot_filenames

    return hyperspectral_pc_df



kind_array = ["cubic"]
coreg_array = ["S_xy", "A_xy"]
exp_array = ["exp1_jan1", "exp1_jan2", "exp2", "exp3"]
n_daq_array = [100, 200, 300]

# kind_array = ["cubic"]
# coreg_array = ["S_xy"]
# exp_array = ["exp1_jan2"]
# n_daq_array = [300]


for i in range(len(kind_array)):
    for j in range(len(coreg_array)):
        for k in range(len(exp_array)):
            for l in range(len(n_daq_array)):

                kind = kind_array[i]
                coreg = coreg_array[j]
                exp = exp_array[k]
                n_daq = n_daq_array[l]

                print(f"\nKind: {kind}\nCoreg: {coreg}\nExperiment: {exp}\nNo. Daq: {n_daq}\n")

                if exp == "exp1_jan1":
                    # GPS coordinates
                    latitude = -33
                    longitude = 150

                    ### Exp 1: Jan 1st
                    if n_daq == 100:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 100\daq_200ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 100\colmap_ws\export_txt"
                    elif n_daq == 200:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 200\daq_200ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 200\colmap_ws\export_txt"
                    elif n_daq == 300:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 300\daq_200ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 300\colmap_ws\export_txt"

                elif exp == "exp1_jan2":
                    # GPS coordinates
                    latitude = -33
                    longitude = 150

                    ## Exp 1: Jan 2nd
                    if n_daq == 100:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 100\daq_54ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 100\colmap_ws\export_txt"
                    elif n_daq == 200:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 200\daq_54ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 200\colmap_ws\export_txt"
                    elif n_daq == 300:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 300\daq_54ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 300\colmap_ws\export_txt"
                    # pass

                elif exp == "exp2":
                    # GPS coordinates
                    latitude = -33
                    longitude = 150

                    ## Exp 2
                    if n_daq == 100:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 100\daq_6ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 100\colmap_ws\export_txt"
                    elif n_daq == 200:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 200\daq_6ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 200\colmap_ws\export_txt"
                    elif n_daq == 300:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 300\daq_6ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 300\colmap_ws\export_txt"
                    # pass

                elif exp == "exp3":
                    # Bobbinhead coordinates:
                    latitude = -33.6591
                    longitude = 151.1599

                    ## Exp 3
                    if n_daq == 100:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 100\daq_6ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 100\colmap_ws\export_txt"
                    elif n_daq == 200:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 200\daq_6ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 200\colmap_ws\export_txt"
                    elif n_daq == 300:
                        shared_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 300\daq_6ms"
                        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 300\colmap_ws\export_txt"
                    # pass


                # Get the hyperspectral point cloud
                hpc_df_path = f"{export_dir}/hyperspectral_pc_df_{coreg}_{kind}_ENU_weighted.csv"
                hpc_df = pd.read_csv(hpc_df_path)
                
                # Subdirectories
                raw_spectra_dir = f"{shared_dir}/spectra"
                weighted_spectra_dir = f"{shared_dir}/weighted_spectra_{coreg}_{kind}"
                output_dir = f"{shared_dir}/surface_reflectance_{coreg}_{kind}"
                plot_dir = f"{shared_dir}/surface_reflectance_plots_{coreg}_{kind}"

                # Start timer
                start_time = time.time()

                hyperspectral_pc_df_surface = compute_and_store_surface_spectra(hpc_df, latitude, longitude, raw_spectra_dir, output_dir, plot_dir)
                hyperspectral_pc_df_surface.to_csv(f"{export_dir}/hyperspectral_pc_df_{coreg}_{kind}_ENU_weighted_SURFACE.csv", index=False)

                # Stop the timer
                end_time = time.time()

                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time:.2f} seconds")



