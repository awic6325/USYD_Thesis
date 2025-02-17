import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime

def append_timestamp_to_filenames(folder_path):

    ### Modifies filenames in the specified folder by appending the 'Date modified' timestamp
    ### in YYYYMMDD_HHMMSS format to the end of the filename.
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Get the last modified time and format it
        timestamp = os.path.getmtime(file_path)
        formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        
        # Split the filename into name and extension
        name, extension = os.path.splitext(filename)
        
        # Create the new filename
        new_filename = f"{name}_{formatted_time}{extension}"
        
        # Rename the file
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_filename}'")


# def process_spectral_data(root_dir, window_length=7, polyorder=2):
def process_spectral_data(root_dir, window_length=11, polyorder=2):
    # Traverse each subdirectory in the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter for .csv files in the current directory
        csv_files = [f for f in filenames if f.lower().endswith('.csv')]
        
        if csv_files:
            # Determine the parent directory to place the adjacent folders
            parent_dir = os.path.dirname(dirpath)

            # Create adjacent folders for outputs with Savitzky-Golay parameters in the name
            spectra_plots_dir = f"{dirpath}_plots"
            spectra_savgol_dir = f"{dirpath}_savgol_w{window_length}_p{polyorder}"
            spectra_savgol_plots_dir = f"{dirpath}_savgol_plots_w{window_length}_p{polyorder}"
            
            # Ensure directories exist
            os.makedirs(spectra_plots_dir, exist_ok=True)
            os.makedirs(spectra_savgol_dir, exist_ok=True)
            os.makedirs(spectra_savgol_plots_dir, exist_ok=True)
            
            # Process each .csv file
            for csv_file in csv_files:
                csv_path = os.path.join(dirpath, csv_file)

                print(f"\nPath: {csv_path}")
                
                # Read CSV without headers and extract wavelengths and intensities
                data = pd.read_csv(csv_path, header=None)
                if len(data) < 2:
                    print(f"Error: {csv_file} does not have enough rows for wavelengths/intensities")
                    continue
                
                # Get data
                wavelengths = data.iloc[0].values
                intensities = data.iloc[1].values
                
                # Plot and save raw spectra
                plt.figure()
                plt.plot(wavelengths, intensities, label='Raw Spectrum')
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('Intensity')
                plt.title(f'Raw Spectrum: {csv_file}')
                plt.legend()
                raw_plot_path = os.path.join(spectra_plots_dir, f'{csv_file[:-4]}_raw.png') # create path
                plt.savefig(raw_plot_path)
                plt.close()
                
                # Apply Savitzky-Golay filter
                filtered_intensities = savgol_filter(intensities, window_length=window_length, polyorder=polyorder)
                
                # # Save filtered spectra as CSV
                # filtered_data = pd.DataFrame({
                #     'Wavelength': wavelengths,
                #     'Filtered Intensity': filtered_intensities
                # })

                # Create a DataFrame with wavelengths and filtered intensities as rows
                filtered_data = pd.DataFrame([wavelengths, filtered_intensities])              

                filtered_csv_path = os.path.join(spectra_savgol_dir, f'{csv_file[:-4]}_savgol.csv')
                # filtered_data.to_csv(filtered_csv_path, index=False)

                # Save filtered spectra as CSV with no headers and no index
                filtered_data.to_csv(filtered_csv_path, header=False, index=False)
                
                # Plot and save filtered spectra
                plt.figure()
                plt.plot(wavelengths, filtered_intensities, label='Savitzky-Golay Filtered Spectrum', color='orange')
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('Intensity')
                plt.title(f'Filtered Spectrum: {csv_file}')
                plt.legend()
                savgol_plot_path = os.path.join(spectra_savgol_plots_dir, f'{csv_file[:-4]}_savgol.png')
                plt.savefig(savgol_plot_path)
                plt.close()
                


######### KU-RING-GAI experiment
# Reference
# root_dir = r"E:\Thesis\project_v3\device_data\20241226_session7_Kuringgai_reference"
# daq_dir = f"{root_dir}/daq_6ms"
# spec_dir = f"{root_dir}/spectralon"
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# Experiment
# root_dir = r"E:\Thesis\project_v3\device_data\20241226_session6_kuringgai_500"
# daq_dir = f"{root_dir}/daq_6ms"
# spec_dir = f"{root_dir}/spectralon"
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)


########## COLOURED SHEET experiment -- Jan 1st
# 100
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 100"
# daq_dir = f"{root_dir}/daq_200ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# 200
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 200"
# daq_dir = f"{root_dir}/daq_200ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# # 300
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 300"
# daq_dir = f"{root_dir}/daq_200ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)


# Reference
# # Pink - w7p2
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session17_ref_pink - PINK w7p2"
# daq_dir = f"{root_dir}/daq_200ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir, window_length=7, polyorder=2)
# process_spectral_data(spec_dir, window_length=7, polyorder=2)

# # Pink - w7p2
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session17_ref_pink - PINK w9p2"
# daq_dir = f"{root_dir}/daq_200ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir, window_length=9, polyorder=2)
# process_spectral_data(spec_dir, window_length=9, polyorder=2)

# Pink - w11p2
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session17_ref_pink - PINK w11p2"
# daq_dir = f"{root_dir}/daq_200ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir, window_length=11, polyorder=2)
# process_spectral_data(spec_dir, window_length=11, polyorder=2)

# ALL
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session17_ref_pink"
# daq_dir = f"{root_dir}/daq_200ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir, window_length=11, polyorder=2)
# process_spectral_data(spec_dir, window_length=11, polyorder=2)


########## COLOURED SHEET experiment -- Jan 2nd
# Test
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - testw11p2"
# daq_dir = f"{root_dir}/daq_54ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# # 100
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 100"
# daq_dir = f"{root_dir}/daq_54ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# # 200
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 200"
# daq_dir = f"{root_dir}/daq_54ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# # 300
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 300"
# daq_dir = f"{root_dir}/daq_54ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# Reference
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session4_outdoor_reference"
# daq_dir = f"{root_dir}/daq_74ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)


########## Driveway experiment -- Jan 6th
# # 100
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 100"
# daq_dir = f"{root_dir}/daq_6ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# # 200
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 200"
# daq_dir = f"{root_dir}/daq_6ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# # 300
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 300"
# daq_dir = f"{root_dir}/daq_6ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

########## Ku-ring-gai experiment -- Dec 26th
# # 100
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 100"
# daq_dir = f"{root_dir}/daq_6ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# # 200
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 200"
# daq_dir = f"{root_dir}/daq_6ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# # 300
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 300"
# daq_dir = f"{root_dir}/daq_6ms"
# spec_dir = f"{root_dir}/spectralon"
# append_timestamp_to_filenames(spec_dir)
# process_spectral_data(daq_dir)
# process_spectral_data(spec_dir)

# Reference
root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session7_Kuringgai_reference"
daq_dir = f"{root_dir}/daq_6ms"
spec_dir = f"{root_dir}/spectralon"
append_timestamp_to_filenames(spec_dir)
process_spectral_data(daq_dir)
process_spectral_data(spec_dir)
