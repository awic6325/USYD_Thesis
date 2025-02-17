import os
import pandas as pd
import matplotlib.pyplot as plt

### Interpolate spectral signals according to timestamp

def extract_timestamp_from_filename(filename):
    """
    Extracts a timestamp from a filename.

    For path1: Assume format 'spectrum_pairXXXX_YYYYMMDD_HHMMSS_XXX_savgol.csv'
    For path2: Assume format 'R_spectralon_(before/after)_YYYYMMDD_HHMMSS.csv'

    Return timestamp
    """
    import datetime
    base_name = os.path.basename(filename)
    
    # Check if the filename corresponds to path1 format
    if base_name.startswith("spectrum_pair"):

        # Extract timestamp from 'spectrum_pairXXXX_YYYYMMDD_HHMMSS_XXX_savgol.csv'
        timestamp_str = base_name.split('_')[2] + base_name.split('_')[3]
        return datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    
    # Check if the filename corresponds to path2 format
    elif base_name.startswith("R_spectralon"):

        # Extract timestamp from 'R_spectralon_(before|after)_YYYYMMDD_HHMMSS.csv'
        # timestamp_str = base_name.split('_')[3] + base_name.split('_')[3].split('.')[0]
        timestamp_str = base_name.split('_')[3] + base_name.split('_')[4].split('.')[0]
        return datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    
    else:
        raise ValueError(f"unexpected filename format: {filename}")

def find_spectralon_files(path2):
    
    ### Finds the latest 'R_spectralon_before' and 'R_spectralon_after' files in the given path
    ## Returns the full paths to both files
    
    before_file = None
    after_file = None
    before_timestamp = None
    after_timestamp = None
    
    for filename in os.listdir(path2):

        if filename.startswith("R_spectralon_before"):
            current_timestamp = extract_timestamp_from_filename(filename)
            
            # if current_timestamp > before_timestamp:
            if before_timestamp is None or current_timestamp > before_timestamp:

                before_file = os.path.join(path2, filename)
                before_timestamp = current_timestamp

        elif filename.startswith("R_spectralon_after"):
            current_timestamp = extract_timestamp_from_filename(filename)

            if after_timestamp is None or current_timestamp > after_timestamp:
                after_file = os.path.join(path2, filename)
                after_timestamp = current_timestamp
    
    if before_file is None or after_file is None:

        raise FileNotFoundError("Could not find both 'R_spectralon_before' and 'R_spectralon_after' files")
    
    return before_file, after_file

### Corrects spectra in path1 by dividing with interpolated Spectralon spectra from path2.
def correct_and_plot_spectra(path1, path2):

    # Find the Spectralon files
    spectralon_before_path, spectralon_after_path = find_spectralon_files(path2)
    
    print(f"Using Spectralon before file: {spectralon_before_path}")
    print(f"Using Spectralon after file: {spectralon_after_path}")
    
    # Load Spectralon spectra
    spectralon_before_data = pd.read_csv(spectralon_before_path, header=None)
    spectralon_after_data = pd.read_csv(spectralon_after_path, header=None)
    
    # Assuming the first row contains wavelengths and the second row contains intensities
    wavelengths = spectralon_before_data.iloc[0].astype(float).values  # Same for both before and after
    intensities_before = spectralon_before_data.iloc[1].astype(float).values
    intensities_after = spectralon_after_data.iloc[1].astype(float).values
    
    # Create folders for interpolated Spectralon data, corrected data and plots
    interpolated_folder = f"{path2}/interpolated"
    interpolated_plots_folder = f"{path2}/interpolated_plots"
    corrected_folder = f"{path1}_corrected"
    corrected_plots_folder = f"{path1}_corrected_plots"

    os.makedirs(interpolated_folder, exist_ok=True)
    os.makedirs(interpolated_plots_folder, exist_ok=True)
    os.makedirs(corrected_folder, exist_ok=True)
    os.makedirs(corrected_plots_folder, exist_ok=True)
    
    # Iterate through each CSV file in Path1
    for filename in os.listdir(path1):

        if filename.endswith('.csv'):

            # Load the current spectrum
            spectrum_path = os.path.join(path1, filename)
            spectrum_data = pd.read_csv(spectrum_path, header=None)
            wavelengths_spectrum = spectrum_data.iloc[0].astype(float).values
            intensities_spectrum = spectrum_data.iloc[1].astype(float).values
            
            # Extract timestamp from the current spectrum filename
            timestamp_spectrum = extract_timestamp_from_filename(filename)
            
            # Interpolate the Spectralon intensities based on the timestamp
            weight_before = (extract_timestamp_from_filename(spectralon_after_path) - timestamp_spectrum).total_seconds() / (extract_timestamp_from_filename(spectralon_after_path) - extract_timestamp_from_filename(spectralon_before_path)).total_seconds()
            weight_after = 1 - weight_before
            print("\nWeight before:", weight_before)
            interpolated_intensities = weight_before * intensities_before + weight_after * intensities_after

            # Save the interpolated Spectralon spectrum as CSV
            interpolated_data = pd.DataFrame([wavelengths_spectrum, interpolated_intensities])
            interpolated_file_path = os.path.join(interpolated_folder, filename)
            interpolated_data.to_csv(interpolated_file_path, header=False, index=False)

            # Plot the interpolated Spectralon spectra and save the plot
            plt.figure()
            plt.plot(wavelengths_spectrum, interpolated_intensities, label='Interpolated Spectralon')
            plt.xlabel('Wavelength, $\lambda$ (nm)')
            plt.ylabel('Intensity')
            plt.title(f'Interpolated Spectralon Spectrum: {filename}')
            plt.legend()

            interpolated_plot_path = os.path.join(interpolated_plots_folder, f'{filename[:-4]}_interpolated.png')
            plt.savefig(interpolated_plot_path)
            plt.close()
            
            print(f"Processed {filename}:")
            print(f"- Interpolated data saved to {interpolated_file_path}")
            print(f"- Plot saved to {interpolated_plot_path}")
            
            # Perform the correction
            corrected_intensities = intensities_spectrum / interpolated_intensities
            
            # Save the corrected spectrum as csv file
            corrected_data = pd.DataFrame([wavelengths_spectrum, corrected_intensities])
            corrected_file_path = os.path.join(corrected_folder, filename)
            corrected_data.to_csv(corrected_file_path, header=False, index=False)

            # Filter for plot - spikes less than 350nm
            mask = (wavelengths_spectrum >= 350) & (wavelengths <= 1000)
          
            # Plot the corrected spectrum and save the plot
            plt.figure()
            plt.plot(wavelengths_spectrum[mask], corrected_intensities[mask], label='corrected spectrum')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Reflectance')
            plt.title(f'Corrected Spectrum: {filename}')
            plt.legend()
            corrected_plot_path = os.path.join(corrected_plots_folder, f'{filename[:-4]}_corrected.png')
            plt.savefig(corrected_plot_path)
            plt.close()
            
            print(f"Processed {filename}:")
            print(f"- Corrected data saved to {corrected_file_path}")
            print(f"- Plot saved to {corrected_plot_path}")



########## COLOURED SHEET experiment -- Jan 1st
# # 100
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 100"
# path1 = f"{root_dir}/daq_200ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# # 200
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 200"
# path1 = f"{root_dir}/daq_200ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# # 300
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 300"
# path1 = f"{root_dir}/daq_200ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)



# Reference
# # Pink - w7p2
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session17_ref_pink - PINK w7p2"
# path1 = f"{root_dir}/daq_200ms/spectra_savgol_w7_p2"
# path2 = f"{root_dir}/spectralon_savgol_w7_p2"
# correct_and_plot_spectra(path1, path2)

# # Pink - w7p2
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session17_ref_pink - PINK w9p2"
# path1 = f"{root_dir}/daq_200ms/spectra_savgol_w9_p2"
# path2 = f"{root_dir}/spectralon_savgol_w9_p2"
# correct_and_plot_spectra(path1, path2)

# # Pink - w11p2
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session17_ref_pink - PINK w11p2"
# path1 = f"{root_dir}/daq_200ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# All
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session17_ref_pink"
# path1 = f"{root_dir}/daq_200ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)


# ########## COLOURED SHEET experiment -- Jan 2nd
# # 100
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 100"
# path1 = f"{root_dir}/daq_54ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# # 200
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 200"
# path1 = f"{root_dir}/daq_54ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# # 300
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 300"
# path1 = f"{root_dir}/daq_54ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# Reference
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session4_outdoor_reference"
# path1 = f"{root_dir}/daq_74ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)


# ########## Driveway experiment -- Jan 6th
# # 100
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 100"
# path1 = f"{root_dir}/daq_6ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# # 200
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 200"
# path1 = f"{root_dir}/daq_6ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# # 300
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 300"
# path1 = f"{root_dir}/daq_6ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)


# ########## Ku-ring-gai experiment -- Dec 26th
# # 100
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 100"
# path1 = f"{root_dir}/daq_6ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# # 200
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 200"
# path1 = f"{root_dir}/daq_6ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# # 300
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 300"
# path1 = f"{root_dir}/daq_6ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

# # Reference
# root_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session7_Kuringgai_reference"
# path1 = f"{root_dir}/daq_6ms/spectra_savgol_w11_p2"
# path2 = f"{root_dir}/spectralon_savgol_w11_p2"
# correct_and_plot_spectra(path1, path2)

