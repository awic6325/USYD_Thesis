import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time

### NOTE: Most efficient approach is to make a new column called WEIGHTED_SPECTRUM_FILENAME
#         - Store the weighted spectrum in a new file, rather than in the dataframe itself
#         - The dataframe will instead store the filename

def compute_and_store_weighted_spectra(hyperspectral_pc_df, spectra_dir, output_dir, plot_dir):
    """
    Compute the weighted spectrum for each row in the hyperspectral_pc_df dataframe, save
    it as a CSV file in the designated output directory, and save plot of the
    spectrum in the plot directory. Update the dataframe to include filenames of
    the saved spectra.

    Parameters:
        hyperspectral_pc_df: pd.DataFrame containing POINT3D_ID, TRACK, other fields
        spectra_dir: Directory containing input spectrum files
        output_dir: Directory to save weighted spectrum files
        plot_dir: Directory to save spectrum plot files

    Return:
        Modified pd.DataFrame with new WEIGHTED_SPECTRUM_FILENAME and PLOT_FILENAME columns
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    os.makedirs(plot_dir, exist_ok=True)  # Ensure the plot directory exists

    def calculate_weighted_spectrum(track, spectra_dir):
        """
        Calculate the weighted spectrum for a single TRACK entry.

        Parameters:
            track (list): List of tuples in the format (IMAGE_ID, WEIGHTING, U, V, Z_C, IMAGE_FILENAME, SPECTRUM_FILENAME).
        """
        if not track:
            return None  # Empty TRACK means no weighted spectrum

        total_weight = 0
        weighted_spectrum = None

        for _, weighting, _, _, _, _, spectrum_filename in track:
            spectrum_filepath = os.path.join(spectra_dir, spectrum_filename)
            try:
                spectrum_data = np.genfromtxt(spectrum_filepath, delimiter=',')
            except FileNotFoundError:
                print(f"Warning: File {spectrum_filepath} not found, skipping.")
                continue

            if weighted_spectrum is None:
                # Initialise weighted spectrum with the same shape as the first spectrum
                weighted_spectrum = np.zeros_like(spectrum_data)

            # Accumulate weighted intensities
            weighted_spectrum[0] = spectrum_data[0]  # Wavelengths remain the same
            weighted_spectrum[1] += weighting * spectrum_data[1]
            total_weight += weighting

        if total_weight > 0:
            # Normalise intensities by the total weight
            weighted_spectrum[1] /= total_weight

        return weighted_spectrum

    ### Save the weighted spectrum to a CSV file
    def save_weighted_spectrum(spectrum, point3d_id):

        if spectrum is None:
            return None

        # file_name = f"weighted_spectrum_{point3d_id}.csv"
        # file_path = os.path.join(output_dir, file_name)
        # with open(file_path, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Wavelength", "Intensity"])
        #     writer.writerows(zip(spectrum[0], spectrum[1]))
        # return file_path

        file_name = f"weighted_spectrum_{point3d_id}.csv"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(spectrum[0]))
            writer.writerow(list(spectrum[1]))
        return file_path

    def save_spectrum_plot(spectrum, point3d_id):

        if spectrum is None:
            return None

        file_name = f"weighted_spectrum_plot_{point3d_id}.png"
        file_path = os.path.join(plot_dir, file_name)

        wavelengths = spectrum[0]
        intensities = spectrum[1]
        
        mask = (wavelengths >= 350) & (wavelengths <= 1000)

        plt.figure()
        plt.plot(wavelengths[mask], intensities[mask], label=f"Point3D ID: {point3d_id}")
        plt.xlabel("Wavelength")
        plt.ylabel("Intensity")
        plt.title(f"Weighted Spectrum for Point3D ID {point3d_id}")
        plt.legend()
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        return file_path

    # Iterate over rows in the DataFrame
    spectrum_filenames = []
    plot_filenames = []

    for _, row in hyperspectral_pc_df.iterrows():
        point3d_id = row['POINT3D_ID']
        track = row['TRACK']
        weighted_spectrum = calculate_weighted_spectrum(track, spectra_dir)
        spectrum_filename = save_weighted_spectrum(weighted_spectrum, point3d_id)
        plot_filename = save_spectrum_plot(weighted_spectrum, point3d_id)

        spectrum_filenames.append(spectrum_filename)
        plot_filenames.append(plot_filename)

    # Add the new columns to the DataFrame
    hyperspectral_pc_df['WEIGHTED_SPECTRUM_FILENAME'] = spectrum_filenames
    hyperspectral_pc_df['PLOT_FILENAME'] = plot_filenames

    count_files = len(list(filter(lambda x: x is not None, spectrum_filenames)))
    count_plots = len(list(filter(lambda x: x is not None, plot_filenames)))
    count_hpc = len(spectrum_filenames)
    print(f"No. of weighted spectra filenames: {count_files}")  # Output
    print(f"No. of weighted spectra plots: {count_plots}")  # Output
    print(f"Total length: {count_hpc}")  # Output
    print(f"HPC return: {count_files/count_hpc*100} %")

    return hyperspectral_pc_df

# Start the timer
start_time = time.time()

# Params
# kind = "cubic"

# coreg = "S_xy"
# coreg = "A_xy"
# 
# exp = "exp1_jan1"
# exp = "exp1_jan2"
# exp = "exp2"
# exp = "exp3"

# n_daq = 100
# n_daq = 200
# n_daq = 300

# kind_array = ["cubic"]
# coreg_array = ["S_xy", "A_xy"]
# exp_array = ["exp1_jan1", "exp1_jan2", "exp2", "exp3"]
# n_daq_array = [100, 200, 300]

kind_array = ["cubic"]
coreg_array = ["S_xy"]
exp_array = ["exp1_jan1"]
n_daq_array = [100]

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


                # Undistorted images
                images_dir = f"{shared_dir}/images_undistorted"

                # Savitzky-Golay filtered REFLECTANCE spectra i.e. savgol experimental spectra DIVIDED by savgol Spectralon spectrum
                # spectra_dir = f"{shared_dir}\spectra_savgol_w7_p2_corrected"
                spectra_dir = f"{shared_dir}/spectra_savgol_w11_p2_corrected"

                # hyperspectral_pc_df_filepath = f"{shared_dir}\hyperspectral_pc_df.csv"
                hyperspectral_pc_df_filepath = f"{export_dir}/hyperspectral_pc_df_{coreg}_{kind}_ENU.csv"

                # Designate a directory to store the weighted_spectra and plots
                weighted_spectra_dir = f"{shared_dir}/weighted_spectra_{coreg}_{kind}"
                weighted_spectra_plots_dir = f"{shared_dir}/weighted_spectra_plots_{coreg}_{kind}"

                # Read in hyperspectral pc
                hyperspectral_pc_df = pd.read_csv(hyperspectral_pc_df_filepath)

                # Ensure TRACK[] is interpreted as a list if stored as a string
                # Use eval() to safely evaluate list-like strings
                hyperspectral_pc_df['TRACK'] = hyperspectral_pc_df['TRACK'].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                )

                # Count non-empty entries in the TRACK[] column
                non_empty_count = hyperspectral_pc_df['TRACK'].apply(bool).sum()
                print(f"Number of non-empty TRACK entries: {non_empty_count}")

                # Determine weighted sums
                modified_df = compute_and_store_weighted_spectra(hyperspectral_pc_df, spectra_dir, weighted_spectra_dir, weighted_spectra_plots_dir)
                modified_df.to_csv(f"{export_dir}/hyperspectral_pc_df_{coreg}_{kind}_ENU_weighted.csv", index=False)


                # Stop the timer
                end_time = time.time()

                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
