import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

def average_tuples_in_track(hyperspectral_pc_df):

    ### Calculate average number of tuples in non-empty lists, stored in the TRACK[] column
    ### Parameters:
    #    hyperspectral_pc_df - pd.DataFrame containing the TRACK column

    hyperspectral_pc_df['TRACK'] = hyperspectral_pc_df['TRACK'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Filter non-empty TRACK entries
    non_empty_tracks = hyperspectral_pc_df['TRACK'].dropna()
    non_empty_tracks = [track for track in non_empty_tracks if isinstance(track, list) and len(track) > 0]

    # print(non_empty_tracks)

    # Ensure TRACK entries are lists and calculate their lengths
    lengths = [len(track) for track in non_empty_tracks if isinstance(track, list)]
    # print(lengths)

    print(f"Min: {min(lengths)}")
    print(f"Max: {max(lengths)}")
    print(f"Mean: {np.mean(lengths)}")

    print(lengths)

    # Convert lengths to numpy array and find unique values and counts
    unique_values, counts = np.unique(lengths, return_counts=True)

    # Convert to a dictionary
    length_frequency = dict(zip(unique_values, counts))
    print("Length Frequencies:", length_frequency)

    total_frequency = sum(counts)
    percentages = [(freq / total_frequency) * 100 for freq in counts]

    # Display histogram
    plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), edgecolor='black', alpha=0.75)
    plt.title('Distribution of lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Calculate and return the average
    if lengths:
        return np.mean(lengths)
    else:
        return 0.0  # Return 0 if no non-empty lists


hyperspectral_pc_df = pd.read_csv(r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 100\colmap_ws\export_txt\hyperspectral_pc_df.csv")
average_tuples = average_tuples_in_track(hyperspectral_pc_df)
# print(f"Average number of tuples in non-empty TRACK lists: {average_tuples}")
