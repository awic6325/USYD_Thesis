import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Viewing distance
# d_v = 15  # Viewing distance in cm
# d_v = 30  # Viewing distance in cm
# d_v = 60  # Viewing distance in cm
# d_v = 90  # Viewing distance in cm
# d_v = 120  # Viewing distance in cm
d_v = 150  # Viewing distance in cm

# Directory
# savefile_dir = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\project_Lenovo_v2\angle_var\20241225_session6_d15cm\anglevar_df"
# savefile_dir = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\project_Lenovo_v2\angle_var\20241225_session7_d30cm\anglevar_df"
# savefile_dir = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\project_Lenovo_v2\angle_var\20241225_session8_d60cm\anglevar_df"
# savefile_dir = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\project_Lenovo_v2\angle_var\20241225_session9_d90cm\anglevar_df"
# savefile_dir = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\project_Lenovo_v2\angle_var\20241225_session11_d120cm\anglevar_df"
# savefile_dir = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\project_Lenovo_v2\angle_var\20241225_session12_d150cm\anglevar_df"
# savefile_dir = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\project_Lenovo_v2\angle_var\best_data\anglevar_150cm_df_clean.csv"

# savefile_dir = r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\project_Lenovo_v2\angle_var\best_data"
savefile_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Full Thesis Results List\Angle_Var\angle_var\best_data"

# Path to angle variance data
anglevar_df_path = f"{savefile_dir}/anglevar_15cm_df_clean.csv"
# anglevar_df_path = f"{savefile_dir}/anglevar_30cm_df_clean.csv"
# anglevar_df_path = f"{savefile_dir}/anglevar_60cm_df_clean.csv"
# anglevar_df_path = f"{savefile_dir}/anglevar_90cm_df_clean.csv"
# anglevar_df_path = f"{savefile_dir}/anglevar_120cm_df_clean.csv"
# anglevar_df_path = f"{savefile_dir}/anglevar_150cm_df_clean.csv"

# Load the angle variance data
# anglevar_df = pd.read_csv("anglevar_df.csv")
anglevar_df = pd.read_csv(anglevar_df_path)

# Extract columns from the dataframe
angles = anglevar_df["Angle"].values
normalised_intensities = anglevar_df["Normalised Intensity"].values

print(anglevar_df)

# Create a cubic spline interpolation function for the normalised intensity
spline = CubicSpline(angles, normalised_intensities)

# Plot the cubic spline
plt.figure(figsize=(8, 6))
angles_fine = np.linspace(min(angles), max(angles), 500)
plt.plot(angles, normalised_intensities, 'o', label='Data Points')
plt.plot(angles_fine, spline(angles_fine), '-', label='Cubic Spline')
plt.title('Cubic Spline Interpolation of Normalised Intensity')
plt.xlabel('Angle (degrees)')
plt.ylabel('Normalised Intensity')
plt.legend()
plt.grid()
plt.savefig(f"{savefile_dir}/cubic_spline_plot.png", dpi=300)
plt.show()

# Polynomial fitting to 5th order
polynomial_coefficients = np.polyfit(angles, normalised_intensities, 5)
polynomial = np.poly1d(polynomial_coefficients)

# Plot the polynomial fit
plt.figure(figsize=(8, 6))
plt.plot(angles, normalised_intensities, 'o', label='Data Points')
plt.plot(angles_fine, polynomial(angles_fine), '-', label='5th Order Polynomial Fit')
plt.title('5th Order Polynomial Fit of Normalised Intensity')
plt.xlabel('Angle (degrees)')
plt.ylabel('Normalised Intensity')
plt.legend()
plt.grid()
plt.savefig(f"{savefile_dir}/polynomial_fit_plot.png", dpi=300)
plt.show()

# Input parameters
box_size = 40  # Box size in pixels
pixel_length = 0.233 / 10  # Convert mm to cm

# Screen dimensions 
screen_width_pixels = 2560  
screen_height_pixels = 1440  

# Calculate the number of white boxes that can fit horizontally and vertically
num_boxes_x = screen_width_pixels // box_size
num_boxes_y = screen_height_pixels // box_size

# Calculate the centre of the screen in pixels
centre_x = (screen_width_pixels - 1) / 2
centre_y = (screen_height_pixels - 1) / 2

# Initialise the output matrix V_xy
V_xy = np.zeros((num_boxes_y, num_boxes_x))
theta_xy = np.zeros((num_boxes_y, num_boxes_x))
d_0_xy = np.zeros((num_boxes_y, num_boxes_x))

# Calculate the normalised intensity for each box
for i in range(num_boxes_y):
    for j in range(num_boxes_x):

        # Calculate the centre of the current box in pixels
        box_centre_x = j * box_size + box_size / 2
        box_centre_y = i * box_size + box_size / 2

        # Calculate the distance d_0 in cm
        d_0 = np.sqrt(((box_centre_x - centre_x) * pixel_length) ** 2 +
                      ((box_centre_y - centre_y) * pixel_length) ** 2)
        d_0_xy[i, j] = d_0

        # Calculate the viewing angle theta (in degrees)
        theta = np.degrees(np.arctan(d_0 / d_v))
        theta_xy[i, j] = theta

        # Interpolate the normalised intensity using the cubic spline
        if 0 <= theta <= 90:  # Ensure theta is within the interpolation range

            # V_xy[i, j] = spline(theta)
            V_xy[i, j] = polynomial(theta)

            # Correct for any error in spline/polynomial
            if V_xy[i, j] > 1:
                V_xy[i, j] = 1 
        else:
            V_xy[i, j] = 0  # Assign a default value if theta is out of range

# Output the result
print("Normalised Intensity Matrix (V_xy):")
print(V_xy)

print("Theta Matrix (theta_xy):")
print(theta_xy)

print("d_0 Matrix (d_0_xy):")
print(d_0_xy)

# Save V_xy to a file
np.savetxt(f"{savefile_dir}/V_xy.csv", V_xy, delimiter=",", fmt="%.3f", header="Normalised Intensity Matrix", comments="")

# Display V_xy as a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(V_xy, cmap='viridis', origin='upper')
plt.colorbar(label='Normalised Intensity')
plt.title('Heatmap of Normalised Intensity (V_xy)')
plt.xlabel('Box Index (Horizontal)')
plt.ylabel('Box Index (Vertical)')
plt.savefig(f"{savefile_dir}/V_xy_heatmap.png", dpi=300)
plt.show()

