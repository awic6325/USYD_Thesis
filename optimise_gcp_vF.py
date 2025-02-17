import numpy as np
from scipy.optimize import minimize

# Objective function for least-squares fitting
def objective(pts):
    points = pts.reshape(-1, 3)  # Reshape flat array to Nx3 matrix
    
    # Include G1 as fixed point at origin
    # points = np.vstack(([0, 0, 0], pts))  # Add G1 back as (0, 0, 0)
    
    error = 0
    
    # Distance errors
    for (i, j), d_ij in measured_distances.items():
        dist = np.linalg.norm(points[i] - points[j])  # Calculated distance between points i and j
        error += (dist - d_ij) ** 2
    
    # Angle errors
    for (i, j, k), theta in measured_angles.items():
        # Calculate angle between vectors (points[i] - points[j]) and (points[k] - points[j])
        vec_ij = points[i] - points[j]
        vec_kj = points[k] - points[j]
        cos_theta = np.dot(vec_ij, vec_kj) / (np.linalg.norm(vec_ij) * np.linalg.norm(vec_kj))
        calculated_angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        error += (calculated_angle - theta) ** 2

        # print(f"vec_ij: {vec_ij}, vec_kj: {vec_kj}, cos_theta: {cos_theta}, angle: {calculated_angle}")

    # print("Error:", error)

    return error


### EXPERIMENT: Coloured Sheets
# Jan 1st and Jan 2nd 2025
# Initial positions for G1 to G6 based on known measurements
initial_points = [
    [0, -74.5, 0],          # G1
    [0, 0, 0],              # G2 
    [-46.5, 0, 0],        # Initial estimate for G3
    [-67, -82, 0],       # Initial guess for G4
    [-97, -63.5, 0],       # Initial guess for G5
    [-103, -2, 0]        # Initial guess for G6
]
initial_points = np.array(initial_points)

# Measured distances between points 
measured_distances = {
    (0, 1): 74.5,       # G1-G2 distance
    (0, 2): 86,       # G1-G3 distance
    (0, 3): 68.5,       # G1-G4 distance
    (0, 4): 98,       # G1-G5 distance    
    (0, 5): 128,       # G1-G6 distance
    (1, 2): 46.5,       # G2-G3 distance
    (1, 3): 106,       # G2-G4 distance
    (1, 4): 105,       # G2-G5 distance
    (1, 5): 103.5,       # G2-G6 distance
    (2, 3): 82,       # G3-G4 distance
    (2, 4): 78.5,       # G3-G5 distance
    (2, 5): 57.5,       # G3-G6 distance
    (3, 4): 35.5,       # G4-G5 distance
    (3, 5): 89,       # G4-G6 distance
    (4, 5): 62,       # G4-G6 distance
}

# Measured angles between triplets of points (in degrees)
measured_angles = {
    (0, 1, 2): np.radians(83),  # Angle G1-G2-G3
    (0, 2, 3): np.radians(49),  # Angle G1-G3-G4
    (0, 4, 5): np.radians(105),    # Angle G1-G5-G6
    (1, 2, 4): np.radians(135),  # Angle G2-G3-G5
    (1, 3, 5): np.radians(61),    # Angle G2-G4-G6
    (2, 3, 5): np.radians(40),    # Angle G3-G4-G6
}

# Optimise the coordinates of all points except the fixed point G1
result = minimize(objective, initial_points.flatten(), method='L-BFGS-B')
optimised_points = (result.x.reshape(-1, 3))  

# Print the optimised coordinates
print("\nEXPERIMENT: Coloured Sheets")
print("Optimised GCP Coordinates:\n", optimised_points)


### EXPERIMENT: Driveway
# Jan 6th 2025
# Initial positions for G1 to G6 based on known measurements
initial_points = [
    [0, 0, 0],          # G1
    [11.5, 111, -8],              # G2 
    [123, 10, 0],        # Initial estimate for G3
    [126, 111.5, -8],       # Initial guess for G4
    [-13.5, 184.5, -20],       # Initial guess for G5
    [119, 183, -20]        # Initial guess for G6
]
initial_points = np.array(initial_points)

# Measured distances between points 
measured_distances = {
    (0, 1): 111,       # G1-G2 distance
    (0, 2): 124.5,       # G1-G3 distance
    (0, 3): 170,       # G1-G4 distance
    (0, 4): 185,       # G1-G5 distance    
    (0, 5): 217,       # G1-G6 distance
    (1, 2): 151,       # G2-G3 distance
    (1, 3): 118,       # G2-G4 distance
    (1, 4): 78.5,       # G2-G5 distance
    (1, 5): 128,       # G2-G6 distance
    (2, 3): 102,       # G3-G4 distance
    (2, 4): 221.5,       # G3-G5 distance
    (2, 5): 173.5,       # G3-G6 distance
    (3, 4): 161.5,       # G4-G5 distance
    (3, 5): 72.5,       # G4-G6 distance
    (4, 5): 131.5,       # G4-G6 distance
}

# Measured angles between triplets of points 
measured_angles = {
    (0, 1, 2): np.radians(55),  # Angle G1-G2-G3
    (0, 2, 3): np.radians(94),  # Angle G1-G3-G4
    (0, 4, 5): np.radians(88),    # Angle G1-G5-G6
    (1, 2, 4): np.radians(9),  # Angle G2-G3-G5
    (1, 3, 5): np.radians(78),    # Angle G2-G4-G6
    (2, 3, 5): np.radians(165),    # Angle G3-G4-G6
}

# Optimise the coordinates of all points
result = minimize(objective, initial_points.flatten(), method='L-BFGS-B')
optimised_points = (result.x.reshape(-1, 3))  

# Print the optimised coordinates
print("\nEXPERIMENT: Driveway")
print("Optimised GCP Coordinates:\n", optimised_points)


### EXPERIMENT: Ku-ring-gai
# Outdoor Experiment 26/12/2024
# Initial positions for G1 to G6 based on known measurements
initial_points = [
    [0, 102.5, 0],          # G1
    [0, 0, 0],              # G2 
    [118.5, 0, 6],        # Initial estimate for G3
    [71, 104, 3],       # Initial guess for G4
    [202, -45, 16],       # Initial guess for G5
    [225, 87, 15]        # Initial guess for G6
]
initial_points = np.array(initial_points)

# Measured distances between points 
measured_distances = {
    (0, 1): 102.5,       # G1-G2 distance
    (0, 2): 158.5,       # G1-G3 distance
    (0, 3): 71.5,       # G1-G4 distance
    (0, 4): 250.0,       # G1-G5 distance    
    (0, 5): 233.5,      # G1-G6 distance
    (1, 2): 118.5,       # G2-G3 distance
    (1, 3): 125.0,       # G2-G4 distance
    (1, 4): 202.0,       # G2-G5 distance
    (1, 5): 243.5,       # G2-G6 distance
    (2, 3): 115.0,       # G3-G4 distance
    (2, 4): 92.0,       # G3-G5 distance
    (2, 5): 135.5,       # G3-G6 distance
    (3, 4): 198.5,       # G4-G5 distance
    (3, 5): 162.5,       # G4-G6 distance
    (4, 5): 131.0,       # G4-G6 distance
}

# Measured angles between triplets of points
measured_angles = {
    (0, 1, 2): np.radians(90),  # Angle G1-G2-G3
    (0, 2, 3): np.radians(25),  # Angle G1-G3-G4
    (0, 4, 5): np.radians(67),    # Angle G1-G5-G6
    (1, 2, 4): np.radians(208),  # Angle G2-G3-G5
    (1, 3, 5): np.radians(125),    # Angle G2-G4-G6
    (2, 3, 5): np.radians(58),    # Angle G3-G4-G6
}

# Optimise the coordinates of all points
result = minimize(objective, initial_points.flatten(), method='L-BFGS-B')
optimised_points = (result.x.reshape(-1, 3))  

# Print the optimised coordinates
print("\nEXPERIMENT: Ku-ring-gai")
print("Optimised GCP Coordinates:\n", optimised_points)
