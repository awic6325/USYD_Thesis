import numpy as np
import pandas as pd

def get_absolute_orientation(hyperspectral_pc_df, P_real, P_colmap):

    # Extract coordinates
    coords = hyperspectral_pc_df[['X', 'Y', 'Z']].values
    scaled_coords = hyperspectral_pc_df[['X_scaled', 'Y_scaled', 'Z_scaled']].values

    # Check for NaNs or Infs
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        raise ValueError("Input data contains NaN or Inf values.")

    # Step 1: Compute the least-squares scale factor
    world_norms = np.linalg.norm(P_real, axis=1)
    reconstructed_norms = np.linalg.norm(P_colmap, axis=1)
    scale_factor = np.sum(world_norms * reconstructed_norms) / np.sum(reconstructed_norms**2)

    print(f"Scaling factor: {scale_factor}")

    # Step 2: Scale the COLMAP points
    P_colmap_scaled = scale_factor * P_colmap

    # Step 3: Compute centroids
    c_real = np.mean(P_real, axis=0)
    c_colmap_scaled = np.mean(P_colmap_scaled, axis=0)

    # Step 4: Centre the point sets
    Q_real = P_real - c_real
    Q_colmap_scaled = P_colmap_scaled - c_colmap_scaled

    # Step 5: Compute the cross-covariance matrix
    H = Q_colmap_scaled.T @ Q_real

    # Step 6: Compute SVD of H
    U, _, Vt = np.linalg.svd(H)

    # Step 7: Compute rotation matrix
    R_matrix = Vt.T @ U.T

    # Ensure a proper rotation (det(R) = 1)
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T

    # Step 8: Compute translation vector
    t = c_real - R_matrix @ c_colmap_scaled

    # Step 9: Define transformation function
    def transform_colmap_to_enu(points_colmap):
        points_colmap_scaled = scale_factor * points_colmap
        return (R_matrix @ points_colmap_scaled.T).T + t
    
    # Find ENU coords
    points_enu = transform_colmap_to_enu(coords)
    print(points_enu)

    # Store in dataframe
    hyperspectral_pc_df['X_ENU'] = points_enu[:, 0]
    hyperspectral_pc_df['Y_ENU'] = points_enu[:, 1]
    hyperspectral_pc_df['Z_ENU'] = points_enu[:, 2]

    # Verify distance preservation
    # Calculate distances between a few pairs of points before transformation
    distances_before = np.linalg.norm(scaled_coords[1] - scaled_coords[0]), np.linalg.norm(scaled_coords[2] - scaled_coords[0])

    # Calculate distances after transformation
    distances_after = np.linalg.norm(points_enu[1] - points_enu[0]), np.linalg.norm(points_enu[2] - points_enu[0])

    print("Distances before transformation:", distances_before)
    print("Distances after transformation:", distances_after)

    # Return updated dataframe
    return hyperspectral_pc_df


### PARAMETERS
kind = 'cubic'

coreg = 'A_xy'
# coreg = 'S_xy'

# n_daq = 100
# n_daq = 200
n_daq = 300

# exp = "exp1_jan1"
# exp = "exp1_jan2"
# exp = "exp2"
exp = "exp3"

###### EXPERIMENT 1: COLOURED PAPER ######
if exp == "exp1_jan1":

    ### JAN 1st ###
    if n_daq == 100:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 100\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            # [   1.19571844,  -74.57558849,    0.        ], # G1
            # [  -2.05405518,   -1.26864251,    0.        ], # G2
            [ -46.74656892,   -1.84413646,    0.        ], # G3
            [ -67.24628804,  -82.56482673,    0.        ], # G4
            [ -94.15123414,  -60.84017706,    0.        ], # G5
            [-104.49757254,   -0.9066333,     0.        ]  # G6
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            # [, , ],
            # [, , ],
            [4.723051, 2.092659, 6.813686],
            [-5.341925, 0.845694, 6.062725],
            [-3.808710, -1.648230, 9.525494],
            [3.429728, -2.406822, 12.181924]
        ])

        colmap_points_2 = np.array([
            # [, , ],
            # [, , ],
            [4.723051, 2.081926, 6.824117],
            [-5.35235, 0.841991, 6.062725],
            [-3.787851, -1.663843, 9.556784],
            [3.408867, -2.411520, 12.185150]
        ])

        colmap_points_3 = np.array([
            # [, , ],
            # [, , ],
            [4.733480, 2.081997, 6.824117],
            [-5.334888, 0.847345, 6.062725],
            [-3.798281, -1.649849, 9.535924],
            [3.440157, -2.399033, 12.174721]
        ])

    elif n_daq == 200:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 200\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [   1.19571844,  -74.57558849,    0.        ],
            [  -2.05405518,   -1.26864251,    0.        ],
            [ -46.74656892,   -1.84413646,    0.        ],
            [ -67.24628804,  -82.56482673,    0.        ],
            [ -94.15123414,  -60.84017706,    0.        ],
            [-104.49757254,   -0.9066333,     0.        ]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [-2.234794, 4.834909, 2.088301],
            [4.549187, 4.520479, 3.337723],
            [3.672726, 1.408482, 6.237499],
            [-3.889466, 0.312750, 6.242162],
            [-2.443845, -1.819748, 8.503242],
            [3.159904, -2.456984, 9.920496]
        ])

        colmap_points_2 = np.array([
            [-2.229391, 4.835836, 2.088301],
            [4.558510, 4.520338, 3.337723],
            [3.672726, 1.396954, 6.246823],
            [-3.893733, 0.315626, 6.237499],
            [-2.456790, -1.822454, 8.503242],
            [3.141256, -2.456711, 9.920496]
        ])

        colmap_points_3 = np.array([
            [-2.238715, 4.834188, 2.088301],
            [4.567835, 4.518797, 3.337723],
            [3.682051, 1.408983, 6.237499],
            [-3.893733, 0.309715, 6.242162],
            [-2.453169, -1.806942, 8.465945],
            [3.150579, -2.456961, 9.920496]
        ])

    elif n_daq == 300:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 1st\20250101_session16_outdoor_colour - 300\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [   1.19571844,  -74.57558849,    0.        ],
            [  -2.05405518,   -1.26864251,    0.        ],
            [ -46.74656892,   -1.84413646,    0.        ],
            [ -67.24628804,  -82.56482673,    0.        ],
            [ -94.15123414,  -60.84017706,    0.        ],
            [-104.49757254,   -0.9066333,     0.        ]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [-2.562067, 3.358494, 3.829951],
            [4.390081, 3.775661, 4.329638],
            [4.346630, -0.400074, 5.748927],
            [-3.040027, -2.740992, 5.862058],
            [-1.008696, -5.326563, 6.925830],
            [4.846315, -5.483831, 7.538344]
        ])

        colmap_points_2 = np.array([
            [-2.583792, 3.380157, 3.820796],
            [4.400944, 3.771214, 4.331935],
            [4.341200, -0.400074, 5.749421],
            [-3.034595, -2.724697, 5.859595],
            [-1.019558, -5.320890, 6.916959],
            [4.824591, -5.468206, 7.534142]
        ])

        colmap_points_3 = np.array([
            [-2.551204, 3.358433, 3.832829],
            [4.390081, 3.792942, 4.324553],
            [4.352062, -0.372917, 5.742537],
            [-3.023733, -2.735561, 5.863111],
            [-1.008696, -5.320890, 6.921775],
            [4.835453, -5.471928, 7.534142]
        ])

elif exp == "exp1_jan2":
    
    ### JAN 2nd ###
    if n_daq == 100:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 100\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [   1.19571844,  -74.57558849,    0.        ],
            [  -2.05405518,   -1.26864251,    0.        ],
            [ -46.74656892,   -1.84413646,    0.        ],
            [ -67.24628804,  -82.56482673,    0.        ],
            [ -94.15123414,  -60.84017706,    0.        ],
            [-104.49757254,   -0.9066333,     0.        ]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [-3.019480, 4.596272, 2.469871],
            [3.595959, 4.141846, 4.777105],
            [2.447626, 0.498314, 7.226850],
            [-5.303045, -0.630600, 5.821351],
            [-4.170973, -3.136789, 8.089948],
            [1.371996, -3.960897, 10.383925]
        ])

        colmap_points_2 = np.array([
            [-3.026180, 4.601917, 2.462831],
            [3.584667, 4.125343, 4.788394],
            [2.439249, 0.498314, 7.226850],
            [-5.299887, -0.626534, 5.821351],
            [-4.159683, -3.141357, 8.096112],
            [1.394575, -3.944498, 10.376519]
        ])

        colmap_points_3 = np.array([
            [-3.022799, 4.596272, 2.468475],
            [3.607246, 4.133419, 4.783192],
            [2.455753, 0.509602, 7.220557],
            [-5.297503, -0.619312, 5.815706],
            [-4.163074, -3.125502, 8.084825],
            [1.371996, -3.949608, 10.376101]
        ])

    elif n_daq == 200:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 200\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [   1.19571844,  -74.57558849,    0.        ],
            [  -2.05405518,   -1.26864251,    0.        ],
            [ -46.74656892,   -1.84413646,    0.        ],
            [ -67.24628804,  -82.56482673,    0.        ],
            [ -94.15123414,  -60.84017706,    0.        ],
            [-104.49757254,   -0.9066333,     0.        ]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [-2.772446, 5.172821, -0.150755],
            [3.632477, 5.316402, 2.639385],
            [2.138199, 2.833389, 6.092959],
            [-5.550188, 1.566910, 4.617285],
            [-4.706945, -0.009960, 7.680239],
            [0.631523, -0.177946, 10.513783]
        ])

        colmap_points_2 = np.array([
            [-2.766247, 5.175083, -0.150755],
            [3.644875, 5.318371, 2.639385],
            [2.132000, 2.837273, 6.086760],
            [-5.543987, 1.577911, 4.604884],
            [-4.688343, -0.006105, 7.682579],
            [0.619125, -0.170692, 10.501381]
        ])

        colmap_points_3 = np.array([
            [-2.766510, 5.171154, -0.144555],
            [3.644875, 5.313763, 2.646270],
            [2.144402, 2.835182, 6.092959],
            [-5.543987, 1.570435, 4.617285],
            [-4.700745, -0.012421, 7.686440],
            [0.631523, -0.168905, 10.501381]
        ])

    elif n_daq == 300:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Coloured Sheets\Jan 2nd 2pm\20250102_session3_outdoor_colour - 300\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [   1.19571844,  -74.57558849,    0.        ],
            [  -2.05405518,   -1.26864251,    0.        ],
            [ -46.74656892,   -1.84413646,    0.        ],
            [ -67.24628804,  -82.56482673,    0.        ],
            [ -94.15123414,  -60.84017706,    0.        ],
            [-104.49757254,   -0.9066333,     0.        ]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [-2.313519, 5.504677, 1.149038],
            [5.322794, 4.596157, 2.647456],
            [4.183994, 1.039978, 6.021892],
            [-4.662663, 0.422047, 5.938214],
            [-3.212194, -2.155231, 8.584840],
            [3.191547, -3.353966, 10.295379]
        ])

        colmap_points_2 = np.array([
            [-2.325130, 5.501549, 1.149038],
            [5.298819, 4.600006, 2.647456],
            [4.178001, 1.055851, 6.009906],
            [-4.650675, 0.426426, 5.937982],
            [-3.200207, -2.167217, 8.597779],
            [3.189045, -3.353655, 10.295379]
        ])

        colmap_points_3 = np.array([
            [-2.309954, 5.498684, 1.155032],
            [5.332212, 4.593639, 2.647456],
            [4.195982, 1.036289, 6.021892],
            [-4.656668, 0.425037, 5.937982],
            [-3.224180, -2.143246, 8.574845],
            [3.189045, -3.340111, 10.283391]
        ])


###### EXPERIMENT 2: DRIVEWAY ######
elif exp == "exp2":
    
    ### JAN 6TH ###
    if n_daq == 100:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 100\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [ -0.4262389,    1.86755151,   7.40097189],
            [ 11.14943833, 111.27138141,  -7.67543766],
            [122.72695102,   9.55623108,  -9.02784324],
            [129.12285937, 111.40814031,  -5.68511519],
            [-13.76223925, 183.1467814,  -27.12023654],
            [117.19089983, 182.75023346, -13.89186332]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [3.126002, -6.097818, 4.595664],
            [4.199012, -3.160452, 3.693534],
            [-2.607899, -4.930920, 5.205475],
            [-0.911201, -2.060617, 4.191926],
            [4.782462, 1.259010, 2.845500],
            [-0.508821, 2.774637, 3.234715]
        ])

        colmap_points_2 = np.array([
            [3.112590, -6.104525, 4.600083],
            [4.192306, -3.149820, 3.691523],
            [-2.601192, -4.943985, 5.207150],
            [-0.911201, -2.053909, 4.190502],
            [4.795875, 1.252304, 2.843340],
            [-0.502115, 2.781343, 3.233669]
        ])

        colmap_points_3 = np.array([
            [3.119296, -6.091112, 4.595613],
            [4.199012, -3.153746, 3.692566],
            [-2.601192, -4.937626, 5.205912],
            [-0.917907, -2.047203, 4.189654],
            [4.790570, 1.238892, 2.846527],
            [-0.502115, 2.767931, 3.234911]
        ])

    elif n_daq == 200:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 200\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [ -0.4262389,    1.86755151,   7.40097189],
            [ 11.14943833, 111.27138141,  -7.67543766],
            [122.72695102,   9.55623108,  -9.02784324],
            [129.12285937, 111.40814031,  -5.68511519],
            [-13.76223925, 183.1467814,  -27.12023654],
            [117.19089983, 182.75023346, -13.89186332]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [2.780546, -6.318792, 4.026475],
            [3.858126, -3.381594, 3.288330],
            [-2.915415, -5.202356, 4.569164],
            [-1.214537, -2.317608, 3.730057],
            [4.450062, 1.027950, 2.653062],
            [-0.806175, 2.507790, 3.005175]
        ])

        colmap_points_2 = np.array([
            [2.785982, -6.326284, 4.026475],
            [3.850632, -3.381594, 3.289750],
            [-2.907922, -5.202356, 4.567998],
            [-1.207044, -2.321354, 3.729279],
            [4.446314, 1.031698, 2.653038],
            [-0.802429, 2.507790, 3.005960]
        ])

        colmap_points_3 = np.array([
            [2.775204, -6.311300, 4.026475],
            [3.865618, -3.381594, 3.287757],
            [-2.915415, -5.209850, 4.570622],
            [-1.222030, -2.325102, 3.731938],
            [4.453808, 1.024204, 2.653186],
            [-0.802429, 2.504044, 3.005676]
        ])

    elif n_daq == 300:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Driveway\20250106_session3_driveway - 300\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [ -0.4262389,    1.86755151,   0],
            [ 11.14943833, 111.27138141,  -10],
            [122.72695102,   9.55623108,  0],
            [129.12285937, 111.40814031,  -10],
            [-13.76223925, 183.1467814,  -25],
            [117.19089983, 182.75023346, -25]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [2.391012, -3.906355, 6.447552],
            [3.377993, -1.310339, 4.735693],
            [-3.453910, -3.362280, 5.873038],
            [-1.837229, -0.780725, 4.156642],
            [3.762859, 2.621960, 2.501099],
            [-1.650688, 3.493069, 1.647443]
        ])

        colmap_points_2 = np.array([
            [2.385117, -3.914127, 6.453417],
            [3.362576, -1.310339, 4.735741],
            [-3.446136, -3.377825, 5.882981],
            [-1.841114, -0.780783, 4.156642],
            [3.766747, 2.610889, 2.507942],
            [-1.654575, 3.493069, 1.646917]
        ])

        colmap_points_3 = np.array([
            [2.398787, -3.914127, 6.452456],
            [3.378120, -1.325018, 4.743465],
            [-3.461682, -3.362280, 5.873020],
            [-1.833342, -0.774036, 4.153498],
            [3.758974, 2.616068, 2.504985],
            [-1.658461, 3.492481, 1.646123]
        ])




###### EXPERIMENT 3: KU-RING-GAI ######
elif exp == "exp3":
    
    ### JAN 6TH ###
    if n_daq == 100:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 100\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [-3.04498690e+00,  1.01632082e+02, -1.24815745e+00],
            [ 2.11349558e+00, -9.35174650e-01, -1.35141773e-02],
            [ 1.20618964e+02,  2.84628816e+00,  6.19902144e+00],
            [ 6.83141689e+01,  1.05068605e+02,  4.50517862e+00],
            [ 1.99169019e+02, -4.38403369e+01,  1.60441857e+01],
            [ 2.29329824e+02,  8.37289510e+01,  1.45137793e+01]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [0.680735, 4.188139, -0.413695],
            [4.304373, 2.796474, 0.743652],
            [2.488966, -0.126732, 3.843478],
            [-0.445824, 2.430000, 1.489906],
            [3.034306, -2.985435, 6.118119],
            [-2.580542, -2.321938, 5.788045]
        ])

        colmap_points_2 = np.array([
            [0.680735, 4.195316, -0.418515],
            [4.304373, 2.799170, 0.736476],
            [2.481791, -0.117174, 3.839648],
            [-0.446651, 2.430136, 1.489906],
            [3.019955, -2.965857, 6.111281],
            [-2.591305, -2.316473, 5.786408]
        ])

        colmap_points_3 = np.array([
            [0.687910, 4.188139, -0.413760],
            [4.333076, 2.835823, 0.707774],
            [2.483107, -0.124350, 3.843478],
            [-0.438648, 2.421652, 1.497082],
            [3.027130, -2.984955, 6.118119],
            [-2.603956, -2.327236, 5.795220]
        ])


    elif n_daq == 200:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 200\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [-3.04498690e+00,  1.01632082e+02, -1.24815745e+00],
            [ 2.11349558e+00, -9.35174650e-01, -1.35141773e-02],
            [ 1.20618964e+02,  2.84628816e+00,  6.19902144e+00],
            [ 6.83141689e+01,  1.05068605e+02,  4.50517862e+00],
            [ 1.99169019e+02, -4.38403369e+01,  1.60441857e+01],
            [ 2.29329824e+02,  8.37289510e+01,  1.45137793e+01]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [-1.457508, 3.685249, -0.202629],
            [2.591225, 3.253616, 0.018503],
            [2.267617, -0.375114, 2.990300],
            [-1.669610, 1.509819, 1.625754],
            [3.961164, -3.148266, 4.700672],
            [-1.589318, -4.032795, 5.557588]
        ])

        colmap_points_2 = np.array([
            [-1.459265, 3.686958, -0.202629],
            [2.585831, 3.195175, 0.018503],
            [2.264225, -0.365238, 2.984906],
            [-1.669610, 1.503849, 1.631147],
            [3.966557, -3.148266, 4.700776],
            [-1.594101, -4.027401, 5.555319]
        ])

        colmap_points_3 = np.array([
            [-1.464434, 3.696037, -0.208023],
            [2.585831, 3.197029, 0.040076],
            [2.262223, -0.380104, 2.995693],
            [-1.680397, 1.509846, 1.625754],
            [3.961164, -3.164448, 4.703427],
            [-1.594101, -4.039165, 5.562981]
        ])

    elif n_daq == 300:

        # Export directory
        export_dir = r"C:\Users\Ashnith\Documents\01_Thesis\19. Final Thesis Data\Experiments\Ku-ring-gai\20241226_session6_kuringgai_500 - 300\colmap_ws\export_txt"

        # GCP coordinates:
        P_real = np.array([
            [-3.04498690e+00,  1.01632082e+02, -1.24815745e+00],
            [ 2.11349558e+00, -9.35174650e-01, -1.35141773e-02],
            [ 1.20618964e+02,  2.84628816e+00,  6.19902144e+00],
            [ 6.83141689e+01,  1.05068605e+02,  4.50517862e+00],
            [ 1.99169019e+02, -4.38403369e+01,  1.60441857e+01],
            [ 2.29329824e+02,  8.37289510e+01,  1.45137793e+01]
        ])

        # Reconstructed Points from COLMAP:
        colmap_points_1 = np.array([
            [-1.344977, 3.422848, -0.090171],
            [2.763938, 2.866798, 0.173189],
            [2.273356, -0.841602, 3.207550],
            [-1.665589, 1.209476, 1.764797],
            [3.899316, -3.738674, 4.970915],
            [-1.820169, -4.460213, 5.766719]
        ])

        colmap_points_2 = np.array([
            [-1.349781, 3.422142, -0.090171],
            [2.754273, 2.813778, 0.201815],
            [2.267631, -0.839790, 3.207550],
            [-1.659863, 1.215664, 1.759071],
            [3.893590, -3.728645, 4.969433],
            [-1.814445, -4.460392, 5.766719]
        ])

        colmap_points_3 = np.array([
            [-1.344977, 3.42214, -0.089607],
            [2.759999, 2.814663, 0.196089],
            [2.267631, -0.831734, 3.201824],
            [-1.659863, 1.209698, 1.764797],
            [3.887864, -3.728645, 4.969918],
            [-1.831620, -4.457413, 5.766719]
        ])


# Stack the datasets into a 3D array
all_points = np.stack([colmap_points_1, colmap_points_2, colmap_points_3], axis=0)

# Compute the average for each ground control point (along axis 0 of the stacked array)
P_colmap = np.mean(all_points, axis=0)

# Load current hyperspectral pc
hyperspectral_pc_df_filename = f"hyperspectral_pc_df_{coreg}_{kind}"
hyperspectral_pc_df_filepath = f"{export_dir}/{hyperspectral_pc_df_filename}.csv"
hyperspectral_pc_df = pd.read_csv(hyperspectral_pc_df_filepath)

# Save hyperspectral pc with coords rotated and transformed to ENU
hyperspectral_pc_df_ENU = get_absolute_orientation(hyperspectral_pc_df, P_real, P_colmap)
hyperspectral_pc_df_ENU.to_csv(f"{export_dir}/{hyperspectral_pc_df_filename}_ENU.csv", index=False)





import matplotlib.pyplot as plt

def display_point_cloud(hyperspectral_pc_df):
    """
    Display the 3D point cloud using the X_scaled, Y_scaled, Z_scaled columns,
    filtering points with non-empty WEIGHTED_SPECTRUM_FILENAME.

    Parameters:
        hyperspectral_pc_df (pd.DataFrame): DataFrame containing the point cloud data.
    """

    # ENU x- and y- coords
    x_enu = hyperspectral_pc_df["X_ENU"]
    y_enu = hyperspectral_pc_df["Y_ENU"]
    rgb_enu = hyperspectral_pc_df[['R', 'G', 'B']].values / 255.0  # Normalise RGB to [0, 1] range

    # Filter points with non-empty WEIGHTED_SPECTRUM_FILENAME
    filtered_df = hyperspectral_pc_df

    # Extract coordinates and RGB colours
    x = filtered_df['X_ENU']
    y = filtered_df['Y_ENU']
    z = filtered_df['Z_ENU']

    # Define limits
    x_min, x_max = -150, 150
    y_min, y_max = -150, 150
    z_min, z_max = -10, 10

    filtered_df = filtered_df[
        (filtered_df["X_ENU"] >= x_min) &
        (filtered_df["X_ENU"] <= x_max) &
        (filtered_df["Y_ENU"] >= y_min) &
        (filtered_df["Y_ENU"] <= y_max) &
        (filtered_df["Z_ENU"] >= z_min) &
        (filtered_df["Z_ENU"] <= z_max)
    ]

    x = filtered_df['X_ENU']
    y = filtered_df['Y_ENU']
    z = filtered_df['Z_ENU']

    rgb = filtered_df[['R', 'G', 'B']].values / 255.0  # Normalise RGB to [0, 1] range

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=rgb, s=5)  # Use RGB for colour

    # Label axes
    ax.set_xlabel('X_scaled')
    ax.set_ylabel('Y_scaled')
    ax.set_zlabel('Z_scaled')

    plt.title("3D Hyperspectral Point Cloud")
    plt.show()


    plt.scatter(x_enu, y_enu, s=1, c=rgb_enu)
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.title('Flattened 2D Point Cloud')
    plt.axis('equal')
    plt.show()

display_point_cloud(hyperspectral_pc_df_ENU)
