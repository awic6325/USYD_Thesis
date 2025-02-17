import pandas as pd
from scipy.interpolate import interp1d

def interpolate_all_ellipse_parameters(ellipse_df, distance):
    """
    Interpolate all ellipse parameters at a given distance in cm
    
    Parameters:
        ellipse_df: pd.dataframe containing ellipse parameters, i.e. 'Distance', 
                                   'Centre_X', 'Centre_Y', 'Major_Axis', 'Minor_Axis', and 'Angle'
        distance: target distance for interpolation, cm
        
    Returns:
        interpolated ellipse parameters at the given distance, as dict
    """
    # Ensure the DataFrame is sorted by distance
    ellipse_df = ellipse_df.sort_values('Distance')

    # Columns to interpolate (exclude Filename)
    columns_to_interpolate = ['Centre_X', 'Centre_Y', 'Major_Axis', 'Minor_Axis', 'Angle']

    # Create interpolation functions for each column
    interpolation_functions = {
        # col: interp1d(ellipse_df['Distance'], ellipse_df[col], kind='linear', fill_value="extrapolate")
        # col: interp1d(ellipse_df['Distance'], ellipse_df[col], kind='quadratic', fill_value="extrapolate")
        col: interp1d(ellipse_df['Distance'], ellipse_df[col], kind='cubic', fill_value="extrapolate")
        for col in columns_to_interpolate
    }
    
    # Interpolate all parameters
    interpolated_values = {col: interpolation_functions[col](distance) for col in columns_to_interpolate}
    interpolated_values['Distance'] = distance  # Add the given distance to the result
    
    return interpolated_values


ellipse_df = pd.read_csv(r"C:\Users\Ashnith\Documents\01_Thesis\11. Python Code\HardwareTest\project_Lenovo_coreg_analysis\Seminar\ellipse_parameters_og.csv")

# Interpolate all ellipse parameters at 100 cm
distance_to_interpolate = 100
result = interpolate_all_ellipse_parameters(ellipse_df, distance_to_interpolate)
print(f"Ellipse parameters at {result['Distance']} cm:")
for param, value in result.items():
    print(f"  {param}: {value:.2f}")
