import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator

def interpolate_new_values(CHIP_data, new_Flow, new_Rcapture):
    # Extract x and y values
    x1 = CHIP_data['Flow']
    x2 = CHIP_data['Rcapture']

    # Extract y values (excluding first two columns which are x values)
    y_values = CHIP_data.iloc[:, 2:].values

    # Create a 2D array for x values
    x_values = np.column_stack((x1, x2))

    # Create an empty dictionary to store interpolation functions for each y column
    interpolations = {}

    # Loop through each y column
    for idx, column_name in enumerate(CHIP_data.columns[2:]):
        # Extract y values for the current column
        y = y_values[:, idx]

        # Construct LinearNDInterpolator for each y column
        interp_func = LinearNDInterpolator(x_values, y)
        interpolations[column_name] = interp_func




    # Initialize a dictionary to store new y values for each column
    new_y_values = {}

    # Calculate new y values for each column using interpolation functions
    for column_name, interp_func in interpolations.items():
        new_y = interp_func((new_Flow, new_Rcapture))
        new_y_values[column_name] = new_y

    # Create a DataFrame to store the new values
    new_data = pd.DataFrame({
        'Flow': new_Flow,
        'Rcapture': new_Rcapture,
        **new_y_values  # Unpack new y values dictionary
    })

    return new_data

CHIP_data = pd.read_csv("MEA-chip.csv", sep=";", decimal=',')

# Example usage:
new_Flow = [10]
new_Rcapture = [80]
new_values = interpolate_new_values(CHIP_data, new_Flow, new_Rcapture)
print(new_values)
