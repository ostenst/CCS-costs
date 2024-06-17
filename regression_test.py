# This file should:
# 1) Read my CHIP_data
# 2) Fit multiple regressors, one for each Y-value
# 3) Make a new estimate_size(function), which takes some X-values, and predict Y-values. Print everything!
# column names are needed, so we may need to pass the whole CHIP_data every time... posible to avoid this?


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# CHIP_data = pd.read_csv("MEA-chip.csv", sep=";", decimal=',')
# X = CHIP_data[['Flow', 'Rcapture']]
# y_columns = CHIP_data.drop(columns=['Flow', 'Rcapture'])

# regression_models = {}

# for y_column in y_columns:

#     y = CHIP_data[y_column]
#     model = LinearRegression()
#     model.fit(X, y)
#     regression_models[y_column] = model

# new_flow_value = 3
# new_rcapture_value = 78
# new_data = pd.DataFrame({'Flow': [new_flow_value], 'Rcapture': [new_rcapture_value]})

# for y_column, model in regression_models.items():
#     predicted_y = model.predict(new_data)
#     print(f"Predicted {y_column}: {predicted_y}")

import pandas as pd
from scipy import interpolate

# Load the data
CHIP_data = pd.read_csv("MEA-chip.csv", sep=";", decimal=',')

# Define the new x values
new_flow_values = [1, 30, 150]  # Example new values for 'Flow'
new_rcapture_values = [90, 90, 90]  # Example new values for 'Rcapture'

# Get the list of y values (dependent variables)
y_columns = CHIP_data.drop(columns=['Flow', 'Rcapture'])

# Create a dictionary to store interpolated y values for each column
interpolated_values = {}

# Loop through each y column and perform linear interpolation
for y_column in y_columns:
    # Define the target variable for the current y column
    y = CHIP_data[y_column]
    
    # Perform linear interpolation
    interp_func = interpolate.interp2d(CHIP_data['Flow'], CHIP_data['Rcapture'], y, kind='linear')
    
    # Store the interpolated values in the dictionary
    interpolated_values[y_column] = interp_func

# Create a new DataFrame to store the interpolated values
interpolated_data = pd.DataFrame(columns=CHIP_data.columns)

# Add new x values to the DataFrame
interpolated_data['Flow'] = new_flow_values
interpolated_data['Rcapture'] = new_rcapture_values

# Interpolate y values for each column and add them to the DataFrame
for y_column, interp_func in interpolated_values.items():
    interpolated_data[y_column] = [interp_func(flow, rcapture) for flow, rcapture in zip(new_flow_values, new_rcapture_values)]

print(interpolated_data["Qreb"], " NAJS!")

# import pandas as pd
# import numpy as np
# from scipy.interpolate import RegularGridInterpolator


# # CHIP_data = pd.read_csv("MEA-chip.csv", sep=";", decimal=',')
# # x = CHIP_data[['Flow', 'Rcapture']]
# # y_columns = CHIP_data.drop(columns=['Flow', 'Rcapture'])


# # interpolated_values = {}

# # # Create grid points for interpolation
# # flow_values = np.unique(CHIP_data['Flow'])
# # rcapture_values = np.unique(CHIP_data['Rcapture'])

# # # Loop through each y column and perform linear interpolation
# # for y_column in y_columns:
# #     # Define the target variable for the current y column
# #     y = CHIP_data[y_column]
    
# #     # Sort the values of y to match the grid defined by 'Flow' and 'Rcapture'
# #     sorted_y = y.values.reshape(len(flow_values), len(rcapture_values), order='F')
    
# #     # Create a RegularGridInterpolator
# #     interp_func = RegularGridInterpolator((flow_values, rcapture_values), sorted_y)
    
# #     # Store the interpolated values in the dictionary
# #     interpolated_values[y_column] = interp_func

# # # Now you have interpolation functions for each y value
# # # You can use these functions to predict new y values for given values of 'Flow' and 'Rcapture'
# # # For example:
# # new_flow_value = 3  # Example new value for 'Flow'. This is my lowest value in the data set!
# # new_rcapture_value = 78  # Example new value for 'Rcapture'

# # interpolated_data = pd.DataFrame(columns=CHIP_data.columns)
# # interpolated_data['Flow'] = new_flow_value
# # interpolated_data['Rcapture'] = new_rcapture_value

# # for y_column, interp_func in interpolated_values.items():
# #     interpolated_y = interp_func([new_flow_value, new_rcapture_value])
# #     interpolated_data[y_column] = interpolated_y
# #     print(f"Interpolated {y_column}: {interpolated_y}")


# # print(interpolated_data["Qreb"], " NAJS!")

# CHIP_data = pd.read_csv("MEA-chip.csv", sep=";", decimal=',')

# #TRYING ON MY OWN:
# x1 = CHIP_data["Flow"]
# x2 = CHIP_data["Rcapture"]

# # for y_column in y_columns:
# #     # Define the target variable for the current y column
# #     y = CHIP_data[y_column]
                  
# #     x1g, x2g = np.meshgrid(x1, x2, indexing='ij')

# #     interp = RegularGridInterpolator((x1, x2), y, bounds_error=False, fill_value=None)


# y = CHIP_data["Qreb"]
                  
# x1g, x2g = np.meshgrid(x1, x2, indexing='ij')

# interp = RegularGridInterpolator((x1, x2), y, bounds_error=False, fill_value=None)

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x1g.ravel(), x2g.ravel(), y.ravel(),
#            s=60, c='k', label='data')

# plt.legend()
# plt.show()