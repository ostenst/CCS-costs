import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

# Read the CSV file
df = pd.read_csv("W2E.csv", sep=";", decimal=',')  # Replace "your_data.csv" with the path to your CSV file

data_columns = ['CO2%', 'Flow']  # Replace with your input column names
output_columns = list(df.columns.difference(data_columns)) # Everything else is an output value!

data = df[data_columns]
output_data = df[output_columns]


### Create X and y from DataFrames
X = data.values
y = output_data.values

# Create a multi-output regression model using linear regression as the base regressor
model = MultiOutputRegressor(LinearRegression())
model.fit(X, y)

# Predict new output given new values of CO2% and Flow
new_data = pd.DataFrame({'CO2%': [11], 'Flow': [109]})  # New data point
predicted_y = model.predict(new_data)

# Convert predicted_y to a DataFrame with appropriate column names
predicted_df = pd.DataFrame(predicted_y, columns=output_data.columns)

# Now you can access predicted values by column names
interesting_output = predicted_df['Qreb']
print(interesting_output)

# Display the first few rows of data and output_data
print("Input data:")
print(data.head())
print("\nOutput data:")
print(output_data.head())

# Make a line plot by plotting output data against "Flow". Make a separate line for each of these outputs: Qreb, Qwash, Tinwash, Qtoabs, Tintoabs, Qliq, COP

# Specify the columns to plot
columns_to_plot = ['Qreb', 'Qwash', 'Tinwash', 'Qtoabs', 'Tintoabs', 'Qliq', 'COP']

# Make a line plot for each specified output column against "Flow"
for output_column in columns_to_plot:
    plt.plot(df['Flow'], df[output_column], label=output_column)
    plt.scatter(new_data["Flow"], predicted_df[output_column], color='red')

# Set labels and title
plt.xlabel('Flue gas flow [kg/s] (Renova@109kg/s)')
plt.ylabel('Output Value')
plt.title('W2E(11 percent CO2) results vs Flow')

# Add legend
plt.legend()

# Show plot
plt.show()