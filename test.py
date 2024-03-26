import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("W2E.csv", sep=";", decimal=',')

# Set input features (X) and output targets (y)
X = df[['CO2%', 'Flow']]
y = df.drop(columns=['CO2%', 'Flow'])

# Create multi-output regression model using linear regression as the base regressor
model = MultiOutputRegressor(LinearRegression())

# Fit the model
model.fit(X, y)

# Predict new output given new values of CO2% and Flow
new_data = pd.DataFrame({'CO2%': [11], 'Flow': [100]})
predicted_y = model.predict(new_data)

# Convert predicted_y to a DataFrame with appropriate column names
predicted_df = pd.DataFrame(predicted_y, columns=y.columns)

# Print the predicted results
print("Predicted values for new example:")
print(predicted_df.head())



columns_to_plot = ['Qreb', 'Qwash', 'Tinwash', 'Qtoabs', 'Tintoabs', 'Qliq', 'COP']

# Make a line plot for each specified output column against "Flow"
for output_column in columns_to_plot:
    plt.plot(df['Flow'], df[output_column], label=output_column)
    plt.scatter(new_data["Flow"], predicted_df[output_column], color='red')

# Set labels and title
plt.xlabel('Flue gas flow [kg/s]')
plt.ylabel('Output Value')
plt.title('W2E results MEA vs flow')

# Add legend
plt.legend()

# Show plot
plt.show()