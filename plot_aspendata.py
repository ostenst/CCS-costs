import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("W2E.csv", sep=";", decimal=',')
print(df.head())
columns_to_normalize = df.columns[2:]  # Exclude the first two columns

df = df.abs()

scaler = MinMaxScaler(feature_range=(0,1))
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Display the first few rows of the normalized DataFrame
print(df.head())

x_values = df.iloc[:, 1]


heat = ['Qreb', 'Qwash', 'Qstrip', 'Qlean', 'Qint2', 'Qint1', 'Qdhx', 'Qdry', 'Qrcond', 'Qrint', 'Qpreliq']
power = ['Wpumps','Wcfg','Wc1','Wc2','Wc3','Wrefr1','Wrefr2','Wrecomp']
names = heat + power

plt.figure(figsize=(8, 10))
# Plot each column as y-values against the second column as x-axis
# for column in df.columns[2:]:
#     plt.plot(x_values, df[column], label=column)
for column in heat:
    plt.plot(x_values, df[column], label=column)
plt.xlabel(df.columns[1] + " [kg/s]")  # Assuming the column has no name
plt.ylabel('Normalized Values')
plt.title('Aspen values vs Flue gas flow [kg/s], W2E @11% CO2')
plt.legend()

plt.figure(figsize=(8, 10))
for column in power:
    plt.plot(x_values, df[column], label=column)
plt.xlabel(df.columns[1] + " [kg/s]")  # Assuming the column has no name
plt.ylabel('Normalized Values')
plt.title('Aspen values vs Flue gas flow [kg/s], W2E @11% CO2')
plt.legend()

print("This shows, ish, that we can approximate heat duties linearly as a function of the flue gas flow!")
# Show plot
plt.show()