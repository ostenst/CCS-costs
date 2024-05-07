import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df_flowrate = pd.read_csv("W2E.csv", sep=";", decimal=',')
df_Rcapture = pd.read_csv("CHP.csv", sep=";", decimal=',')
# print(df.head())
# columns_to_normalize = df.columns[3:]  # Exclude the first two columns, or three!

# df = df.abs()

# scaler = MinMaxScaler(feature_range=(0,1))
# df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# # Display the first few rows of the normalized DataFrame
# print(df.head())


heat = ['Qreb', 'Qwash', 'Qstrip', 'Qlean', 'Qint2', 'Qint1', 'Qdhx', 'Qdry', 'Qrcond', 'Qrint', 'Qpreliq']
power = ['Wpumps','Wcfg','Wc1','Wc2','Wc3','Wrefr1','Wrefr2','Wrecomp']
names = heat + power

for i, df in enumerate([df_Rcapture, df_flowrate]):
    x_values = df.iloc[:,i]

    plt.figure(figsize=(8, 10))
    for column in heat:
        plt.plot(x_values, df[column], label=column)
    plt.xlabel(df.columns[i])  # Assuming the column has no name
    plt.ylabel('Energy [kW]')
    plt.title('Aspen values vs ' + df.columns[i]) 
    plt.legend()

    plt.figure(figsize=(8, 10))
    for column in power:
        plt.plot(x_values, df[column], label=column)
    plt.xlabel(df.columns[i])  # Assuming the column has no name
    plt.ylabel('Energy [kW]')
    plt.title('Aspen values vs ' + df.columns[i]) 
    plt.legend()

print("This shows, ish, that we can approximate heat duties linearly as a function of the flue gas flow!")
# Show plot
plt.show()