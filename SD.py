import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ema_workbench.analysis import prim

# Perform SD:
# Select subset of moderate size, i.e. where capture cost is below 250EUR/t:

# Read results
experiments = pd.read_csv("all_experiments.csv", sep=",", decimal='.')
outcomes = pd.read_csv("all_outcomes.csv", sep=",", decimal='.')

unique_names = experiments['Name'].unique()
scenario_subset = pd.DataFrame()

for name in unique_names:
    # Assign data to rectangle
    subset = outcomes[outcomes['Name'] == name]
    subset_mean = subset["cost_specific"].mean()

    # Append the subset if it has a mean below 250EUR/t
    if subset_mean < 250:
        scenario_subset = pd.concat( [scenario_subset, experiments[experiments['Name'] == name] ], ignore_index=True)

print(scenario_subset.shape)
print(scenario_subset.head())



x = experiments.iloc[:, 0:22]
y = outcomes["cost_specific"] < 180
# y = data.iloc[:, 15].values
prim_alg = prim.Prim(x, y, threshold=0.8, peel_alpha=0.1)
box1 = prim_alg.find_box()
box1.show_tradeoff()
box1.inspect(4, style="graph")
box1.inspect(8, style="graph")
box1.inspect(16, style="graph")

plt.show()