import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ema_workbench.analysis import regional_sa
from itertools import combinations
from matplotlib.colors import ListedColormap
# from ema_workbench.analysis import prim
from prim_constrained import PrimedData, Box, prim_recursive

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
    if subset_mean < 220:
        scenario_subset = pd.concat( [scenario_subset, experiments[experiments['Name'] == name] ], ignore_index=True)

print(scenario_subset.shape)
print(scenario_subset.head())

x = experiments.iloc[:, 0:22]
y = outcomes["cost_specific"] < 120
y = y.astype(int)

# Pre-made PRIM:
# y = data.iloc[:, 15].values
# prim_alg = prim.Prim(x, y, threshold=0.8, peel_alpha=0.1)
# box1 = prim_alg.find_box()
# box1.show_tradeoff()
# box1.inspect(4, style="graph")
# box1.inspect(8, style="graph")
# box1.inspect(16, style="graph")

# Create a list of the most important features, which we can constrain PRIM to:
important_features = ["duration", "fCO2_B", "cheat", "i"]
feature_combinations = [list(combination) for r in range(1, len(important_features) + 1)
                        for combination in combinations(important_features, r)]

# # My own PRIM:
# # fig = regional_sa.plot_cdfs(x, y)
Data = PrimedData(x, y)
# peeling_trajectory = []
# box = Box(id=0)
# box.calculate(Data)
# peeling_trajectory = prim_recursive(Data, box, peeling_trajectory, max_iterations=40, constrained_to=["duration", "fCO2_B", "cheat", "i"], objective_function="LENIENT1")
# peeling_trajectory[3].print_info()
# peeling_trajectory[9].print_info()
# peeling_trajectory[24].print_info()

# # Plotting PRIM trajectory
# x_values = [box.coverage for box in peeling_trajectory]
# y_values = [box.density for box in peeling_trajectory]
# colors = [box.n_lims for box in peeling_trajectory]
# colors = np.array(colors, dtype=int)
# num_colors = len(set(colors))
# cmap = plt.cm.get_cmap('cool', num_colors)
# plt.scatter(x_values, y_values, c=colors, cmap=cmap, alpha=0.8, label='Algorithm 1')
# plt.xlabel('Coverage')
# plt.ylabel('Density')
# plt.xlim(0, 1.2)
# plt.ylim(0, 1.2)
# plt.colorbar(label='Number of Constrained dimensions')

# plt.show()
# Create a new plot object
plt.figure(figsize=(10, 8))

# Define a list of markers for each combination
markers = ['D', '+', '.', 'o', '*', 's', '^', 'v', '<', '>']
custom_colors = ['limegreen', 'aquamarine', 'deepskyblue', 'lightsteelblue', 'mediumpurple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# Plot peeling_trajectories for each feature combination
for idx, combination in enumerate(feature_combinations):
    peeling_trajectory = []
    box = Box(id=idx)
    box.calculate(Data)
    peeling_trajectory = prim_recursive(Data, box, peeling_trajectory, max_iterations=40, constrained_to=combination, objective_function="LENIENT2")

    # Plotting results of algorithm version 1 onto the existing plot
    x_values = [box.coverage for box in peeling_trajectory]
    y_values = [box.density for box in peeling_trajectory]
    n_lims_values = [box.n_lims for box in peeling_trajectory]

    # Create a unique colormap for each combination based on the unique n_lims values
    unique_n_lims = sorted(set(n_lims_values))
    num_colors = len(unique_n_lims)
    cmap = ListedColormap(custom_colors[:num_colors])


    # Assign a unique marker style for each combination
    marker = markers[idx % len(markers)]

    # Scatter plot with color and marker based on n_lims
    plt.scatter(x_values, y_values, c=n_lims_values, cmap=cmap, alpha=0.8, label=', '.join(combination), marker=marker)

plt.xlabel('Coverage')
plt.ylabel('Density')
plt.xlim(0, 1.3)
plt.ylim(0, 1.3)

# Show legend with black markers
legend = plt.legend()
for handle in legend.legendHandles:
    handle.set_edgecolor('black')

# Add colorbar to indicate the n_lims values
cbar = plt.colorbar(ticks=unique_n_lims)
cbar.set_label('n_lims')

plt.show()