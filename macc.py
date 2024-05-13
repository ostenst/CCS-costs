import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Read results
experiments = pd.read_csv("all_experiments.csv", sep=",", decimal='.')
outcomes = pd.read_csv("all_outcomes.csv", sep=",", decimal='.')

class MACC_rectangle:
    def __init__(self, name):
        self.name = name
        self.experiments = None
        self.outcomes = None
        self.width = None
        self.mean = None

unique_names = experiments['Name'].unique()
rectangles = []

for name in unique_names:

    # Assign data to rectangle
    rectangle = MACC_rectangle(name)
    subset_experiments = experiments[experiments['Name'] == name]
    rectangle.experiments = subset_experiments

    subset_outcomes = outcomes[outcomes['Name'] == name]
    rectangle.outcomes = subset_outcomes

    # Calculate it's width and mean
    rectangle.width = rectangle.outcomes['mCO2'].mean() # [tCO2/yr]
    rectangle.mean  = rectangle.outcomes['cost_specific'].mean() # [tCO2/yr]

    rectangles.append(rectangle)

# Sort rectangles based on mean
sorted_rectangles = sorted(rectangles, key=lambda x: x.mean)

fig, ax = plt.subplots()
x_start = 0

norm = Normalize(vmin=experiments['duration'].min(), vmax=experiments['duration'].max())
cmap = cm.viridis

# Plot each rectangle
for rectangle in sorted_rectangles:

    x_end = x_start + rectangle.width/1000
    # for i, row in rectangle.outcomes.iterrows():
    #     color_i = cmap(norm(rectangle.experiments['duration'][i]))

    #     ax.plot([x_start, x_end], [row["cost_specific"], row["cost_specific"]], color=color_i , alpha = 1)
    for i, row in rectangle.outcomes.iterrows():
        # color_i = cmap(norm(rectangle.experiments['duration'][i]))

        ax.plot([x_start, x_end], [row["cost_specific"], row["cost_specific"]], color="black" , alpha = 0.07)

    ax.plot([x_start, x_end], [rectangle.mean, rectangle.mean], color='red')
    x_start = x_end  # Update the starting point for the next rectangle

# Set axis labels and title
ax.set_xlabel('Cumulative amount of CO2 captured [kt/yr]')
ax.set_ylabel('Specific CO2 capture cost [EUR/tCO2]')
ax.set_title('MACC for capturing CO2 at Swedish CHPs')

ax.set_xlim(0, x_start+30)  # Set x-axis limits based on the total width of all rectangles
ax.set_ylim(-30, 300)  # Set y-axis limits based on the specified height H

# Create a colorbar for the colormap
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Capacity factor [h/yr]')

plt.savefig('MACC.png')
plt.show()