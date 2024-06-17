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

    # Calculate its width and mean
    rectangle.width = rectangle.outcomes['mCO2'].mean()  # [tCO2/yr]
    rectangle.mean  = rectangle.outcomes['cost_specific'].mean()  # [tCO2/yr]

    rectangles.append(rectangle)

sorted_rectangles = sorted(rectangles, key=lambda x: x.mean)

# BEGIN PLOTTING
fig, axs = plt.subplots(3, 1, figsize=(15, 16))  # Create 4 subplots in a 2x2 grid
axs = axs.flatten()

instructions = {}
instructions[axs[0]] = {"nlines":"single"}
instructions[axs[1]] = {"nlines":"multiple", "color":"grey", "alpha":0.1, "extremes":"yes"}
# instructions[axs[2]] = {"nlines":"multiple", "colormap":"duration"}
instructions[axs[2]] = {"nlines":"multiple", "colormap":"duration", "scenario":5200} #NOTE: MAYBE PLACE cheat HERE?

x_start = 0


# Plot each rectangle in each subplot
for ax, instruct in instructions.items():
    x_start = 0  # Reset x_start for each subplot

    if "colormap" in instruct:
        feature = instruct["colormap"]
        norm = Normalize(vmin=experiments[feature].min(), vmax=experiments[feature].max())
        cmap = cm.viridis
        
    for rectangle in sorted_rectangles:
        x_end = x_start + rectangle.width / 1000
        
        if rectangle.name in ["Vartaverket KVV 8 ","Eskilstuna KVV", "Sveg KVV"]:
            axs[0].annotate(rectangle.name, xy=((x_start+x_end)/2, rectangle.mean), xytext=(x_start, rectangle.mean-100), 
                            arrowprops=dict(facecolor='grey', arrowstyle='->', color='grey'), fontsize=13, color='grey')

        # Multiple lines?
        if instruct["nlines"] == "multiple":
            for i, row in rectangle.outcomes.iterrows():

                # Decide how to color:
                if "colormap" in instruct:
                    color_i = cmap(norm(rectangle.experiments[feature][i]))
                    alpha = 1
                else:
                    color_i = instruct["color"]
                    alpha = instruct["alpha"]

                if "scenario" in instruct:
                    if rectangle.experiments[feature][i] >= instruct["scenario"] and rectangle.experiments["cheat"][i] < 80: #NOTE: THIS IS MY PRIM SCENARIO FOUND
                        color_i = "green"
                        alpha = 0.3
                    else:
                        color_i = "grey"
                        alpha = 0.1

                ax.plot([x_start, x_end], [row["cost_specific"], row["cost_specific"]], color=color_i, alpha=alpha)

        ax.plot([x_start, x_end], [rectangle.mean, rectangle.mean], color='black')
        x_start = x_end  # Update the starting point for the next rectangle


    # Set axis labels and title for each subplot
    ax.set_ylabel('Cost of CO2 capture [EUR/tCO2]', fontsize=12)
    ax.set_xlim(0, x_start + 30)  # Set x-axis limits based on the total width of all rectangles
    ax.set_ylim(-30, 400)  # Set y-axis limits based on the specified height H
axs[2].set_xlabel('Cumulative amount of CO2 captured [kt/yr]', fontsize=12)
axs[2].annotate("Scenarios of high capacity factors and low district heating prices, found by PRIM", xy=(3700, 100), xytext=(2800, 30), 
                            arrowprops=dict(facecolor='green', arrowstyle='->', color='green'), fontsize=13, color='green')
# Set a common title for the figure
fig.suptitle('Marginal Abatement Cost Curve (MACC) for capturing CO2 at Swedish CHPs', fontsize=13)

# Comment out the colorbar for now
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=axs, orientation='horizontal')
# cbar.set_label('Capacity factor [h/yr]')

plt.savefig('MACC.png', dpi=600)
plt.show()
