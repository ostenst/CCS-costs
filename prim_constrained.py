"""
Constrained PRIM algorithm for interpretable scenario discovery.

This implementation takes the non-constrained uncertainty dimensions as an argument,
which allows for better control of the peeling trajectories analyzed.

Author: Oscar Stenström
Date: 2023-12-30
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

class PrimedData:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.y_name = y.name
        self.data = pd.concat([x, y], axis=1)
        self.limits = {}
        self.find_limits()

    def find_limits(self):
        for feature, values in self.x.items():
            if self.is_numeric(self.x[feature]):
                min_value = values.min()
                max_value = values.max()
                self.limits[feature] = {'min': min_value, 'max': max_value, 'slicesz': abs(max_value-min_value)*0.05}
            elif self.is_categorical(self.x[feature]):
                categories = values.unique()
                self.limits[feature] = {'categories': categories}
    
    def is_numeric(self,column):
        return pd.api.types.is_numeric_dtype(column)
    def is_categorical(self,column):
        return column.dtype == 'O'


class Box:
    def __init__(self, id=None, lims={}, coverage=None, density=None, mass=None, mean=None):
        self.id = id
        self.lims = lims
        self.n_lims = len(self.lims)
        self.coverage = coverage
        self.density = density
        self.mass = mass
        self.mean = mean

    def print_info(self):
        print(f"Box ID: {self.id}")
        print("Limits:")
        print("{:<20}".format("Feature:"), end="")
        all_keys = set(key for limits in self.lims.values() for key in limits.keys())
        for key in all_keys:
            print("{:<15}".format(key), end="")
        print()
        for feature, limits in self.lims.items():
            print("{:<20}".format(feature), end="")  
            for key in all_keys:
                value = limits.get(key, "N/A")    
                if isinstance(value, list):
                    formatted_value = ", ".join(map(lambda x: f"{float(x):.2f}" if x.replace('.', '', 1).isdigit() else str(x), value))
                elif isinstance(value, (int, float)):
                    formatted_value = f"{float(value):.2f}"
                else:
                    formatted_value = str(value)   
                print("{:<15}".format(formatted_value), end="")  
            print()
        print(f"Coverage: {float(self.coverage):.2f}")
        print(f"Density: {float(self.density):.2f}")
        print()


    def calculate(self, Data):
        # Find the data subset of this box by filtering on lims
        df = Data.data
        num_mask = pd.Series(True, index=df.index)
        for feature, values in self.lims.items():
            if 'min' in values:
                num_mask &= (df[feature] >= values['min'])
            if 'max' in values:
                num_mask &= (df[feature] <= values['max'])
        cat_mask = pd.Series(True, index=df.index)
        for feature, values in self.lims.items():
            if 'categories' in values:
                if isinstance(values['categories'], list):
                    cat_mask &= df[feature].isin(values['categories'])
                else:
                    cat_mask &= df[feature].isin([values['categories']])
        final_mask = num_mask & cat_mask
        subset = df[final_mask]

        # Calculate properties
        cases_of_interest = df[df[Data.y_name] == 1]
        subset_of_interest = subset[subset[Data.y_name] == 1]

        self.coverage = len(subset_of_interest) / len(cases_of_interest)
        if len(subset) != 0:
            self.density = len(subset_of_interest) / len(subset)
        else:
            self.density = 0
        self.mass = len(subset)
        self.mean = self.density

def prim_recursive(Data, box, peeling_trajectory, max_iterations=100, constrained_to=None, objective_function="LENIENT1"): 

    peeling_trajectory.append(box)
    count = len(peeling_trajectory)

    # If unconstrained, consider all x-features
    if constrained_to == None:
        constrained_to = list(Data.x.columns)

    # Produce candidate boxes
    box_candidates = []
    for feature,values in Data.x.items(): 

        if feature not in constrained_to:
            continue

        if Data.is_numeric(Data.x[feature]):
            #Get the limits based on previous box.
            if feature in box.lims:
                low_dict = copy.deepcopy(box.lims)
                high_dict = copy.deepcopy(box.lims)

                lims_low = box.lims[feature]["min"] + Data.limits[feature]["slicesz"]
                lims_high = box.lims[feature]["max"] - Data.limits[feature]["slicesz"]

                low_dict[feature] = {"min":lims_low,"max":box.lims[feature]["max"]}
                high_dict[feature] = {"min":box.lims[feature]["min"],"max":lims_high}

            if feature not in box.lims:
                low_dict = copy.deepcopy(box.lims)
                high_dict = copy.deepcopy(box.lims)
                
                lims_low = Data.limits[feature]["min"] + Data.limits[feature]["slicesz"]
                lims_high = Data.limits[feature]["max"] - Data.limits[feature]["slicesz"]
                
                low_dict[feature] = {"min":lims_low,"max":Data.limits[feature]["max"]}
                high_dict[feature] = {"min":Data.limits[feature]["min"],"max":lims_high}
                
            box_low = Box(id=count, lims=low_dict)
            box_high = Box(id=count, lims=high_dict)
            box_low.calculate(Data)
            box_high.calculate(Data)
            box_candidates.append(box_low)
            box_candidates.append(box_high)
        
        if Data.is_categorical(Data.x[feature]):
            lims_dict = copy.deepcopy(box.lims)
            #If previous box is restricted in all categories, I should add all categories. So I later can remove them...
            if feature not in lims_dict: #I think ths only occurs for the first box? 
                lims_dict[feature] = {"categories":[]}
                for category in Data.limits[feature]['categories']:
                    # print("Storing", category, " in new box dict")
                    lims_dict[feature]["categories"].append(category)

            # At this point, the lims_dict will always have 1-3 categories in this particular feature. I think.
            for category in lims_dict[feature]["categories"]:
                limsi_dict = copy.deepcopy(lims_dict)
                limsi_dict[feature]["categories"].remove(category)
                boxi = box_low = Box(id=count, lims=limsi_dict)
                boxi.calculate(Data)
                box_candidates.append(boxi)

    # Candidate boxes will be evaluated against chosen criterion
    def criterion(candidate, box):
        if objective_function == "LENIENT1":
            return                (candidate.density - box.density) / (box.mass - candidate.mass)
        if objective_function == "LENIENT2":
            return candidate.mass*(candidate.density - box.density) / (box.mass - candidate.mass)
        if objective_function == "ORIGINAL": #TODO: Buggy, produces boxes with decreasing densities. Check what candidates are produced and chosen below!
            return                (candidate.density - box.density)

    # Update to the box that maximizes the chosen objective
    objective = []
    for candidate in box_candidates:
        if candidate.mass < box.mass:
            objective.append( criterion(candidate,box) )
    max_index = np.argmax(objective)
    box = box_candidates[max_index]
    # print("Best box of iteration", max_iterations)
    # box.print_info()

    # Stop if not improving objective
    if box.density==1 or max(objective)<=0 or max_iterations==0:
        return peeling_trajectory

    return prim_recursive(Data, box, peeling_trajectory, max_iterations-1, constrained_to=constrained_to, objective_function=objective_function)

def main():
    # Below is an example if constrained PRIM usage
    data = pd.read_csv("test.csv", index_col=False)
    x = data.iloc[:, 1:11]
    x.iloc[:, 0] = "FOAK"
    x.iloc[3:100, 0] = "NOAK" 
    x.iloc[100:200, 0] = "skoa" 
    y = data.iloc[:, 15]

    Data = PrimedData(x,y)
    peeling_trajectory = []
    box = Box(id=0)
    box.calculate(Data)

    peeling_trajectory = prim_recursive(Data,box,peeling_trajectory,max_iterations=40, constrained_to=None, objective_function="LENIENT2")
    # peeling_trajectory = prim_recursive(Data,box,peeling_trajectory,max_iterations=40, constrained_to=["Cellulosic cost", "Biomass backstop price", "Pricing"], objective_function="LENIENT2")
    peeling_trajectory[20].print_info()

    # Plotting
    x_values = [box.coverage for box in peeling_trajectory]
    y_values = [box.density for box in peeling_trajectory]
    colors = [box.n_lims for box in peeling_trajectory]
    colors = np.array(colors, dtype=int)
    num_colors = len(set(colors))
    cmap = plt.cm.get_cmap('tab10', num_colors)
    plt.scatter(x_values, y_values, c=colors, cmap=cmap, alpha=0.8)
    plt.xlabel('Coverage')
    plt.ylabel('Density')
    plt.colorbar(label='Number of Limits')
    plt.show()

if __name__ == "__main__":
    main()