"Testing K-prototypes new version, using modes instead of categorical proportions"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors
from Kfunctions import *


# Read results
experiments = pd.read_csv("all_experiments.csv", sep=",", decimal='.')
outcomes = pd.read_csv("all_outcomes.csv", sep=",", decimal='.')
data = pd.concat([experiments.drop(columns=['Name']), outcomes], axis=1)

# Apply interesting label
data['Label'] = np.where((data['cost_specific'] < 150) | (data['fuel_penalty'] < 0.23), 'Interesting', 'Uninteresting')

# NORMALIZE NUMERICAL FEATURES
data, scalers = normalize_numerical_features(data)
print("Normalized Data head:\n", data.head())

# INITIALIZE clusters (requires knowledge of the features to consider)
clusters = []
# clusters.append(Cluster(c_num=[0.4]*22, c_cat={"Label": {"Uninteresting": 0}}, name="C1")) # These first centroids need to match everything else!
# clusters.append(Cluster(c_num=[0.6]*22, c_cat={"Label": {"Interesting": 0}}, name="C2"))
clusters.append(Cluster(c_num=[0.4]*5, c_cat={}, name="C1")) # These first centroids need to match everything else!
clusters.append(Cluster(c_num=[0.6]*5, c_cat={}, name="C2"))

all_columns = data.columns.tolist()
# outcome_columns = outcomes.columns.tolist()

ignore_columns = all_columns
ignore_columns.append("Cluster")
for column in ["duration","cheat","i","celc","fCO2_B"]:
    ignore_columns.remove(column)

# ignore_columns = ["scenario","policy","model","Cluster","Label"]
# ignore_columns += outcome_columns

# ASSIGN EXAMPLES TO clusters, AND ITERATE
data, clusters = cluster_dataframe(data, clusters, iterations=10, interesting_weight=1, categorical_weight=0.5, ignore=ignore_columns)


# Plot originally scaled data
data = revert_normalized(data, scalers)
print(" ")
print("Reverted Data head:\n", data.head())

print("... Issue: the centroids are very similar. They do not want to separate the interesting cases... Unsure how to resolve.")