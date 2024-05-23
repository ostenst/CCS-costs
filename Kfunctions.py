import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors

def housing_data(n_examples = 0, interesting_threshold = 600000):
    np.random.seed(42)
    area = np.random.uniform(15, 150, n_examples)

    locations = np.random.choice(['A', 'B', 'C', 'D', 'E'], size=n_examples)
    locations.sort()

    perturbation = np.random.normal(loc=0, scale=0.05, size=n_examples)
    locations_perturbed = np.array([ord(loc) + p for loc, p in zip(locations, perturbation)])

    def calculate_housing_price(area, location):
        base_price = 5000  # Base price per square meter
        location_multiplier = {'A': 1.2, 'B': 0.8, 'C': 0.8, 'D': 1.2, 'E': 0.8}
        randomness_factor = np.random.uniform(0.8, 1.2)  # Adjust the range based on desired randomness
        return base_price * area * location_multiplier[location] * randomness_factor

    housing_prices = [calculate_housing_price(a, loc) for a, loc in zip(area, locations)]

    data = pd.DataFrame({'Area': area, 'Location': locations, 'Location_perturbed': locations_perturbed, 'HousingPrice': housing_prices})
    data['Label'] = np.where(data['HousingPrice'] > interesting_threshold, 'Interesting', 'Uninteresting')
    return data

def plot_housing_data(data, show = False):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    scatter1 = axs[0].scatter(data['Area'], data['Location_perturbed'], c=data['HousingPrice'], cmap='viridis', edgecolors='k', s=100)
    axs[0].set_xlabel('Area (Square Meters)')
    axs[0].set_yticks(np.arange(ord('A'), ord('F')))
    axs[0].set_yticklabels(['A', 'B', 'C', 'D', 'E'])
    axs[0].set_ylabel('Location (Perturbed)')
    axs[0].set_title('Housing Prices vs Mixed Features')
    fig.colorbar(scatter1, ax=axs[0], label='Housing Price')

    scatter2 = axs[1].scatter(data['Area'], data['Location_perturbed'], c=data['Label'].map({'Interesting': 1, 'Uninteresting': 0}),
                            cmap='viridis', edgecolors='k', s=100)
    axs[1].set_xlabel('Area (Square Meters)')
    axs[1].set_yticks(np.arange(ord('A'), ord('F')))
    axs[1].set_yticklabels(['A', 'B', 'C', 'D', 'E'])
    axs[1].set_ylabel('Location (Perturbed)')
    axs[1].set_title('Housing Prices Labeled')
    fig.colorbar(scatter2, ax=axs[1], ticks=[0, 1], label='Label (0: Uninteresting, 1: Interesting)')

    plt.tight_layout()
    if show == True:
        plt.show()

class Cluster:
    def __init__(self, c_num=None, c_cat=None, name=None):
        self.name = name
        self.examples = None
        self.c_num = c_num
        self.c_cat = c_cat
        self.size = 0
        self.updated = False

    def add_example(self, example):
        # Adding example to cluster
        if self.examples is None:
            self.examples = pd.DataFrame([example], columns=example.keys())
        else:
            self.examples = pd.concat([self.examples, pd.DataFrame([example])], ignore_index=True)
        # self.size += 1 # NO! It's better to determine size in update_centroids, because then we can weigh each example

    def update_centroids(self, interesting_weight, ignore):
        # NOTE: we assume that update_centroids() is called after examples have been added, otherwise this reset causes bugs:
        self.updated = True
        # if self.size == 0:
        #     print("--- Cluster has no examples! ---")
        #     return
        c_num_old = self.c_num
        self.c_num = None
        for index, example in self.examples.iterrows():
            # # CHECK IF INTERESTING, GIVE MORE WEIGHT... but how do we "give more weight" to examples, so that when summarized, the centroids still remain normalized???
            # if example['Label'] in ['Interesting', 1, 'True']:
            #     distance_combined = distance_combined * interesting_weight
            # else:
            #     distance_combined = distance_combined * (1-interesting_weight)

            # Identifying the numerical and categorical features to consider when updating centroids
            x_num, x_cat = self.distinguish_features(example, ignore=ignore) #TODO: Move to before loop, for faster model runs.

            if example['Label'] in ['Interesting', 1, 'True']:
                x_num *= interesting_weight
                cat_frequency = interesting_weight
                self.size += interesting_weight
            else:
                x_num *= (1-interesting_weight)
                cat_frequency = (1-interesting_weight)
                self.size += (1-interesting_weight)

            # Adding features to numerical centroid #TODO: add more weight if interesting? Maybe: 1 x_num vector for interesting , one for uninteresting. FInd average!
            if self.c_num is None: 
                self.c_num = x_num
            else:
                self.c_num += x_num
 
            # Adding features to categorical centroid. Nested dict keeps track of frequency of categories in this cluster.
            # if frequencies are counted in natural numbers:
            # if self.c_cat is None:
            #     self.c_cat = {key: {value: 1} for key, value in x_cat.items()}
            # else:
            #     for key, value in x_cat.items():
            #         if key in self.c_cat:
            #             if value in self.c_cat[key]:
            #                 self.c_cat[key][value] += 1
            #             else:
            #                 self.c_cat[key][value] = 1
            #         else:
            #             self.c_cat[key] = {value: 1}
                
            # if frequencies are counted in fractions based on weights:
                
            
            if self.c_cat is None:
                self.c_cat = {key: {value: cat_frequency} for key, value in x_cat.items()}
            else:
                for key, value in x_cat.items():
                    if key in self.c_cat:
                        if value in self.c_cat[key]:
                            self.c_cat[key][value] += cat_frequency
                        else:
                            self.c_cat[key][value] = cat_frequency
                    else:
                        self.c_cat[key] = {value: cat_frequency}

        # Finalizing the centroids:
        # if self.size == 0:
        #     # raise ValueError("Cluster size is zero, cannot finalize centroids.")
        #     print("Size 0 ???")
        #     return
        # else:
        #     self.c_num /= self.size
        #     print("FINAL", self.c_num)
        #     # Set c_cat to equal a dictionary that contains all keys but only the categories that are most frequent
        #     for key, value_freq_dict in self.c_cat.items():
        #         most_frequent_value = max(value_freq_dict, key=value_freq_dict.get)
        #         self.c_cat[key] = {most_frequent_value: value_freq_dict[most_frequent_value]}
        if self.size != 0:
            self.c_num /= self.size
        else:
            self.c_num = c_num_old # Reset this centroid if it contains no examples

            # Set c_cat to equal a dictionary that contains all keys but only the categories that are most frequent
        for key, value_freq_dict in self.c_cat.items():
            most_frequent_value = max(value_freq_dict, key=value_freq_dict.get)
            self.c_cat[key] = {most_frequent_value: value_freq_dict[most_frequent_value]}
        
    def distinguish_features(self, example, ignore=[]): #Consider moving to CentroidClass()
        example_filtered = {key: value for key, value in example.items() if key not in ignore}
        x_num = np.array([example_filtered[key] for key, value in example_filtered.items() if np.issubdtype(type(value), np.number)])
        x_cat = {key: value for key, value in example_filtered.items() if not np.issubdtype(type(value), np.number)}

        return x_num, x_cat

    def reset(self):
        # Recursively loop through self.c_cat and set all frequencies to 0
        def reset_frequency(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    reset_frequency(value)  # Recursive call for nested dictionaries
                else:
                    d[key] = 0

        if self.c_cat is not None:
            reset_frequency(self.c_cat)

        self.examples = None
        self.size = 0
        self.updated = False

    def __str__(self):
        return f"Numerical Centroid: {self.c_num}, Categorical Centroid (Modes): {self.c_cat}"

    def calculate_distances(self, example, categorical_weight, ignore): 
        # Identifying the numerical and categorical features to consider when calculating distance
        x_num, x_cat = self.distinguish_features(example, ignore=ignore)

        # Numerical distance, normalized
        distance_num = np.linalg.norm(x_num - self.c_num)

        # Categorical distance (Jaccard distance)
        flattened_c_cat = self.transform_centroid_dict(self.c_cat)
        distance_cat = self.calculate_jaccard_distance(set(x_cat.items()), set(flattened_c_cat.items()))

        distance_combined = (1-categorical_weight)*distance_num + categorical_weight*distance_cat
        return distance_combined

    def calculate_jaccard_distance(self, set1, set2):
        # Jaccard distance between two sets
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_distance = 1.0 - intersection / union if union > 0 else 0.0
        return jaccard_distance

    def transform_centroid_dict(self,c_cat):
        transformed_dict = {}
        
        for key, value in c_cat.items():
            if isinstance(value, dict):
                # If the value is another dictionary, get the first key and use it as the value
                transformed_dict[key] = list(value.keys())[0]
            else:
                # If the value is not a dictionary, use the value as is
                transformed_dict[key] = value
        
        return transformed_dict

# Function to normalize numerical features
def normalize_numerical_features(data):
    scalers = []

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            scaler = MinMaxScaler()
            data[column] = scaler.fit_transform(data[[column]])
            scalers.append(scaler)

    return data, scalers

# Function to revert data to original scale using the scaler
def revert_normalized(data, scalers):
    i = 0
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = scalers[i].inverse_transform(data[[column]])
            i+=1

    return data

def cluster_dataframe(data, clusters, iterations=999, interesting_weight=0.5, categorical_weight=0.5, ignore=["Cluster"]):
    for i in range(0,iterations):

        min_distance = 999 
        for _, example in data.iterrows():
            
            # Compare example to each centroid
            for cluster in clusters:
                distance_combined = cluster.calculate_distances(example, categorical_weight, ignore)

                if distance_combined < min_distance:
                    min_distance = distance_combined
                    min_cluster = cluster

            min_cluster.add_example(example)
            data.at[_, 'Cluster'] = min_cluster.name
    
        # UPDATE EACH CENTROID BASED ON ITS CLUSTER
        print("Iteration =", i+1, ", we update the centroids to: ")
        for cluster in clusters:
            cluster.update_centroids(interesting_weight, ignore)
            print(cluster)
            cluster.reset()
            
    return data, clusters