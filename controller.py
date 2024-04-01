"""
My main controller of the study!

"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import seaborn as sns
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot separately
from functions import *
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from SALib.analyze import sobol
from ema_workbench import Samplers
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from ema_workbench.analysis import feature_scoring

from ema_workbench import (
    Model,
    RealParameter,
    IntegerParameter,
    ScalarOutcome,
    Constant,
    ema_logging,
    perform_experiments
)
from ema_workbench.em_framework.evaluators import Samplers
from model import CCS_CHP



if __name__ == "__main__":

    # Read data and fit regressors
    plant_data = pd.read_csv("plant_data.csv", sep=";", decimal=',')
    print(plant_data.head())

    W2E_data = pd.read_csv("W2E.csv", sep=";", decimal=',')
    X = W2E_data[['CO2%', 'Flow']]
    y = W2E_data.drop(columns=['CO2%', 'Flow'])

    W2E_regression = MultiOutputRegressor(LinearRegression())
    W2E_regression.fit(X, y)

    # Iterate over the CHPs and call RDM model
    for index, row in plant_data.iterrows():

        CHP = CHP_plant(
                name=row["Plant Name"],
                fuel=row["Fuel (W=waste, B=biomass)"],
                Qdh=float(row["Heat output (MWheat)"]),
                P=float(row["Electric output (MWe)"]),
                Qfgc=float(row["Existing FGC heat output (MWheat)"]),
                ybirth=float(row["Year of commissioning"]),
                Tsteam=float(row["Live steam temperature (degC)"]),
                psteam=float(row["Live steam pressure (bar)"])
            )
        CHP.estimate_rankine()

        model = Model("CCSproblem", function=CCS_CHP)       
        model.uncertainties = [
            RealParameter("eta_boiler", 0.783, 0.957),
            RealParameter("fCO2_B", 0.144, 0.176),
            RealParameter("fCO2_W", 0.099, 0.121),
            RealParameter("dTreb", 9, 11),
            RealParameter("Tsupp", 77.4, 94.6),
            RealParameter("Tlow", 34.2, 41.8),
            RealParameter("dTmin", 6.3, 7.7),

            RealParameter("alpha", 5.508, 6.732),
            # RealParameter("beta", 0.57, 0.700),
            RealParameter("CEPCI", 1.0909, 1.333),
            RealParameter("fixed", 0.054, 0.066),
            RealParameter("ownercost", 0.18, 0.22),
            RealParameter("WACC", 0.045, 0.055),
            IntegerParameter("yexpenses", 2, 4),
            RealParameter("rescalation", 0.027, 0.033),
            RealParameter("i", 0.0675, 0.0825),
            IntegerParameter("t", 23, 27),
            RealParameter("celc", 36, 44),
            RealParameter("cheat", 13.5, 16.5),
            RealParameter("duration", 7200, 8800),
            RealParameter("cMEA", 1.8, 2.2),
        ]
        model.levers = [
            # RealParameter("i", 0.05, 0.10),
            RealParameter("beta", 0.57, 0.700),
        ]
        model.outcomes = [
            ScalarOutcome("cost_specific", ScalarOutcome.MINIMIZE),
            ScalarOutcome("consumer_cost", ScalarOutcome.MINIMIZE),
            ScalarOutcome("fuel_penalty", ScalarOutcome.MINIMIZE),
        ]
        model.constants = [
            Constant("CHP", CHP), # Overwrite the default CHP=None value
            Constant("W2E_regression", W2E_regression),
            Constant("W2E_data", W2E_data),
        ]

        # Perform the experiments (check Sobol requirement for the number of scenarios)
        print(f"Exploring outcomes of implementing CCS at {CHP.name}:")
        n_scenarios = 100
        n_policies = 4

        results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
        experiments, outcomes = results
        CHP.plot_plant()
    
        # Convert experiments to a DataFrame
        df_experiments = pd.DataFrame(experiments)
        df_experiments.to_csv("experiments.csv", index=False)

        data = pd.DataFrame(outcomes)
        data.to_csv("outcomes.csv", index=False)

        # data["policy"] = experiments["policy"] #add the policy-information of my experiments, to the outcomes
        # sns.pairplot(data, hue="policy", vars=list(outcomes.keys()))
        data["beta"] = experiments["beta"] #add the policy-information of my experiments, to the outcomes
        sns.pairplot(data, hue="beta", vars=list(outcomes.keys()))

        # Perform SA
        print(f"\nPerforming Sobol SA for {CHP.name}:")
        plt.figure(figsize=(6, 70))
        x = experiments
        y = outcomes
        fs = feature_scoring.get_feature_scores_all(x, y)
        sns.heatmap(fs, cmap="viridis", annot=True)

        # sobol_stats, s2, s2_conf = analyze(model, results, "cost_specific")
        # print(sobol_stats.head(20))
        sa_results = perform_experiments(model, scenarios=100, uncertainty_sampling=Samplers.SOBOL)
        experiments, outcomes = sa_results

        problem = get_SALib_problem(model.uncertainties)
        Si = sobol.analyze(problem, outcomes["cost_specific"], calc_second_order=True, print_to_console=False)
        scores_filtered = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
        Si_df = pd.DataFrame(scores_filtered, index=problem["names"])

        sns.set_style("white")
        fig, ax = plt.subplots(1)

        indices = Si_df[["S1", "ST"]]
        err = Si_df[["S1_conf", "ST_conf"]]

        indices.plot.bar(yerr=err.values.T, ax=ax, colormap='viridis')
        fig.set_size_inches(8, 6)
        fig.subplots_adjust(bottom=0.3)

plt.show()