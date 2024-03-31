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
    # ema_logging.log_to_stderr(ema_logging.INFO)
    # # NOTE: I should construct a unique function= for each plant. This could be done by importing my model(), and then below I just modify it slightly to make it unique:
    # # CCS_function = CCS_function(uniqueCHP)
    # # then send this CCS_function to the RDMModel()

    # # instantiate the model
    # lake_model = Model("lakeproblem", function=lake_problem)

    # Read data and fit regressors
    plant_data = pd.read_csv("plant_data_test.csv", sep=";", decimal=',')
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
        CHP.plot_plant()

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
            RealParameter("beta", 0.57, 0.700),
            RealParameter("CEPCI", 1.0909, 1.333),
            RealParameter("fixed", 0.054, 0.066),
            RealParameter("ownercost", 0.18, 0.22),
            RealParameter("WACC", 0.045, 0.055),
            IntegerParameter("yexpenses", 2, 4),
            RealParameter("rescalation", 0.027, 0.033),
            # RealParameter("i", 0.0675, 0.0825),
            IntegerParameter("t", 23, 27),
            RealParameter("celc", 36, 44),
            RealParameter("cheat", 13.5, 16.5),
            RealParameter("duration", 7200, 8800),
            RealParameter("cMEA", 1.8, 2.2),
        ]
        model.levers = [
            RealParameter("i", 0.05, 0.10),
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

        # Now what?
        n_scenarios = 10
        n_policies = 3

        results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
        print(results)

        experiments, outcomes = results
    
        # Convert experiments to a DataFrame
        df_experiments = pd.DataFrame(experiments)
        df_experiments.to_csv("experiments.csv", index=False)

        data = pd.DataFrame(outcomes)
        data.to_csv("outcomes.csv", index=False)

        # data["policy"] = experiments["policy"] #add the policy-information of my experiments, to the outcomes
        # sns.pairplot(data, hue="policy", vars=list(outcomes.keys()))
        data["i"] = experiments["i"] #add the policy-information of my experiments, to the outcomes
        sns.pairplot(data, hue="i", vars=list(outcomes.keys()))

        plt.show()