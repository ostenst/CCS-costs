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
from ema_workbench.analysis import prim


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
from scipy.interpolate import LinearNDInterpolator

def analyze(results, ema_model, ooi):
    """analyze results using SALib sobol, returns a dataframe"""

    _, outcomes = results

    problem = get_SALib_problem(ema_model.uncertainties)
    y = outcomes[ooi]
    sobol_indices = sobol.analyze(problem, y)
    sobol_stats = {key: sobol_indices[key] for key in ["ST", "ST_conf", "S1", "S1_conf"]}
    sobol_stats = pd.DataFrame(sobol_stats, index=problem["names"])
    sobol_stats.sort_values(by="ST", ascending=False)
    s2 = pd.DataFrame(sobol_indices["S2"], index=problem["names"], columns=problem["names"])
    s2_conf = pd.DataFrame(
        sobol_indices["S2_conf"], index=problem["names"], columns=problem["names"]
    )

    return sobol_stats, s2, s2_conf

if __name__ == "__main__":

    # Read data and fit regressors
    plant_data = pd.read_csv("plants-chip-all.csv", sep=";", decimal=',')
    print(plant_data.head())

    W2E_data = pd.read_csv("MEA-w2e.csv", sep=";", decimal=',')
    X = W2E_data[['CO2%', 'Flow']]
    y = W2E_data.drop(columns=['CO2%', 'Flow'])
    W2E_regression = MultiOutputRegressor(LinearRegression())
    W2E_regression.fit(X, y)

    CHIP_data = pd.read_csv("MEA-chip.csv", sep=";", decimal=',')
    print(CHIP_data.head())

    all_experiments = pd.DataFrame()
    all_data = pd.DataFrame()

    # Create interpolation functions for each y column
    x1 = CHIP_data['Flow']
    x2 = CHIP_data['Rcapture']
    x_values = np.column_stack((x1, x2))
    y_values = CHIP_data.drop(columns=['Flow', 'Rcapture']).values
    
    CHIP_interpolations = {}

    for idx, column_name in enumerate(CHIP_data.drop(columns=['Flow', 'Rcapture'])):

        y = y_values[:, idx]
        interp_func = LinearNDInterpolator(x_values, y)
        CHIP_interpolations[column_name] = interp_func

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
        
        if CHP.fuel == "W": # Model currently only works with wood chips, i.e. B
            continue
        CHP.estimate_rankine()

        model = Model("CCSproblem", function=CCS_CHP)       
        model.uncertainties = [
            RealParameter("eta_boiler", 0.92, 0.96),
            RealParameter("fCO2_B", 0.13, 0.18),    # from BEIRON, SwedenMACC
            RealParameter("fCO2_W", 0.09, 0.15),
            RealParameter("dTreb", 7, 12),
            RealParameter("Tsupp", 78, 100),        #From Kåre
            RealParameter("Tlow", 43, 55),
            RealParameter("dTmin", 5, 10),

            RealParameter("alpha", 5.8, 6.5),       #From Eliasson, 2021
            RealParameter("beta", 0.60, 0.67),
            RealParameter("CEPCI", 1.0909, 1.333),
            RealParameter("fixed", 0.04, 0.08),     #6% from BEIRON, SwedenMACC
            RealParameter("ownercost", 0.10, 0.25), #20% from NETL, cost methdoology osti.gov
            RealParameter("WACC", 0.03, 0.07),      #~5% from NETL? Ish
            IntegerParameter("yexpenses", 2, 5),
            RealParameter("rescalation", 0.02, 0.04), #~3% NETL
            RealParameter("i", 0.04, 0.15),
            IntegerParameter("t", 22, 32),
            RealParameter("celc", 30, 180),
            RealParameter("cheat", 50, 180), 
            RealParameter("cMEA", 1.2, 2.8),
        ]
        model.levers = [
            RealParameter("duration", 3000, 6000),
            RealParameter("rate", 78, 94),
        ]
        model.outcomes = [
            ScalarOutcome("mCO2", ScalarOutcome.MAXIMIZE),
            ScalarOutcome("cost_specific", ScalarOutcome.MINIMIZE),
            ScalarOutcome("consumer_cost", ScalarOutcome.MINIMIZE),
            ScalarOutcome("fuel_penalty", ScalarOutcome.MINIMIZE),
            ScalarOutcome("energy_deficit", ScalarOutcome.MINIMIZE),
        ]
        model.constants = [
            Constant("CHP", CHP), # Overwrite the default CHP=None value
            Constant("W2E_regression", W2E_regression),
            Constant("W2E_data", W2E_data),
            Constant("CHIP_interpolations", CHIP_interpolations),
            Constant("CHIP_data", CHIP_data),
        ]
        ema_logging.log_to_stderr(ema_logging.INFO)

        # Perform the experiments (check Sobol requirement for the number of scenarios)
        print(f"Exploring outcomes of implementing CCS at {CHP.name}:")
        n_scenarios = 10
        n_policies = 10

        results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
        experiments, outcomes = results
        # CHP.plot_plant()
    
        # Convert experiments to DataFrames and save
        df_experiments = pd.DataFrame(experiments)
        df_experiments["Name"] = CHP.name
        df_experiments.to_csv("experiments.csv", index=False)
        all_experiments = pd.concat([all_experiments, df_experiments], ignore_index=True)

        data = pd.DataFrame(outcomes)
        data["Name"] = CHP.name
        data.to_csv("outcomes.csv", index=False)
        all_data = pd.concat([all_data, data], ignore_index=True)

        # # Perform SA here? GOAL: To make a very distinct MACC. Don't need crazy runs, just many, for PRIM to find. But I need to know what to constrain to. So Sobol is good
        # if CHP.name == "Vartaverket KVV 8 ":
        #     results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.SOBOL, lever_sampling = Samplers.SOBOL)
        #     sobol_stats, s2, s2_conf = analyze(results, model,"cost_specific")
        #     print(sobol_stats)
        #     print(s2)
        #     print(s2_conf)

        # # Plotting results NOTE: DO NOT PLOT IF MANY PLANTS
        # data["duration"] = experiments["duration"] #add the policy-information of my experiments, to the outcomes
        # sns.pairplot(data, hue="duration", vars=list(outcomes.keys()))

        # plt.figure(figsize=(8, 6))
        # plt.scatter(df_experiments['duration'], df_experiments['beta'], c=data['cost_specific'], cmap='viridis', label='cost')
        # plt.colorbar(label='Y')  # Add color bar legend
        # plt.xlabel('duration')
        # plt.ylabel('beta')
        # plt.title('duration vs beta Colored by Y')
        # plt.legend()
        # plt.show()

    # Loop is done. Now it's time to construct my "global" resuls, e.g. the MACC curve.
    all_experiments.to_csv("all_experiments.csv", index=False)
    all_data.to_csv("all_outcomes.csv", index=False)
        
plt.show()