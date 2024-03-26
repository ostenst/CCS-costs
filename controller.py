"""
Model() for CCS-cost analysis of 1 plant
Later, the Controller() will ask the Model() to run many times given plant_data.
"""
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

# DEFINE MY INPUT DATA AND REGRESSIONS OF ASPEN DATA
plant_data = pd.read_csv("plant_data.csv", sep=";", decimal=',')
print(plant_data.head())

W2E_data = pd.read_csv("W2E.csv", sep=";", decimal=',')
X = W2E_data[['CO2%', 'Flow']]
y = W2E_data.drop(columns=['CO2%', 'Flow'])

W2E_regression = MultiOutputRegressor(LinearRegression())
W2E_regression.fit(X, y)

for index, row in plant_data.iterrows():
    
    # INITIALIZE CHP
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
    
    # ESTIMATE ENERGY BALANCE AND FLUE GASES                                                   
    CHP.estimate_performance()                                                                      # <--- BEGIN RDM EFTER THIS ESTIMATION? YES!
    
    if CHP.fuel == "B":
        CHP.fCO2 = 0.16
    if CHP.fuel == "W":
        CHP.fCO2 = 0.11
    
    CHP.mCO2 = CHP.Qfuel * 0.355 #[tCO2/h]
    CHP.Vfg = 2000000/110*CHP.mCO2/(CHP.fCO2/0.04) #[m3/h]
    CHP.print_info()

    # ESTIMATE SIZE OF MEA PLANT AND THE ENERGY PENALTY
    MEA = MEA_plant(CHP) #should be f(volumeflow,%CO2)... maybe like this: MEA_plant(host=chp, constr_year, currency, discount, lifetime)

    if CHP.fuel == "W":
        MEA.estimate_size(W2E_regression, W2E_data)
    if CHP.fuel == "B":
        print("Aspen data not available for bio-chip fired")

    dTreb = 10
    Plost, Qlost, reboiler_steam = CHP.energy_penalty(MEA, dTreb)
    CHP.print_info()

    CHP.plot_plant()
    CHP.plot_plant(capture_states=reboiler_steam)

    # HEAT INTEGRATION WORK BELOW
    if CHP.fuel == "W" or "B": # Arbitrary before industrial cases are added
        consider_dcc = False

    stream_data = MEA.identify_streams(consider_dcc)
    composite_curve = MEA.merge_heat(stream_data)
    MEA.plot_streams(stream_data)

    Tsupp = 86
    Tlow = 38
    MEA.dTmin = 7

    Qrecovered = MEA.available_heat(composite_curve, Tsupp, Tlow)
    CHP.Qdh += round(Qrecovered/1000)
    CHP.print_info()
    MEA.plot_hexchange()

plt.show()
