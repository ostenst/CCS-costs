"""
Model() for CCS-cost analysis of 1 plant
Later, the Controller() will ask the Model() to run many times given plant_data.
"""
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

def CCS_CHP(
    eta_boiler=0.87,
    fCO2_B=0.16,
    fCO2_W=0.11,
    dTreb=10,
    Tsupp=86,
    Tlow=38,
    dTmin=7,

    alpha=6.12,
    beta=0.6336,
    CEPCI=600/550,
    fixed=0.06,
    ownercost=0.2,
    WACC=0.05,
    yexpenses=3,
    rescalation=0.03,
    i=0.075,
    t=25,
    celc=40,
    cheat=15,
    cMEA=2,

    duration=5000,
    rate=90,

    CHP=None,
    W2E_regression=None,
    W2E_data=None,
    CHIP_interpolations=None,
    CHIP_data=None
):
    # Output should be one or many KPIs?
    MultiObjective = False

    # Collect assumptions in dicts
    technology_assumptions = {
        "eta_boiler": eta_boiler,
        "fCO2_B": fCO2_B,
        "fCO2_W": fCO2_W,
        "dTreb": dTreb,
        "Tsupp": Tsupp,
        "Tlow": Tlow,
        "dTmin": dTmin,

        "rate": rate
    }
    
    economic_assumptions = {
        'alpha': alpha,
        'beta': beta,
        'CEPCI': CEPCI,
        'fixed': fixed,
        'ownercost': ownercost,
        'WACC': WACC,
        'yexpenses': yexpenses,
        'rescalation': rescalation,
        'i': i,
        't': t,
        'celc': celc,
        'cheat': cheat,
        'duration': duration,
        'cMEA': cMEA
    }

    # Size MEA plant and integrate it
    Vfg, fCO2 = CHP.burn_fuel(technology_assumptions)
    # print("VOLUME OF FLUE GASES", Vfg)
    # print("FRACTION OF CO2", fCO2)

    MEA = MEA_plant(CHP)
    if CHP.fuel == "W":
        print("/// The W2E regression does not fit the estimate_size() function! ///")
        MEA.estimate_size(W2E_regression, W2E_data)
    elif CHP.fuel == "B":
        MEA.estimate_size(CHIP_interpolations, CHIP_data)
    Plost, Qlost = CHP.energy_penalty(MEA)

    # Recover excess heat
    if CHP.fuel == "W" or CHP.fuel == "B":  # Arbitrary before industrial cases are added
        consider_dcc = False

    stream_data = MEA.select_streams(consider_dcc)
    composite_curve = MEA.merge_heat(stream_data)

    Qrecovered = MEA.available_heat(composite_curve)
    CHP.Qdh += round(Qrecovered)

    # Determine costs and KPIs
    CAPEX, aCAPEX, fixedOPEX = MEA.CAPEX_costs(economic_assumptions, escalate=True)
    energyOPEX, otherOPEX = MEA.OPEX_costs(economic_assumptions, Plost, Qlost, Qrecovered)

    cost_labels = ['aCAPEX', 'fixedOPEX', 'energyOPEX', 'otherOPEX']
    costs = [aCAPEX, fixedOPEX, energyOPEX, otherOPEX]                                          # MEUR/yr
    costs_specific = [x*10**6 / (CHP.mCO2*economic_assumptions['duration']) for x in costs]     # EUR/tCO2
    cost_specific = sum(costs_specific)                                                         # EUR/tCO2
    emission_intensity = 0.355*CHP.Qfuel / (CHP.Qdh+CHP.Qfgc+CHP.P)                             # tCO2/MWhoutput
    consumer_cost = emission_intensity * 0.9 * cost_specific                                    # EUR/MWhoutput

    energy_deficit = (Plost + (Qlost - Qrecovered))*economic_assumptions['duration']            # MWh/yr
    fuel_penalty = (Plost + (Qlost - Qrecovered))/(CHP.Qfuel)                                   # % of input fuel used for capture

    if fuel_penalty < 0: # NOTE: Include this criteria in the real analysis later
        print(" ")
        print("These assumptions are unfeasible:")
        print(Plost, Qlost, Qrecovered)
        print(CHP.Vfg/3600*0.8, " kg/s")
        print(MEA.data.head())
        for key, value in technology_assumptions.items():
            print(key, ":", value)
        CHP.plot_plant()
        MEA.plot_hexchange()
        plt.show()
        raise ValueError
    # MEA.plot_streams(stream_data)
    # MEA.plot_hexchange()
    if MultiObjective:
        return CAPEX, costs, costs_specific, cost_specific, consumer_cost, energy_deficit, fuel_penalty
    else:
        return [cost_specific, consumer_cost, fuel_penalty]




if __name__ == "__main__":
    # DEFINE MY INPUT DATA AND REGRESSIONS OF ASPEN DATA
    plant_data = pd.read_csv("plant_data_test.csv", sep=";", decimal=',')
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
        CHP.estimate_rankine() #Begin RDM after this

        technology_assumptions = {
        "eta_boiler": 0.87,
        "fCO2_B": 0.16,  # For fuel type "B" #Höga? Johanna tror 8-10% waste, och 13-16% biochips
        "fCO2_W": 0.11,  # For fuel type "W"
        "dTreb": 10,
        "Tsupp": 86,
        "Tlow": 38,
        "dTmin": 7,
        }
        
        Vfg, fCO2 = CHP.burn_fuel(technology_assumptions) #LHV! Är det 10-12MJ/kg? (Johanna), kanske 8 om fuktigare
        CHP.print_info()
        CHP.plot_plant()

        # ESTIMATE SIZE OF MEA PLANT AND THE ENERGY PENALTY
        MEA = MEA_plant(CHP)

        if CHP.fuel == "W":
            MEA.estimate_size(W2E_regression, W2E_data)
        elif CHP.fuel == "B":
            print("Aspen data not available for bio-chip fired")

        Plost, Qlost = CHP.energy_penalty(MEA)
        CHP.print_info()
        CHP.plot_plant()

        # RECOVER EXCESS HEAT TO DH
        if CHP.fuel == "W" or CHP.fuel == "B":  # Arbitrary before industrial cases are added
            consider_dcc = False

        stream_data = MEA.select_streams(consider_dcc)
        composite_curve = MEA.merge_heat(stream_data)
        MEA.plot_streams(stream_data)

        Qrecovered = MEA.available_heat(composite_curve)
        CHP.Qdh += round(Qrecovered)
        CHP.print_info()
        MEA.plot_hexchange()


        # DETERMINE CAPEX 
        # TODO: determine social vs private cost (maybe cascade? how does this increase the costs for end-consumers?)(compare with Levinh, 2019)(EU fines for non-compliance?)
        # NPV = sum(t=1->n)( cash(t)/(1+i)^t ) #(https://link.springer.com/article/10.1007/s10973-021-10833-z)

        economic_assumptions = {
        'alpha': 6.12,
        'beta': 0.6336,
        'CEPCI': 600/550,
        'fixed': 0.06,
        'ownercost': 0.2,
        'WACC': 0.05,
        'yexpenses': 3,
        'rescalation': 0.03,
        'i': 0.075,
        't': 25,
        'celc': 40,
        'cheat': 15,
        'duration': 8000,
        'cMEA': 2
        }

        CAPEX, aCAPEX, fixedOPEX = MEA.CAPEX_costs(economic_assumptions, escalate=True)
        energyOPEX, otherOPEX = MEA.OPEX_costs(economic_assumptions, Plost, Qlost, Qrecovered)

        # SYNTHESIZE KPIs AND PLOT
        cost_labels = ['aCAPEX', 'fixedOPEX', 'energyOPEX', 'otherOPEX']
        costs = [aCAPEX, fixedOPEX, energyOPEX, otherOPEX] #MEUR/yr
        costs_specific = [x*10**6 / (CHP.mCO2*economic_assumptions['duration']) for x in costs]     #EUR/tCO2

        emission_intensity = 0.355*CHP.Qfuel / (CHP.Qdh+CHP.Qfgc+CHP.P)     #tCO2/MWhoutput, NOTE: these are total emissions produced, 
        consumer_cost = emission_intensity * 0.9 * sum(costs_specific)      #EUR/MWhoutput, NOTE: only 90% of these are captured and should be paid for by consumers

        energy_deficit = (Plost + (Qlost - Qrecovered))*economic_assumptions['duration']
        fuel_penalty = (Plost + (Qlost - Qrecovered))/(CHP.Qfuel)

        print('Cost of capture: ', round(sum(costs_specific)), 'EUR/tCO2')
        print('Consumer added cost: +', round(consumer_cost), 'EUR/MWh')
        print('Energy deficit: ', round(energy_deficit/1000), 'GWh/yr')
        print('Fuel penalty: ', round(fuel_penalty*100,1), '%')

        # FOAK: we can add system contingenciy of 1st, 2nd, 3rd plant (towards). Let's treat HPC as NOAK standalone, but FOAK when systemintegrated.
        
        # ADD SITE-TEA


        # Plotting the: costs_specific
        plt.figure(figsize=(10, 6))

        # Define the positions for each segment
        positions = range(len(costs_specific))
        width = 0.5

        # Plot each segment
        bottom = 0
        for i, cost in enumerate(costs_specific):
            plt.bar(0, cost, bottom=bottom, label=cost_labels[i])
            bottom += cost

        plt.xlabel('Cost Element')
        plt.ylabel('Specific Cost (EUR/tCO2)')
        plt.title('Specific Costs Breakdown')
        plt.ylim(0, sum(costs_specific) * 1.2)
        plt.legend()

        plt.grid(axis='y')
        
    plt.show()
