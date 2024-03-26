"""
Model() for CCS-cost analysis of 1 plant
Later, the Controller() will ask the Model() to run many times given plant_data.
"""
import pandas as pd
import matplotlib.pyplot as plt
from functions import *

# DEFINE MY INPUT DATA
W2E_data = pd.read_csv("W2E.csv", sep=";", decimal=',') 
plant_data = pd.read_csv("plant_data.csv", sep=";", decimal=',')
print(plant_data.head())

for index, row in plant_data.iterrows():
    # INITIALIZE AND EVALUATE THE CHP
    chp = CHP(
        Name=row["Plant Name"],
        Fuel=row["Fuel (W=waste, B=biomass)"],
        Qdh=float(row["Heat output (MWheat)"]),
        P=float(row["Electric output (MWe)"]),
        Qfgc=float(row["Existing FGC heat output (MWheat)"]),
        Tsteam=float(row["Live steam temperature (degC)"]),
        psteam=float(row["Live steam pressure (bar)"])
    )
    chp.estimate_performance()
    chp.print_info()
    MEA = MEA_plant(chp) #should be f(volumeflow,%CO2)... maybe like this: MEA_plant(host=chp, constr_year, currency, discount, lifetime)
    MEA.estimate_size(W2E_data)
    print(MEA.data["Qdry"])

    # RDM MODEL SHOULD MAYBE START HERE? No... or well it should be before we assign discountrates etc. But maybe after extracting Aspen_data!
    Plost, Qlost, reboiler_steam = chp.energy_penalty(MEA)
    chp.print_info()

    chp.plot_plant()
    chp.plot_plant(capture_states=reboiler_steam)

    # HEAT INTEGRATION WORK BELOW
    # considered_streams = ['wash', 'toabs', 'strip', 'lean', 'int2', 'int1', 'dcc', 'dhx', 'dry', 'dum', 'rcond', 'rint', 'preliq']
    # considered_streams = ['wash', 'toabs', 'strip', 'lean', 'int2', 'int1', 'dhx', 'dry', 'rcond', 'rint', 'preliq']
    considered_streams = ['wash', 'strip', 'lean', 'int2', 'int1', 'dhx', 'dry', 'rcond', 'rint', 'preliq'] # Add 'dcc' (not 'toabs'?) when DCC HEX is needed

    stream_data = MEA.identify_streams(considered_streams)
    MEA.plot_streams(stream_data)

    temperature_ranges = MEA.find_ranges(stream_data)
    composite_curve = MEA.merge_heat(temperature_ranges, stream_data)

    Tsupp = 86
    Tlow = 38
    dTmin = 7

    Qsupp, Qlow, Qpinch, Tpinch = MEA.available_heat2(composite_curve, Tsupp, Tlow, dTmin=dTmin)
    MEA.plot_hexchange(Qsupp, Qlow, Qpinch, Tpinch, dTmin, Tlow, Tsupp) #TODO: Move some values to the self. of CHP or MEA
    Qrecovered = (Qpinch-Qsupp) + (Qlow-Qpinch)
    Qcw = MEA.composite_curve[-1][0]-Qlow
    chp.Qdh += round(Qrecovered/1000)
    chp.print_info()

plt.show()



# # But what is the cost of this heat exchange? Find areas!
# U = 1 #kW/m2K, (Deng compr study)
# Alow, Ahigh, Acw = exchanger_areas(Qhighgrade, Qlowgrade, Qcw, U, dTmin, Tmax, Tsupp, Tlow, Tend)
# print("I believe in these areas, but they are dissimilar from Aspen: ", Ahigh, Alow, Acw)

# # Integrate the HEXs
# Qdh, mcoolingwater = chp.heat_integrate(Qhighgrade, Qlowgrade, Qcw)
# print(Qdh, mcoolingwater)
# chp.print_info()



# Aspen_data = MEA.linearReg(Aspen_data) #This is ok and can happen, but some things (e.g. Acool1) should not be used
# ebalance = MEA.Ebalance(Aspen_data,chp)

# direct_costs = []
# for equipment in MEA.equipment_list:
#     direct_cost = MEA.direct_cost(equipment)
#     direct_costs.append(direct_cost)


# plt.figure(figsize=(10, 6))
# plt.bar(MEA.equipment_list, direct_costs, color='blue')
# plt.xlabel('Equipment Names')
# plt.ylabel('Direct Costs [kEUR]')
# plt.title('Direct Costs of Equipment Items')
# plt.xticks(rotation=45, ha='right')  
# plt.tight_layout()
# # plt.show()

# # Non-EnergyCosts:
# TDC = sum(direct_costs)
# print(" ")
# print("TotalDirectCost is", TDC)
# print("TDC specific annualized is", MEA.specific_annualized(TDC) )
# TCR, aCAPEX, cost_H2O, cost_MEA, cost_maintenance, cost_labor = MEA.NOAK_escalation(TDC)

# # Rough estimate of EnergyCost (site-TEA):
# steam_price = 28.4          #EUR/MWh
# coolingwater_price = 0.02   #EUR/t
# elec_price = 60             #EUR/MWh

# Qreb = MEA.data["Q_REBOIL"].values[0] #kW
# Qcool = 0                             #kW
# for cooler in ['B4', 'COOL1', 'COOL2', 'COOL3', 'DCCHX', 'DRYCOOL', 'DUMCOOL']:
#     key = 'Q_' + cooler
#     Qcool += MEA.data[key].values[0] 
# Welc = 0                                #kW
# for pump in ['DCCPUMP', 'PUMP', 'B311']:
#     key = 'W_' + pump
#     Welc += MEA.data[key].values[0]

# cost_steam = Qreb/1000 * MEA.duration * steam_price/1000 #kEUR/a
# mcool = -Qcool/(4.18*15) #kg/s assuming cp=4.18kJ/kg and dT=15C)
# cost_coolingwater = mcool/1000 * 3600*MEA.duration * coolingwater_price/1000 #kEUR/a
# cost_elec = Welc/1000 * MEA.duration * elec_price/1000 #kEUR/a
# print(cost_steam, cost_coolingwater, cost_elec)
# print("Promising, because: CostSteam dominates, and is ~= 2*aCAPEX. Also CostWelc is about ~= CostSteam/10 (not counting Compr&Lique), and CostMaintenance is < aCAPEX/2. These general patterns are consistent with Ali, 2019. But MEAmakeup is weirdly high!")

# variable_names = ['aCAPEX', 'cost_elec', 'cost_coolingwater', 'cost_steam', 'cost_MEA', 'cost_maintenance', 'cost_labor', 'cost_H2O']
# variable_costs = [aCAPEX, cost_elec, cost_coolingwater, cost_steam, cost_MEA, cost_maintenance, cost_labor, cost_H2O]

# # Plotting the bar chart
# plt.figure(figsize=(10, 6))
# plt.bar(variable_names, variable_costs, color='green')
# plt.xlabel('Cost Variables')
# plt.ylabel('Cost (kEUR/a)')
# plt.title('Costs of Different Variables')
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# plt.tight_layout()

# # Show the plot
# plt.show()
