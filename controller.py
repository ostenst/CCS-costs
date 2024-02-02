"""
Model() for CCS-cost analysis of 1 plant
Later, the Controller() will ask the Model() to run many times given plant_data.
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
from sklearn.linear_model import LinearRegression
from functions import *

# DEFINE MY INPUT DATA
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
plant_data = {
    "City": ["Göteborg"],
    "Plant Name": ["Renova"],
    "Fuel (W=waste, B=biomass)": ["W"],
    "Heat output (MWheat)": [126],
    "Electric output (MWe)": [45],
    "Existing FGC heat output (MWheat)": [38],
    "Year of commissioning": [1995],            #Select the DH system here somehow. Dont care about year. #DH average temp is 47C return and 86C supply
    "Live steam temperature (degC)": [400],
    "Live steam pressure (bar)": [40]
}
# plant_data = {
#     "City": ["Stockholm (South)"],
#     "Plant Name": ["Värtaverket KVV 8"],
#     "Fuel (W=waste, B=biomass)": ["B"],
#     "Heat output (MWheat)": [215],
#     "Electric output (MWe)": [130],
#     "Existing FGC heat output (MWheat)": [90],
#     "Year of commissioning": [2016],
#     "Live steam temperature (degC)": [560],
#     "Live steam pressure (bar)": [140]
# }
plant_data = pd.DataFrame(plant_data)           #Each input plant will be a row in a dataframe
x = plant_data.iloc[0]
MEA_data = pd.read_csv("MEA_testdata.csv", sep=";", header=None, index_col=0) #TODO: Consider storing in dict, superfast!
Aspen_data = MEA_data.transpose()

# INITIALIZE AND EVALUATE THE CHP
chp = CHP(
    Name=x["Plant Name"],
    Fuel=x["Fuel (W=waste, B=biomass)"],
    Qdh=x["Heat output (MWheat)"],
    P=x["Electric output (MWe)"],
    Qfgc=x["Existing FGC heat output (MWheat)"],
    Tsteam=x["Live steam temperature (degC)"],
    psteam=x["Live steam pressure (bar)"]
)
chp.estimate_performance(plotting=False)
chp.print_info()
# chp.plot_plant()

# SIZE A MEA RETROFIT GIVEN THE HOST PLANT AND ASPEN RUNS
MEA = MEA_plant(chp, Aspen_data) #should be f(volumeflow,%CO2)
# print(Aspen_data)

# Find nominal Ebalance
A,B,C,D = chp.states
mtot = chp.Qfuel*1000 / (A.h-C.h)
TCCS = MEA.data["T_REBOIL"].values[0] + 10 #Assuming dTmin = 10 in reboiler
pCCS = steamTable.psat_t(TCCS)
Ta = steamTable.t_ps(pCCS,A.s)

a = State("a",pCCS,s=A.s,mix=True) #If mixed! You need to add a case if we are outside (in gas phase)
d = State("d",pCCS,satL=True)
mCCS = MEA.data["Q_REBOIL"].values[0] / (a.h-d.h)
mB = mtot-mCCS
print(mtot)
print(mCCS)
chp.P = mtot*(A.h-a.h) + mB*(a.h-B.h)
print("Power, ", chp.P)
chp.P = mtot*(A.h-a.h) + mB*(a.h-B.h) - (MEA.data["W_B311"].values[0]+100)
print("Power, ",chp.P)

print("Q", chp.Qdh*1000)
chp.Qdh = mB*(B.h-C.h)
print("Q", chp.Qdh)
print("Qfgc", chp.Qfgc*1000)


chp.plot_plant(capture_states=[a,d])



# # HEAT INTEGRATION WORK BELOW
# # List of components
# components = ['B4', 'COOL1', 'COOL2', 'COOL3', 'DCCHX', 'DRYCOOL', 'DUMCOOL']

# # Given points for each component
# Q_values = [-Aspen_data[f"Q_{component}"].values[0] for component in components]
# TIN_values = [Aspen_data[f"TIN_{component}"].values[0] for component in components]
# TOUT_values = [Aspen_data[f"TOUT_{component}"].values[0] for component in components]

# # Plotting lines for each component
# plt.figure(figsize=(10, 8))
# for i in range(len(components)):
#     plt.plot([0, Q_values[i]], [TIN_values[i], TOUT_values[i]], marker='o', label=components[i])

# # Adding labels and title
# plt.xlabel('Q')
# plt.ylabel('Temperature (T)')
# plt.title('Streams to be cooled')
# plt.legend()

# Show the plot
# plt.show()

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
