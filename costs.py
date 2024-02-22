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

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
class State:
    def __init__(self, Name, p=None, T=None, s=None, satL=False, satV=False, mix=False):
        self.Name = Name
        if satL==False and satV==False and mix==False:
            self.p = p
            self.T = T
            self.s = steamTable.s_pt(p,T)
            self.h = steamTable.h_pt(p,T)
        if satL==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = steamTable.sL_p(p)
            self.h = steamTable.hL_p(p) 
        if satV==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = steamTable.sV_p(p)
            self.h = steamTable.hV_p(p)
        if mix==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = s
            self.h = steamTable.h_ps(p,s)
        if self.p is None or self.T is None or self.s is None or self.h is None:
            raise ValueError("Steam properties cannot be determined")

    def __str__(self):
        return self.Name
        # return f"s = {round(self.s,1)} | T = {round(self.T)} | h = {round(self.h)}"  
    def plot(self, pressure=False):
        plt.scatter(self.s, self.T, label=str(self), marker='x', color='r')
        plt.annotate( str(self)+': '+str(round(self.p,2))+' bar' if pressure else str(self), (self.s, self.T), textcoords="offset points", xytext=(5,5))
    
# initiate Plant(plant_data)
#     Approximates the plant as a rankine cycle, outputs Q,P,Qfuel, VCO2 etc.,

#     ((( For Paper(III):
#     Optionally could have two modes: condensing(Power)(not older plants?) or backpressure(DesignAlpha) or turbinebypass(HOB).
#         - "As a shift from back pressure to turbine bypass roughly results in 1:1 exchange of electric power to heat" Levihn
#         - Beiron has nice simplifieable assumptions for this!
#     Optionally could have heat pump HP:
#         - Can be used to further move P=>Q dynamically. But do only small consideration in Paper(II)
#         - I.e., in this paper, we only consider that HP could be available OR NOT for the low-grade CCS heat. If available, use. If not, dont use? ASK!!! )))

#     Has two methods:
#         - integrateCCS_norecovery
#                 - maybe don't care about HP, this is just supposed to be "stupid" anyway.
#         - integrateCCS_fullrecovery (consider changing to "optimized" recovery for certain HEX costs)... No, let's do:
#                 - Maximize HEX between useful waste-heat no matter what (this should have a cost!!!)
#                 - if HP:
#                     in reality, cost is dependent on if we use the HP or not, which depends on price. But this paper is not transient. Constant heat+elec price.
#                         => easy! we make a check: Is HP reducing cost? If yes, then the model uses the HP.
#                 - if not HP:
#                     Should it install..? No! We focus on the CCS costs, and let this remain for Paper(III)

#     This should maybe have 3 combinations of Q/P outputs?
#     1) Nominal QP
#     2) Norec QP
#     3) Fullrec QP

#     KeyOutputs:
#     - Qfuel, Q, P
#     - Volume of CO2
#     - Concentration of CO2

class CHP:
    def __init__(self, Name, Fuel=None, Qdh=0, P=0, Qfgc=0, Tsteam=0, psteam=0, Qfuel=0, Vfg=0, mCO2=0, duration=8000):
        self.name = Name
        self.fuel = Fuel
        self.Qdh = Qdh
        self.P = P
        self.Qfgc = Qfgc
        self.Tsteam = Tsteam
        self.psteam = psteam
        self.Qfuel = Qfuel
        self.Vfg = Vfg
        self.mCO2 = mCO2
        self.duration = duration

        if Fuel == "B":
            self.fCO2 = 0.13
        elif Fuel == "W":
            self.fCO2 = 0.09
        self.states = None

    def print_info(self):
        table_format = "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}"
        print(table_format.format("Name", "P", "Qdh+fc", "Qfuel", "Fuel", "Vfg"))
        print(table_format.format("-" * 20, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))
        print(table_format.format(self.name, round(self.P), round(self.Qdh+self.Qfgc), round(self.Qfuel), self.fuel, round(self.Vfg)))

    def estimate_performance(self, plotting=False):
        # ENERGY BALANCE
        Ptarget = self.P
        Qdh = self.Qdh
        A = State("A", self.psteam, self.Tsteam)

        max_iterations = 100
        pcond_guess = 2
        Pestimated = 0
        i = 0
        tol = 0.05
        while abs(Pestimated - Ptarget) > Ptarget*tol and i < max_iterations:
            pcond_guess = pcond_guess - 0.1
            B = State("B", p=pcond_guess, s=A.s, mix=True)
            C = State("C", pcond_guess, satL=True)
            msteam = Qdh/(B.h-C.h)
            Pestimated = msteam*(A.h-B.h)
            i += 1
        if i == max_iterations:
            Pestimated = None
            msteam = 0
        Qfuel = msteam*(A.h-C.h)
        self.Qfuel = Qfuel
        self.P = Pestimated             # TODO: Apply uncertainty% to P/Q?
        D = State("D", self.psteam, satL=True)

        # FLUEGASES
        # qbio = 18.6 #[MJ/kg,dry] is 18.6 correct? No, it Hs!
        # c_content = 0.50
        # q_c = qbio*c_content #[MJ/kgC]
        # nCO2 = 44/12 #[kgCO2/kgC]
        # eCO2 = nCO2/q_c #[kgCO2/MJ]
        # eCO2 = eCO2*3600/1000 #[tCO2/MWh] [GW]*[X]*[h/a] = [tCO2/a] => [X] = [tCO2/h/GW]
        # print(eCO2)
        # Johanna does this easily from Qfuel and fueltype, just one x mulitplication. Nice!
        self.mCO2 = Qfuel * 0.355 #[tCO2/h]
        Vfg = 2000000/110*self.mCO2/(self.fCO2/0.04) #[m3/h]
        self.Vfg = Vfg

        self.states = A,B,C,D
        if plotting == True:
            self.plot_plant(A,B,C,D)

        if msteam is not None and Pestimated is not None and Qfuel > 0 and pcond_guess > 0 and self.mCO2 > 0 and Vfg > 0:
            return 
        else:
            raise ValueError("One or more of the variables (msteam, Pestimated, Qfuel, pcond_guess, mCO2, Vfg) is not positive.")

    def plot_plant(self,A,B,C,D):
        T_start = 0.01
        T_end = 373.14
        num_points = 100  
        T_values = [T_start + (i / (num_points - 1)) * (T_end - T_start) for i in range(num_points)]
        saturation_line = []
        for T in T_values:
            sL = steamTable.sL_t(T)
            sV = steamTable.sV_t(T)
            saturation_line.append([sL,T])
            saturation_line.append([sV,T])
        plt.figure(figsize=(8, 6))  # Set the figure size
        for state in saturation_line:
            plt.plot(state[0], state[1], marker='.', linestyle='-', color='k', markersize=1)
        plt.xlim(0, 10)
        plt.ylim(0, 600)
        plt.xlabel('s [kJ/kgC]')
        plt.ylabel('T [C]')
        plt.title('s,T cycle of CHP')
        plt.grid(True)

        A.plot(pressure=True)
        B.plot(pressure=True)
        C.plot()
        D.plot()

        def draw_line(state1, state2, color='g'):
            plt.plot([state1.s, state2.s], [state1.T, state2.T], linestyle='-', color=color)
        draw_line(A, B, color='cornflowerblue')
        draw_line(B, C, color='cornflowerblue')
        draw_line(C, D, color='b')
        draw_line(D, State(" ", self.psteam, satV=True), color='b')
        draw_line(State(" ", self.psteam, satV=True), A, color='b')
        plt.show()

# data = {
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
data = {
    "City": ["Göteborg"],
    "Plant Name": ["Renova"],
    "Fuel (W=waste, B=biomass)": ["W"],
    "Heat output (MWheat)": [126],
    "Electric output (MWe)": [45],
    "Existing FGC heat output (MWheat)": [38],
    "Year of commissioning": [1995],
    "Live steam temperature (degC)": [400],
    "Live steam pressure (bar)": [40]
}

# Creating a DataFrame
df = pd.DataFrame(data)
x = df.iloc[0]
#Select the DH system here somehow. Dont care about year.
#DH average temp is 47C return and 86C supply

# // MAIN //
chp = CHP(
    Name=x["Plant Name"],
    Fuel=x["Fuel (W=waste, B=biomass)"],
    Qdh=x["Heat output (MWheat)"],
    P=x["Electric output (MWe)"],
    Qfgc=x["Existing FGC heat output (MWheat)"],
    Tsteam=x["Live steam temperature (degC)"],
    psteam=x["Live steam pressure (bar)"]
)

chp.print_info()
chp.estimate_performance(plotting=False)
print(" ... now plant performance was estimated to:")
chp.print_info()

# SEND FLUEGASES TO SIZING
df = pd.read_csv("MEA_testdata.csv", sep=";", header=None, index_col=0) #TODO: Consider storing in dict, superfast!
Aspen_data = df.transpose()
print(Aspen_data)

costs_of_evryething=f(Aspen_data)

class MEA_plant:
    def __init__(self, host_plant, Aspen_data, construction_year=2024, currency_factor=0, discount=0.08, lifetime=25):
        self.host = host_plant
        self.data = Aspen_data
        self.mfluegas = Aspen_data["M_FLUEGAS"].values[0]
        self.rhofluegas = Aspen_data["RHO_FLUEGAS"].values[0]
        #self.data_sized = None THIS IS THE DATAFRAME THAT SHOULD BE SENT TO direct_cost

        self.equipment_list = ['B4', 'COOL1', 'COOL2', 'COOL3', 'DCCHX', 'DRYCOOL', 'DUMCOOL', 'HEX', 'B5', 'PA2627', 'DCCPUMP',
                  'PUMP','STRIPPER','WASHER','DCC','ABSORBER','FLASH1','FLASH2','DRYFLASH', 'REBOIL', 'B311']
        self.construction_year = construction_year
        self.currency_factor = currency_factor
        self.discount = discount
        self.lifetime = lifetime
        self.annualization = 0
        for n in range(1, lifetime):
            self.annualization += 1/(1 + discount)**n
        self.duration = host_plant.duration

    # def linearRegression(Aspen_data):
        # TODO: MAKE PROPER MEA_TESTDATA FIRST
        # return SIZES

    def direct_cost(self, equipment): 
        #TODO: Double check these, also compressor function?
        #TODO: Missing COMPRESSION&LIQUEFACTION stages!
        df = self.data #TODO: ASPEN DATA HAS SOME SIZES, BUT NOT ALL! SOME ARE GIVEN BY ENERGY BALANCE (HEXs)
        HEX_list = ['B4', 'COOL1', 'COOL2', 'COOL3', 'DCCHX', 'DRYCOOL', 'DUMCOOL', 'HEX', 'B5', 'PA2627']
        pump_list = ['DCCPUMP','PUMP']
        tower_list = ['STRIPPER','WASHER','DCC','ABSORBER']
        flash_list = ['FLASH1','FLASH2','DRYFLASH']

        if equipment in HEX_list:                               #TODO: Calculate Areas in E-BALANCE function first! Where HEATPUMPS can be installed for full recovery, but (later, PaperIII) be operated flexibly.
            key = 'A_' + equipment
            area = df[key][1]
            cost = 2.8626 * area**0.7988        #EUR2015 convert with CEPCI, AVOIDED COST REQUIRES GRID EMISSIONS (Scope2), TRANPOSRT=0 for now (Scope3, storage, leakage)
                                                # How to integrate low CONC%? 

        if equipment in pump_list:
            key = 'VT_' + equipment
            volumeflow = df[key][1]
            cost = 32.147 * (volumeflow*1000)**0.6029 #L/s

        if equipment in tower_list: #TODO: CHECK ABSORBER DIMENSIONS; PROBABLY WRONG IN EXCEL
            key = 'D_' + equipment
            diameter = df[key][1]
            key = 'H_' + equipment
            height = df[key][1]
            volume = 3.1415 * diameter**2/4 * height
            cost = 91.764 * volume**0.6154

        if equipment in flash_list:
            key = 'V_' + equipment
            volume = df[key][1]
            cost = 66.927 * volume**0.5047

        if equipment == 'REBOIL':
            key = 'Q_' + equipment
            Qreb = df[key][1]
            U = 1.5 #kW/m2K (Biermann)
            dT1 = 131.1-121.09  #Assuming reboiler temps. and dTmin #TODO: MOVE THIS TO THE E-BALANCE FUNCTION
            dT2 = 131.09-121.1
            LMTD = (dT1 - dT2)/math.log(dT1/dT2)
            A = Qreb/(U*LMTD)
            cost = 1.6758 * A**0.8794
        
        if equipment == 'B311':
            key = 'W_' + equipment
            work = df[key][1]
            cost = 2.7745 * work**0.7814

        # TODO: Apply currency conversion here.
        return cost

    def NOAK_escalation(self, TDC):
        # Adding sstartup-material! 8 % of total material required, scale by CO2 t/h (check the whitepaper)
        # aCAPEX:
        process_contingency = 0.15
        TDCPC = TDC*(1 + process_contingency)
        indirect_costs = 0.25
        EPC = TDCPC*(1 + indirect_costs)
        project_contingency = 0.30
        TPC = EPC*(1 + project_contingency) #Use for OPEX
        ownercost_interest = 0.095
        TCR = TPC*(1 + ownercost_interest)

        aCAPEX = TCR/self.annualization

        # OPEX (non-energy): (Site-TEA-Tharun)¨
        mMEA = self.data["M_MAKEUP"].values[0]      #kg/s
        mMEA = mMEA/1000 * 3600*self.duration       #tMEA/a
        cost_MEA = mMEA * 1.7                       #kEUR/a

        mH2O = self.data["M_H2OIN"].values[0]       #kg/s
        mH2O = mH2O/1000 * 3600*self.duration       #tH2O/a
        cost_H2O = mH2O * 0.02/1000                 #kEUR/a 
       
        cost_maintenance = TPC * 0.045              #kEUR/a 
        cost_labor = 411                            #kEUR/a 

        return TCR, aCAPEX, cost_H2O, cost_MEA, cost_maintenance, cost_labor

    def specific_annualized(self, cost):
        cost = cost/self.annualization              #kEUR,annualized
        mCO2 = self.data["MCO2_CO2OUT"].values[0]   #kgCO2/s
        mCO2 = mCO2/1000 * 3600*self.duration       #tCO2/a
        cost_specific = cost/mCO2 * 1000            #EUR/tCO2
        return cost_specific
# NOAK factors:                 SOURCES:
# PC = 0-10                     white paper
# IC = 25 (14 EBTF)             site-TEA Tharun/Max
# Proj cont = 30                white paper
# OC&Interest = 9.5 (15 EBTF, maybe 7 Rubin) site-TEA Tharun/Max

# FOAK factors (intended for TRL6-7, according to Rubin whitepaper):
# PC = 20-35 (for TRL 5-6) or 30-70 (for TRL4)  white paper
# IC = 25 (14 EBTF)             site-TEA Tharun/Max
# Proj cont = 50                white paper
# OC&Interest = 9.5 (15 EBTF, maybe 7 Rubin) site-TEA Tharun/Max
# Also add redundancy etc.      white paper

MEA = MEA_plant(chp, Aspen_data) #SHOULD BE FUNCTION OF VOLUME FLOW, AND PERCENTAGE

redundancy = 0
direct_costs = []
for equipment in MEA.equipment_list:
    direct_cost = MEA.direct_cost(equipment)
    direct_costs.append(direct_cost)

    if equipment in ['DCCPUMP','PUMP','B311','FLASH1','FLASH2','DRYFLASH']: #TODO: What components should have redundancy? What %level? Or just apply 1 factor?
        redundancy += direct_cost
plt.figure(figsize=(10, 6))
plt.bar(MEA.equipment_list, direct_costs, color='blue')
plt.xlabel('Equipment Names')
plt.ylabel('Direct Costs [kEUR]')
plt.title('Direct Costs of Equipment Items')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
# plt.show()


# Non-EnergyCosts:
TDC = sum(direct_costs)
print(" ")
print("TotalDirectCost is", TDC)
print("TDC specific annualized is", MEA.specific_annualized(TDC) )
TCR, aCAPEX, cost_H2O, cost_MEA, cost_maintenance, cost_labor = MEA.NOAK_escalation(TDC)

# Rough estimate of EnergyCost (site-TEA):
steam_price = 28.4          #EUR/MWh
coolingwater_price = 0.02   #EUR/t
elec_price = 60             #EUR/MWh
elec_CO2 = 0                # FOR INDUSTRIES!

steam_CO2 = 0

Qreb = MEA.data["Q_REBOIL"].values[0] #kW
Qcool = 0                             #kW
for cooler in ['B4', 'COOL1', 'COOL2', 'COOL3', 'DCCHX', 'DRYCOOL', 'DUMCOOL']:
    key = 'Q_' + cooler
    Qcool += MEA.data[key].values[0] 
Welc = 0                                #kW
for pump in ['DCCPUMP', 'PUMP', 'B311']:
    key = 'W_' + pump
    Welc += MEA.data[key].values[0]

cost_steam = Qreb/1000 * MEA.duration * steam_price/1000 #kEUR/a
mcool = -Qcool/(4.18*15) #kg/s assuming cp=4.18kJ/kg and dT=15C)
cost_coolingwater = mcool/1000 * 3600*MEA.duration * coolingwater_price/1000 #kEUR/a
cost_elec = Welc/1000 * MEA.duration * elec_price/1000 #kEUR/a
print(cost_steam, cost_coolingwater, cost_elec)
print("Promising, because: CostSteam dominates, and is ~= 2*aCAPEX. Also CostWelc is about ~= CostSteam/10 (not counting Compr&Lique), and CostMaintenance is < aCAPEX/2. These general patterns are consistent with Ali, 2019. But MEAmakeup is weirdly high!")

variable_names = ['aCAPEX', 'cost_elec', 'cost_coolingwater', 'cost_steam', 'cost_MEA', 'cost_maintenance', 'cost_labor', 'cost_H2O']
variable_costs = [aCAPEX, cost_elec, cost_coolingwater, cost_steam, cost_MEA, cost_maintenance, cost_labor, cost_H2O]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(variable_names, variable_costs, color='green')
plt.xlabel('Cost Variables')
plt.ylabel('Cost (kEUR/a)')
plt.title('Costs of Different Variables')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()

# Show the plot
plt.show()
