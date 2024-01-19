"""
Model() for CCS-cost analysis of 1 plant
Later, the Controller() will ask the Model() to run many times given plant_data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from pyXSteam.XSteam import XSteam
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
    def __init__(self, Name, Fuel=None, Qdh=0, P=0, Qfgc=0, Tsteam=0, psteam=0, Qfuel=0, Vfg=0):
        self.name = Name
        self.fuel = Fuel
        self.Qdh = Qdh
        self.P = P
        self.Qfgc = Qfgc
        self.Tsteam = Tsteam
        self.psteam = psteam
        self.Qfuel = Qfuel
        self.Vfg = Vfg
        if Fuel == "B":
            self.fCO2 = 0.13
        elif Fuel == "W":
            self.fCO2 = 0.09
        self.states = None

    def print_info(self):
        table_format = "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}"
        print(table_format.format("Name", "P", "Qdh", "Qfuel", "Fuel", "Vfg"))
        print(table_format.format("-" * 20, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))
        print(table_format.format(self.name, round(self.P), round(self.Qdh), round(self.Qfuel), self.fuel, round(self.Vfg)))

    def estimate_performance(self):
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
        D = State("D", self.psteam, satL=True)
        if i == max_iterations:
            Pestimated = None
            msteam = 0
        Qfuel = msteam*(A.h-C.h)
        self.Qfuel = Qfuel
        self.P = Pestimated             # TODO: Apply uncertainty% to P/Q?

        # FLUEGASES
        # qbio = 18.6 #[MJ/kg,dry] is 18.6 correct? No, it Hs!
        # c_content = 0.50
        # q_c = qbio*c_content #[MJ/kgC]
        # nCO2 = 44/12 #[kgCO2/kgC]
        # eCO2 = nCO2/q_c #[kgCO2/MJ]
        # eCO2 = eCO2*3600/1000 #[tCO2/MWh] [GW]*[X]*[h/a] = [tCO2/a] => [X] = [tCO2/h/GW]
        # print(eCO2)
        # Johanna does this easily from Qfuel and fueltype, just one x mulitplication. Nice!
        mCO2 = Qfuel * 0.355 #[tCO2/h]
        Vfg = 2000000/110*mCO2/(self.fCO2/0.04) #[m3/h]
        self.Vfg = Vfg

        self.states = A,B,C,D
        if msteam is not None and Pestimated is not None and Qfuel > 0 and pcond_guess > 0 and mCO2 > 0 and Vfg > 0:
            return
        else:
            raise ValueError("One or more of the variables (msteam, Pestimated, Qfuel, pcond_guess, mCO2, Vfg) is not positive.")

    def plot_plant(self):
        A,B,C,D = self.states
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

data = {
    "City": ["Stockholm (South)"],
    "Plant Name": ["Värtaverket KVV 8"],
    "Fuel (W=waste, B=biomass)": ["B"],
    "Heat output (MWheat)": [215],
    "Electric output (MWe)": [130],
    "Existing FGC heat output (MWheat)": [90],
    "Year of commissioning": [2016],
    "Live steam temperature (degC)": [560],
    "Live steam pressure (bar)": [140]
}
# data = {
#     "City": ["Stockholm (South)"],
#     "Plant Name": ["IGELSTA"],
#     "Fuel (W=waste, B=biomass)": ["B"],
#     "Heat output (MWheat)": [155],
#     "Electric output (MWe)": [85],
#     "Existing FGC heat output (MWheat)": [60],
#     "Year of commissioning": [2016],
#     "Live steam temperature (degC)": [540],
#     "Live steam pressure (bar)": [90]
# }

# Creating a DataFrame
df = pd.DataFrame(data)

#Select the DH system here somehow. Dont care about year.
#DH average temp is 47C return and 86C supply

# // MAIN //
# Select our plant
x = df.iloc[0]
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

# Estimate it's performance
chp.estimate_performance()
chp.print_info()
chp.plot_plant()
print("%CO2 input for Aspen:", chp.fCO2)
print("Volume input for Aspen:", chp.Vfg)

# Size a MEA CCS unit
# To do this, I need to run the Aspen model for many fCO2 and Vfg and record the SIZES = f(costs). 
# I can close this for now, go to StuDat(Aspen) and prepare an equipment list in Excel