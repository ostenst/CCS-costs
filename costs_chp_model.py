"""
Model() for CCS-cost analysis of 1 plant
Later, the Controller() will ask the Model() to run many times given plant_data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self,Name,Fuel=None,Q=0,P=0,Qfgc=0,Tsteam=0,psteam=0):
        self.name = Name
        self.fuel = Fuel
        self.Q = Q
        self.P = P
        self.Qfgc = Qfgc
        self.Tsteam = Tsteam
        self.psteam = psteam

    def estimate_performance(self):
        # Assuming condensation pressure at 1 bar (0.4-1.2 is typical range) and eta_is=0.88:
        p_condenser = 1
        eta_is = 1 #Seems like this perfect expansion "counterbalances" any missing reheat% and other efficiency gains.
        HHV = 10.2*1000 #[kJ/kg]
        LHV = 8.2*1000 #[kJ/kg]
        C_content = 0.50 #TODO: define a general fuel instead properly, since this gives fluegas volume. Check Ebsilon for Lambda? Check LHV/HHV, how to handle? Should link to M_content.
        M_content = 0.40
        t = 8760*0.70

        A = State("A", self.psteam, self.Tsteam)
        Bis = State("Bis", p=p_condenser, s=A.s, mix=True)
        hB = A.h - (A.h-Bis.h)*eta_is
        if steamTable.x_ph(p_condenser,hB)==1:
            B = State("B", p_condenser,steamTable.t_ph(p_condenser,hB))
        else:  
            B = State("B", p_condenser,s=steamTable.s_ph(p_condenser, hB), mix=True)
        C = State("C", p_condenser, satL=True)
        D = State("D", self.psteam, satL=True)

        energy_balance = np.array([
        [0, 1, 1, -HHV, 0],
        [0, 0, 1, -(HHV-LHV), 0],
        [0, 1, 0, 0, -(A.h-C.h)],
        [0, 0, 0, 0, (A.h-B.h)],
        [1, 0, 0, 0, -(B.h-C.h)],
        ])
        constants = [0, 0, 0, self.P*1000, 0]
        [Q, Qboiler, Qfgc, mfuel, msteam] = np.linalg.solve(energy_balance, constants)
        mCO2 = mfuel*(1-M_content)/1000 * C_content*44/12 * t*3600 #tCO2/yr

        results = {
            'P' : self.P*1000,
            'QDH': Q,
            'Qboiler': Qboiler,
            'Qfgc': Qfgc,
            'mfuel': mfuel,
            'msteam': msteam,
            'mCO2' : mCO2
        }
        print("Results:")
        for variable, value in results.items():
            print(f"{variable}: {value}")
        # self.plot_plant(A,B,C,D) 
            

        # TODO: Estimate fluegas volume and CO2 concentration!
            
        # TODO: A "test" function that tells us if this estimate is reasonable or not!

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
        draw_line(A, B, color='b')
        draw_line(B, C, color='cornflowerblue')
        draw_line(C, D, color='cornflowerblue')
        draw_line(D, A, color='b')
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

# Creating a DataFrame
df = pd.DataFrame(data)
x = df.iloc[0]
#Select the DH system here somehow. Dont care about year.
#DH average temp is 47C return and 86C supply

# // MAIN //
chp = CHP(
    Name=x["Plant Name"],
    Fuel=x["Fuel (W=waste, B=biomass)"],
    Q=x["Heat output (MWheat)"],
    P=x["Electric output (MWe)"],
    Qfgc=x["Existing FGC heat output (MWheat)"],
    Tsteam=x["Live steam temperature (degC)"],
    psteam=x["Live steam pressure (bar)"]
)

chp.estimate_performance()
