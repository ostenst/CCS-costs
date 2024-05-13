"""
Helper functions for the controller.py main script
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from ema_workbench.em_framework import get_SALib_problem
from SALib.analyze import sobol


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

class CHP_plant:
    def __init__(self, name, fuel=None, Qdh=0, P=0, Qfgc=0, ybirth=0, Tsteam=0, psteam=0):
        self.name = name
        self.fuel = fuel
        self.Qdh = Qdh
        self.P = P
        self.Qfgc = Qfgc
        self.ybirth = ybirth
        self.Tsteam = Tsteam
        self.psteam = psteam
        self.technology_assumptions = None
        
        self.Qboiler = None
        self.Qfuel = None
        self.Vfg = None
        self.fCO2 = None
        self.mCO2 = None
        self.states = None
        self.reboiler_steam = None

    def print_info(self):
        table_format = "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}"
        print(table_format.format("Name", "P", "Qdh+fc", "Qfuel", "Fuel", "Vfg"))
        print(table_format.format("-" * 20, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))
        print(table_format.format(self.name, round(self.P), round(self.Qdh+self.Qfgc), round(self.Qfuel), self.fuel, round(self.Vfg)))

    def estimate_rankine(self, plotting=False):
        Ptarget = self.P
        Qdh = self.Qdh
        A = State("A", self.psteam, self.Tsteam)

        max_iterations = 100
        pcond_guess = 5
        Pestimated = 0
        i = 0
        tol = 0.03
        while abs(Pestimated - Ptarget) > Ptarget*tol and i < max_iterations:
            pcond_guess = pcond_guess - 0.1
            B = State("B", p=pcond_guess, s=A.s, mix=True)
            C = State("C", pcond_guess, satL=True)
            msteam = Qdh/(B.h-C.h)
            Pestimated = msteam*(A.h-B.h)
            i += 1
        if i == max_iterations:
            print(self.name)
            raise ValueError("Couldn't estimate Rankine cycle!")

        Qboiler = msteam*(A.h-C.h)
        self.Qboiler = Qboiler
        self.P = Pestimated
        D = State("D", self.psteam, satL=True)
        self.states = A,B,C,D

        if plotting == True:
            self.plot_plant(A,B,C,D)

        if msteam is not None and Pestimated is not None and Qboiler > 0 and pcond_guess > 0:
            return
        else:
            print(self.name)
            print(self.states[0].T)
            raise ValueError("One or more of the variables (msteam, Pestimated, Qboiler, pcond_guess) is not positive.")
        
    def burn_fuel(self, technology_assumptions):
        # TODO: Check method for fuel=>flue gas. Also turn emission factors etc. into uncertainties X, assumptions.
        self.technology_assumptions = technology_assumptions

        self.Qfuel = 1 / technology_assumptions["eta_boiler"] * self.Qboiler

        if self.fuel == "B":
            self.fCO2 = technology_assumptions["fCO2_B"]
        elif self.fuel == "W":
            self.fCO2 = technology_assumptions["fCO2_W"]

        self.mCO2 = self.Qfuel * 0.400  # [tCO2/h]  old value was 0.355
        self.Vfg = 2000000 / 110 * self.mCO2 / (self.fCO2 / 0.04)  # [m3/h]

        return self.Vfg, self.fCO2
    
    def plot_plant(self, show=False):
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
        plt.figure(figsize=(8, 6))  
        for state in saturation_line:
            plt.plot(state[0], state[1], marker='.', linestyle='-', color='k', markersize=1)
        plt.xlim(0, 10)
        plt.ylim(0, 600)
        plt.xlabel('s [kJ/kgC]')
        plt.ylabel('T [C]')
        plt.title(f's,T cycle of CHP ({self.name})')
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

        if self.reboiler_steam != None:
            a,d = self.reboiler_steam
            a.plot(pressure=True)
            d.plot()
            draw_line(a, d, color='cornflowerblue')

        if show:
            plt.show()
        return
    
    def energy_penalty(self, MEA):
        dTreb = self.technology_assumptions["dTreb"]

        A,B,C,D = self.states
        mtot = self.Qboiler*1000 / (A.h-C.h) #WE HAVE TOO MUCH MASS; BECAUSE WE USE THE UPDATED QFUEL
        TCCS = MEA.get("Treb") + dTreb
        pCCS = steamTable.psat_t(TCCS)
        # Ta = steamTable.t_ps(pCCS,A.s)
        a = State("a",pCCS,s=A.s,mix=True) #NOTE: Debug? If mixed! You need to add a case if we are outside (in gas phase)
        d = State("d",pCCS,satL=True)
        mCCS = MEA.get("Qreb") / (a.h-d.h)
        mB = mtot-mCCS

        W = 0
        for Wi in ["Wpumps","Wcfg","Wc1","Wc2","Wc3","Wrefr1","Wrefr2","Wrecomp"]:
            W += MEA.get(Wi)

        if a.p > B.p: # Check Rankine plots, the new power output depends on the pressures of pDH and pCCS
            Pnew = mtot*(A.h-a.h) + mB*(a.h-B.h) - W #Subtract pump, comp work etc.
        else: 
            Pnew = mtot*(A.h-B.h) + mCCS*(B.h-a.h) - W

        Plost = (mtot*(A.h-B.h) - Pnew)/1000
        self.P = Pnew/1000

        Qnew = mB*(B.h-C.h)
        Qlost = (mtot*(B.h-C.h) - Qnew)/1000
        self.Qdh = Qnew/1000

        self.reboiler_steam = [a,d]
        return Plost, Qlost

class MEA_plant:
    def __init__(self, host_plant):
        self.host = host_plant
        self.data = None
        self.composite_curve = None
        self.dTmin = host_plant.technology_assumptions["dTmin"]
        self.rate = host_plant.technology_assumptions["rate"]
        self.QTdict = None
        self.economics = None

    def get(self, parameter):
        return self.data[parameter].values[0]

    def estimate_size(self, interpolations, Aspen_data):
        # df = Aspen_data #TODO: Move this outside of RDM loop? NO. We can't, since burn_fuel() relies on our technology_assumptions
        # X = df[['CO2%', 'Flow']]
        # y = df.drop(columns=['CO2%', 'Flow'])

        # model = MultiOutputRegressor(LinearRegression())
        # model.fit(X, y)

        # # NOTE: THE BELOW FOUR ROWS WORKED FOR OLD W2E REGRESSION WHERE "CO2%" and "Flow" WAS USED
        # y = Aspen_data.drop(columns=['CO2%', 'Flow']) # Need to keep this, to be able to name the columns of predicted_df
    
        # # print("Estimated flue gas volume: ", self.host.Vfg, " [m3/h], or ", self.host.Vfg/3600*0.8, " [kg/s]" )
        # new_input = pd.DataFrame({'CO2%': [self.host.fCO2*100], 'Flow': [self.host.Vfg/3600*0.8]})  # Fraction of CO2=>percentage, and massflow [kg/s], of flue gases

        # y = Aspen_data.drop(columns=['CO2', 'Flow', 'Rcapture']) 
        # new_input = pd.DataFrame({'CO2': [self.host.fCO2*100], 'Flow': [self.host.Vfg/3600*0.8], 'Rcapture': [self.rate]})

        # predicted_y = model.predict(new_input)
        # predicted_df = pd.DataFrame(predicted_y, columns=y.columns)

        # print("MEA plant is this big, after assuming densityFG = 0.8kg/m3: ")
        # print(predicted_df.head())
        # self.data = predicted_df
        # return 
        
        # Initialize a dictionary to store new y values for each column
        new_Flow = [self.host.Vfg/3600*0.8]
        if new_Flow[0] < 3: #NOTE: my interpolation function does not work below 3 kg/s / above 170kgs
            new_Flow = [3]
        if new_Flow[0] > 170:
            new_Flow = [170]

        new_Rcapture = [self.rate]
        new_y_values = {}

        # Calculate new y values for each column using interpolation functions
        for column_name, interp_func in interpolations.items():
            new_y = interp_func((new_Flow, new_Rcapture))
            new_y_values[column_name] = new_y

        # Create a DataFrame to store the new values
        new_data = pd.DataFrame({
            'Flow': new_Flow,
            'Rcapture': new_Rcapture,
            **new_y_values  # Unpack new y values dictionary
        })

        # print("MEA plant is this big, after assuming densityFG = 0.8kg/m3: ")
        # print(new_data.head())
        # print("--- WITH Qint1 = ", new_data["Qint1"], " check the Tint1 temperatures in csv MEA file")
        self.data = new_data
        return 

    def select_streams(self, consider_dcc):
        considered_streams = ['wash', 'strip', 'lean', 'int2', 'int1', 'dhx', 'dry', 'rcond', 'rint', 'preliq'] # For CHPs
        if consider_dcc:
            considered_streams.append('dcc') # For industrial cases with hot flue gases that first need to be cooled before entering the amine absorber

        stream_data = {}
        for component in considered_streams:
            stream_data[component] = {
                'Q': -self.get(f"Q{component}"),
                'Tin': self.get(f"Tin{component}")-273.15,
                'Tout': self.get(f"Tout{component}")-273.15
            }
        return stream_data

    def find_ranges(self, stream_data):
        temperatures = []
        for component, data in stream_data.items():
            temperatures.extend([data['Tin'], data['Tout']])

        unique_temperatures = list(dict.fromkeys(temperatures)) 
        unique_temperatures.sort(reverse=True)

        temperature_ranges = []
        for i in range(len(unique_temperatures) - 1):
            temperature_range = (unique_temperatures[i + 1], unique_temperatures[i])
            temperature_ranges.append(temperature_range)

        return temperature_ranges

    def merge_heat(self, stream_data):
        temperature_ranges = self.find_ranges(stream_data)

        composite_curve = [[0, temperature_ranges[0][1]]] # First data point has 0 heat and the highest temperature
        Qranges = []
        for temperature_range in temperature_ranges:
            Ctot = 0
            for component, data in stream_data.items():
                TIN = data['Tin']
                TOUT = data['Tout']
                Q = data['Q']
                C = Q/(TIN-TOUT)
                
                if TIN >= temperature_range[1] and TOUT <= temperature_range[0]:
                    Ctot += C

            Qrange = Ctot*(temperature_range[1]-temperature_range[0])
            Qranges.append(Qrange)
            composite_curve.append([sum(Qranges), temperature_range[0]])
        self.composite_curve = composite_curve
        return composite_curve
    
    def available_heat(self, composite_curve):
        Tsupp = self.host.technology_assumptions["Tsupp"]
        Tlow = self.host.technology_assumptions["Tlow"]
        
        shifted_curve = [[point[0], point[1] - self.dTmin] for point in composite_curve]
        curve = shifted_curve
        # Calculate the distances of each point from the line formed by connecting the endpoints. Find the elbow point (point of maximum curvature).
        def distance(p1, p2, p):
            x1, y1 = p1
            x2, y2 = p2
            x0, y0 = p
            return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances = [distance(curve[0], curve[-1], point) for point in curve]
        differences = np.diff(distances)
        max_curvature_index = np.argmax(differences) + 1

        # Finding low and high points:
        def linear_interpolation(curve, ynew):
            # Find the nearest points
            y_values = [point[1] for point in curve]    # NOTE: I think this is buggy for unfeasible composite curves, i.e. when some Qcool approach zero and we have weirds "kinks" in the composite curve
            nearest_index = min(range(len(y_values)), key=lambda i: abs(y_values[i] - ynew))
            x1, y1 = curve[nearest_index]
            if nearest_index == 0:
                x2, y2 = curve[1]
            elif nearest_index == len(curve) - 1:
                x2, y2 = curve[-2]
            else:
                x2, y2 = curve[nearest_index + 1]

            # Perform inverse linear interpolation
            if y2 == y1:
                return x1 + (x2 - x1) * (ynew - y1) / (y2 - y1)
            else:
                return x1 + (x2 - x1) * (ynew - y1) / (y2 - y1)
        
        Qsupp = linear_interpolation(curve, Tsupp)
        Qlow = linear_interpolation(curve, Tlow)
        Qpinch, Tpinch = curve[max_curvature_index][0], curve[max_curvature_index][1]

        self.QTdict = {
            "supp": [Qsupp, Tsupp],
            "low": [Qlow, Tlow],
            "pinch": [Qpinch, Tpinch]
        }

        Qrecovered = (Qpinch-Qsupp) + (Qlow-Qpinch)
        return Qrecovered/1000  #MW

    def CAPEX_costs(self, economic_assumptions, escalate=True):
        self.economics = economic_assumptions
        X = economic_assumptions

        CAPEX = X['alpha'] * (self.host.Vfg / 3600) ** X['beta']  #[MEUR] (Eliasson, 2021) who has cost year=2016. Nm3 ~= m3 for our flue gases! NOTE: THIS IS FOR 13% CO2, generic study
        CAPEX *= X['CEPCI']                                         #NOTE: this CAPEX represents TDC. We may or may not add escalation to this.
        fixedOPEX = X['fixed'] * CAPEX                              #% of TDC

        if escalate:
            CAPEX *= 1 + X['ownercost']
            escalation = sum((1 + X['rescalation']) ** (n - 1) * (1 / X['yexpenses']) for n in range(1, X['yexpenses'] + 1))
            cfunding = sum(X['WACC'] * (X['yexpenses'] - n + 1) * (1 + X['rescalation']) ** (n - 1) * (1 / X['yexpenses']) for n in range(1, X['yexpenses'] + 1))
            CAPEX *= escalation + cfunding                      #This is TASC.
            #TODO: we can include system_contingency if we consider HPC a FOAK. This could be +0-30% maybe? (Towards, Rubin)

        annualization = (X['i'] * (1 + X['i']) ** X['t']) / ((1 + X['i']) ** X['t'] - 1)
        aCAPEX = annualization * CAPEX                      #[MEUR/yr]

        return CAPEX, aCAPEX, fixedOPEX
    
    def OPEX_costs(self, economic_assumptions, Plost, Qlost, Qrecovered):
        X = economic_assumptions
        energyOPEX = (Plost * X['celc'] + (Qlost - Qrecovered) * X['cheat']) * X['duration'] * 10 ** -6
        otherOPEX = self.get("Makeup") * X['cMEA'] * 3600 * X['duration'] * 10 ** -6
        return energyOPEX, otherOPEX
    
    def plot_streams(self, stream_data, show=False):
        plt.figure(figsize=(10, 8))
        num_streams = len(stream_data)
        colormap = plt.cm.get_cmap('Paired', num_streams)  # Using the Tab20B colormap

        for i, (component, data) in enumerate(stream_data.items()):
            Q = data['Q']
            TIN = data['Tin']
            TOUT = data['Tout']

            # Assigning a unique color to each component based on Tab20B colormap
            color = colormap(i)

            plt.plot([0, Q], [TIN, TOUT], marker='o', color=color, label=f"Q{component}")

        # Adding labels and title
        plt.xlabel('Q [kW]')
        plt.ylabel('Temperature [C]')
        plt.title('Streams To Cool')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
    
    def plot_hexchange(self, show=False): #TODO: Move temperatures to the CHP or the MEA class!
        Qsupp, Tsupp = self.QTdict["supp"]
        Qlow, Tlow = self.QTdict["low"]
        Qpinch, Tpinch = self.QTdict["pinch"]

        plt.figure(figsize=(10, 8))
        composite_curve = self.composite_curve
        shifted_curve = [[point[0], point[1] - self.dTmin] for point in composite_curve]
        (Qpinch-Qsupp) + (Qlow-Qpinch)

        plt.plot([0, self.get("Qreb")], [self.get("Treb"), self.get("Treb")], marker='*', color='#a100ba', label='Qreboiler')
        plt.plot([point[0] for point in composite_curve], [point[1] for point in composite_curve], marker='o', color='red', label='T of CCS streams')
        plt.plot([point[0] for point in shifted_curve], [point[1] for point in shifted_curve], marker='o', color='pink', label='T shifted')
        plt.plot([Qpinch, Qlow], [Tpinch, Tlow], marker='x', color='#069AF3', label='Qlowgrade')
        plt.plot([Qpinch, Qsupp], [Tpinch, Tsupp], marker='x', color='blue', label='Qhighgrade')
        plt.plot([Qlow, composite_curve[-1][0]], [20, 15], marker='o', color='#0000FF', label='Cooling water') # NOTE: hard-coded CW temps.

        plt.text(26000, 55, f'dTmin={round(self.dTmin,2)} C', color='black', fontsize=12, ha='center', va='center')
        plt.text(26000, 115, f'Qreb={round(self.get("Qreb")/1000)} MW', color='#a100ba', fontsize=12, ha='center', va='center')       
        plt.text(5000, 60, f'Qhighgrade={round((Qpinch-Qsupp)/1000)} MW', color='#0000FF', fontsize=12, ha='center', va='center')
        plt.text(5000, 40, f'Qlowgrade={round((Qlow-Qpinch)/1000)} MW', color='#069AF3', fontsize=12, ha='center', va='center')
        plt.text(10000, 15, f'Qcoolingwater={round((composite_curve[-1][0]-Qlow)/1000)} MW', color='#0000FF', fontsize=12, ha='center', va='center')

        plt.xlabel('Q [kW]')
        plt.ylabel('T [C]')
        plt.title(f'[{self.host.name}] Heat exchange between composite curve and district heating')
        plt.legend()
        if show:
            plt.show()
        return
