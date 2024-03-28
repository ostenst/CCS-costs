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
    def __init__(self, name, fuel=None, Qdh=0, P=0, Qfgc=0, ybirth=0, Tsteam=0, psteam=0, Qfuel=0, Vfg=0, mCO2=0):
        self.name = name
        self.fuel = fuel
        self.Qdh = Qdh
        self.P = P
        self.Qfgc = Qfgc
        self.ybirth = ybirth
        self.Tsteam = Tsteam
        self.psteam = psteam
        self.Qfuel = Qfuel
        self.Vfg = Vfg
        self.fCO2 = 0
        self.mCO2 = mCO2
        self.states = None

    def print_info(self):
        table_format = "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}"
        print(table_format.format("Name", "P", "Qdh+fc", "Qfuel", "Fuel", "Vfg"))
        print(table_format.format("-" * 20, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))
        print(table_format.format(self.name, round(self.P), round(self.Qdh+self.Qfgc), round(self.Qfuel), self.fuel, round(self.Vfg)))

    def estimate_performance(self, plotting=False):
        Ptarget = self.P
        Qdh = self.Qdh
        A = State("A", self.psteam, self.Tsteam)

        max_iterations = 100
        pcond_guess = 4
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

        Qfuel = msteam*(A.h-C.h)
        self.Qfuel = Qfuel
        self.P = Pestimated
        D = State("D", self.psteam, satL=True)
        self.states = A,B,C,D

        if plotting == True:
            self.plot_plant(A,B,C,D)

        if msteam is not None and Pestimated is not None and Qfuel > 0 and pcond_guess > 0:
            return
        else:
            raise ValueError("One or more of the variables (msteam, Pestimated, Qfuel, pcond_guess) is not positive.")

    def plot_plant(self, capture_states=None, show=False):
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

        if capture_states != None:
            a,d = capture_states
            a.plot(pressure=True)
            d.plot()
            draw_line(a, d, color='cornflowerblue')

        if show:
            plt.show()
        return
    
    def energy_penalty(self, MEA, dTreb):
        A,B,C,D = self.states
        mtot = self.Qfuel*1000 / (A.h-C.h)
        TCCS = MEA.get("Treb") + dTreb
        pCCS = steamTable.psat_t(TCCS)
        Ta = steamTable.t_ps(pCCS,A.s)

        a = State("a",pCCS,s=A.s,mix=True) #NOTE: Debug? If mixed! You need to add a case if we are outside (in gas phase)
        d = State("d",pCCS,satL=True)
        mCCS = MEA.get("Qreb") / (a.h-d.h)
        mB = mtot-mCCS

        W = 0
        for Wi in ["Wpumps","Wcfg","Wc1","Wc2","Wc3","Wrefr1","Wrefr2","Wrecomp"]:
            W += MEA.get(Wi)

        Pnew = mtot*(A.h-a.h) + mB*(a.h-B.h) - W #Subtract pump, comp work etc.
        Plost = self.P - Pnew/1000
        self.P = Pnew/1000

        Qnew = mB*(B.h-C.h)/1000
        Qlost = self.Qdh - Qnew
        self.Qdh = Qnew

        reboiler_steam = [a,d]
        return Plost, Qlost, reboiler_steam

class MEA_plant:
    def __init__(self, host_plant, construction_year=2024, currency_factor=0, discount=0.08, lifetime=25):
        self.host = host_plant
        self.data = None
        self.composite_curve = None
        self.dTmin = None
        self.QTdict = None
        self.economics = None

    def get(self, parameter):
        return self.data[parameter].values[0]

    def estimate_size(self, model, Aspen_data):
        # df = Aspen_data #TODO: Move this outside of RDM loop?
        # X = df[['CO2%', 'Flow']]
        # y = df.drop(columns=['CO2%', 'Flow'])

        # model = MultiOutputRegressor(LinearRegression())
        # model.fit(X, y)

        y = Aspen_data.drop(columns=['CO2%', 'Flow']) # Need to keep this, to be able to name the columns of predicted_df
    
        # print("Estimated flue gas volume: ", self.host.Vfg, " [m3/h], or ", self.host.Vfg/3600*0.8, " [kg/s]" )
        new_input = pd.DataFrame({'CO2%': [self.host.fCO2*100], 'Flow': [self.host.Vfg/3600*0.8]})  # Fraction of CO2=>percentage, and massflow [kg/s], of flue gases
        predicted_y = model.predict(new_input)
        predicted_df = pd.DataFrame(predicted_y, columns=y.columns)

        # print("MEA plant is this big: ")
        # print(predicted_df.head())
        self.data = predicted_df
        return 

    def identify_streams(self, consider_dcc):
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
    
    def available_heat(self, composite_curve, Tsupp, Tlow):

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
            y_values = [point[1] for point in curve]
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

        CAPEX = X['alpha'] * (self.host.Vfg / 3600) ** X['beta']  #[MEUR] (Eliasson, 2021) who has cost year=2016. Nm3 ~= m3 for our flue gases!
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

        plt.text(26000, 55, f'dTmin={self.dTmin} C', color='black', fontsize=12, ha='center', va='center')
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