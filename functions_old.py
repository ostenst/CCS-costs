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
            self.fCO2 = 0.16
        elif Fuel == "W":
            self.fCO2 = 0.11
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

        # FLUEGASES #TODO: calculate based on johanna+thunman
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

        if capture_states != None:
            a,d = capture_states
            a.plot(pressure=True)
            d.plot()
            draw_line(a, d, color='cornflowerblue')

        if show:
            plt.show()
        return
    
    def energy_penalty(self, MEA):
        A,B,C,D = self.states
        mtot = self.Qfuel*1000 / (A.h-C.h)
        TCCS = MEA.get("Treb") + 10 #Assuming dTmin = 10 in reboiler MEA.get
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
    
    def heat_integrate(self, Qhighgrade, Qlowgrade, Qcw):
        self.Qdh += (Qhighgrade + Qlowgrade)/1000
        mcoolingwater = Qcw/(4.18*(20-15)) # Assuming cw properties, TODO: mcoolingwater is very high!? But the heat capacity Qcw seems reasonable.
        return self.Qdh, mcoolingwater

class MEA_plant:
    def __init__(self, host_plant, construction_year=2024, currency_factor=0, discount=0.08, lifetime=25):
        self.host = host_plant
        self.data = None
        # self.data = Aspen_data #self.data should point to the individual plant data, not the total Aspen_data
        # self.mfluegas = Aspen_data["M_FLUEGAS"].values[0]
        # self.rhofluegas = Aspen_data["RHO_FLUEGAS"].values[0]
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
        self.composite_curve = None

    def estimate_size(self, Aspen_data):
        df = Aspen_data
        X = df[['CO2%', 'Flow']]
        y = df.drop(columns=['CO2%', 'Flow'])

        model = MultiOutputRegressor(LinearRegression())
        model.fit(X, y)

        new_input = pd.DataFrame({'CO2%': [self.host.fCO2*100], 'Flow': [self.host.Vfg/3600*0.8]})  # Fraction of CO2=>percentage, and massflow [kg/s], of flue gases
        predicted_y = model.predict(new_input)
        predicted_df = pd.DataFrame(predicted_y, columns=y.columns)
        print("MEA plant is this big: ")
        print(predicted_df.head())
        self.data = predicted_df
        return 

    def get(self, parameter):
        return self.data[parameter].values[0]

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
            
            #cost2024 = cost2015*(CEPCI2024/CEPCI2015), check what is a good year, e.g. 2022? 2024 is a very rough estimate.
            #Apply exchange rate of the year considered! e.g. to USD (e.g. average of the year)

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
        mMEA = self.get("M_MAKEUP")      #kg/s
        mMEA = mMEA/1000 * 3600*self.duration       #tMEA/a
        cost_MEA = mMEA * 1.7                       #kEUR/a

        mH2O = self.get("M_H2OIN")       #kg/s
        mH2O = mH2O/1000 * 3600*self.duration       #tH2O/a
        cost_H2O = mH2O * 0.02/1000                 #kEUR/a 
       
        cost_maintenance = TPC * 0.045              #kEUR/a 
        cost_labor = 411                            #kEUR/a 

        return TCR, aCAPEX, cost_H2O, cost_MEA, cost_maintenance, cost_labor

    def specific_annualized(self, cost):
        cost = cost/self.annualization              #kEUR,annualized
        mCO2 = self.get("MCO2_CO2OUT")              #kgCO2/s
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
    def identify_streams(self, stream_indicies):
        stream_data = {}
        for component in stream_indicies:
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

        unique_temperatures = list(dict.fromkeys(temperatures))  # Remove duplicates by converting to dictionary keys and back to list
        unique_temperatures.sort(reverse=True)

        temperature_ranges = []
        for i in range(len(unique_temperatures) - 1):
            temperature_range = (unique_temperatures[i + 1], unique_temperatures[i])
            temperature_ranges.append(temperature_range)

        return temperature_ranges

    def merge_heat(self, temperature_ranges, stream_data):
        # Should also return the actual composite curve points!
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

    def plot_composite(self, temperature_ranges, Qranges, Tmax, Tsupp, Thigh, Tlow, dTmin=10, show=False):
        plt.figure(figsize=(10, 8))
        Tmax = Tmax - dTmin
        current_q = 0
        Qinteresting = []
        for i, temp_range in enumerate(temperature_ranges):
            next_q = current_q + Qranges[i]

            for Tinteresting in [Tmax, Tsupp, Thigh, Tlow]: # I look for all temps in each range
                if temp_range[0]-dTmin <= Tinteresting <= temp_range[1]-dTmin:
                    q_value = current_q + (Tinteresting - (temp_range[1]-dTmin))*(next_q - current_q)/(temp_range[0] - temp_range[1])
                    Qinteresting.append(q_value)

            labels = ["Treal","Tshifted"] if i == 0 else [None, None]  
            plt.plot([current_q, next_q], [temp_range[1], temp_range[0]], marker='o', color='#FF0000', label=labels[0])
            plt.plot([current_q, next_q], [temp_range[1]-dTmin, temp_range[0]-dTmin], color='#FA8072', marker='o', label=labels[1])
            current_q = next_q

        # Plot heat sinks
        plt.plot([0, self.get("Qreb")], [self.get("Treb"), self.get("Treb")], marker='*', color='#a100ba', label='Qreboiler')
        plt.plot([Qinteresting[0], Qinteresting[2]], [Tsupp, Thigh], marker='o', color='#0000FF', label='DH high-grade')
        plt.plot([Qinteresting[2], Qinteresting[3]], [Thigh, Tlow], marker='o', color='#069AF3', label='DH low-grade')
        plt.plot([Qinteresting[3], current_q], [20, 15], marker='o', color='#0000FF', label='Cooling water') # NOTE: hard-coded CW temps.

        # Annotate interesting info
        Q_highgrade = Qinteresting[2]-Qinteresting[0] # dT are given by our assumptions! Composite curve has (120+10)C->(61+10)C , DH has 61C->86C
        Q_lowgrade  = Qinteresting[3]-Qinteresting[2]
        Q_cw = current_q - Qinteresting[3]
        plt.text(26000, 70, f'dTmin={dTmin} C', color='black', fontsize=12, ha='center', va='center')
        plt.text(26000, 115, f'Qreb={round(self.get("Qreb")/1000)} MW', color='#a100ba', fontsize=12, ha='center', va='center')       
        plt.text(5000, 70, f'Qhighgrade={round(Q_highgrade/1000)} MW', color='#0000FF', fontsize=12, ha='center', va='center')
        plt.text(5000, 52, f'Qlowgrade={round(Q_lowgrade/1000)} MW', color='#069AF3', fontsize=12, ha='center', va='center')
        plt.text(10000, 20, f'Qcoolingwater={round(Q_cw/1000)} MW', color='#0000FF', fontsize=12, ha='center', va='center')

        # Adding labels and title
        plt.xlabel('Q [kW]')
        plt.ylabel('Temperature [C]')
        plt.title('Hot Composite Curve vs Cold Utilities')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()

    def available_heat(self, temperature_ranges, Qranges, Tmax, Tsupp, Thigh, Tlow, dTmin=10):
        # Looks for the real DH temperatures, at the shifted (dTmin) composite curve
        # NOTE: this assumes heat exchange is possible between Tmax and Thigh, ensure this is true.
        Tmax = Tmax - dTmin # This temp. belongs to the composite curve and should be shifted. Should be ok now!
        current_q = 0
        Qinteresting = []
        for i, temp_range in enumerate(temperature_ranges): #IM IN ONE RANGE
            
            next_q = current_q + Qranges[i]  
            for Tinteresting in [Tmax, Tsupp, Thigh, Tlow]: # I look for all temps in each range
                if temp_range[0]-dTmin <= Tinteresting <= temp_range[1]-dTmin:
                    q_value = current_q + (Tinteresting - (temp_range[1]-dTmin))*(next_q - current_q)/(temp_range[0] - temp_range[1])
                    Qinteresting.append(q_value)
            current_q = next_q
        Tend = temp_range[0] 

        Q_highgrade = Qinteresting[2]-Qinteresting[0] # dT are given by our assumptions! Composite curve has (120+10)C->(61+10)C , DH has 61C->86C
        Q_lowgrade  = Qinteresting[3]-Qinteresting[2]
        Q_cw = current_q - Qinteresting[3]

        return Q_highgrade, Q_lowgrade, Q_cw, Tend
    
    def available_heat2(self, composite_curve, Tsupp, Tlow, dTmin=10):

        plt.clf()
        shifted_curve = [[point[0], point[1] - dTmin] for point in composite_curve]
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

        # # Plot the elbow point on the curve
        # plt.plot([point[0] for point in composite_curve], [point[1] for point in composite_curve], marker='o', color='pink')
        # plt.plot([point[0] for point in shifted_curve], [point[1] for point in shifted_curve], marker='o', color='red')
        # plt.plot(curve[max_curvature_index][0], curve[max_curvature_index][1], marker='x', color='blue', markersize=10, label='Elbow Point')
        # plt.scatter(Qlow, Tlow, marker='x', color='blue', label='Tlow')
        # plt.scatter(Qsupp, Tsupp, marker='x', color='blue', label='Tlow')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Curve with Elbow Point')
        # plt.legend()
        # plt.show()
        # print("Elbow point coordinates:", curve[max_curvature_index])

        return

    def exchanger_areas(self, Qhighgrade, Qlowgrade, Qcw, U, dTmin, Tmax, Tsupp, Tlow, Tend):
        Alow = Qlowgrade/(U*dTmin) #Constant, see composite curve

        dT1 = Tmax - Tsupp
        dT2 = dTmin
        dTlm = (dT1-dT2)/math.log(dT1/dT2)
        Ahigh = Qhighgrade/(U*dTlm)

        dT1 = (Tlow+dTmin) - 20         # ASSUMING CW of 15C->20C
        dT2 = Tend - 15
        dTlm = (dT1-dT2)/math.log(dT1/dT2)
        Acw = Qcw/(U*dTlm)
        return Alow, Ahigh, Acw
    
def find_points_on_curve(curve, x_values):
    """
    Find the y-values on the curve corresponding to specific x-values.

    Parameters:
        curve (list): List of points representing the curve in the form [[x1, y1], [x2, y2], ...].
        x_values (list): List of x-values for which to find the corresponding y-values.

    Returns:
        dict: A dictionary containing the x-values as keys and their corresponding y-values as values.
    """
    points = {}
    for x_target in x_values:
        # Find the two closest points on the curve to the specified x-value
        closest_points = sorted(curve, key=lambda point: abs(point[0] - x_target))[:2]
        # Perform linear interpolation to find the y-value corresponding to the x-value
        x1, y1 = closest_points[0]
        x2, y2 = closest_points[1]
        y_value = y1 + (y2 - y1) * (x_target - x1) / (x2 - x1)
        # Add the interpolated y-value to the dictionary
        points[x_target] = y_value
    return points