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
    CHP.estimate_performance()                                                                      # <--- BEGIN RDM_MODEL() EFTER THIS ESTIMATION? YES! Send CHP as argument...?
    eta_boiler = 0.87
    CHP.Qfuel *= 1/eta_boiler #LHV! Är det 10-12MJ/kg? (Johanna), kanske 8 om fuktigare
    
    if CHP.fuel == "B": #TODO: fixa sambandet bränslekomposition->CO2frac och rökgasvolym
        CHP.fCO2 = 0.16 #Höga? Johanna tror 8-10% waste, och 13-16% biochips
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

    # RECOVER EXCESS HEAT TO DH
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

    # DETERMINE CAPEX                                                                     <--- BELOW CALCS NEEDS TO TURN INTO MEA.methods()
    # TODO: determine social vs private cost (maybe cascade? how does this increase the costs for end-consumers?)(compare with Levinh, 2019)(EU fines for non-compliance?)
    # NPV = sum(t=1->n)( cash(t)/(1+i)^t )

    alpha = 6.12
    beta = 0.6336
    CAPEX = alpha*(CHP.Vfg/3600)**beta # [MEUR] (Eliasson, 2021) who has cost year = 2016 #TODO: Need T,p of fluegases to convert to Nm3 , before sending to
    CEPCI = 600/550 #(https://link.springer.com/article/10.1007/s10973-021-10833-z)
    CAPEX *= CEPCI              # This is time-adjusted TDC=TIC
    fixedOPEX = 0.06 * CAPEX #MEUR/yr

    owners_cost = 0.2
    CAPEX *= 1+owners_cost      # This is TOC

    WACC = 0.05
    yexpenses = 3
    rescalation = 0.03

    escalation = 0
    for n in range(1,yexpenses+1):
        escalation += (1+rescalation)**(n-1)*(1/yexpenses)
    cfunding = 0
    for n in range(1,yexpenses+1):
        cfunding += WACC*(yexpenses-n+1)*(1+rescalation)**(n-1)*(1/yexpenses)
    CAPEX *= escalation+cfunding # This is TASC

    i = 0.075
    t = 25
    annualization = (i*(1+i)**t )/((1+i)**t-1) 
    aCAPEX = annualization*CAPEX #MEUR/yr

    # DETERMINE var-OPEX
    celc = 40
    cheat = 15
    duration = 8000
    energyOPEX = (Plost*celc + Qlost*cheat)*duration *10**-6 
    cMEA = 2 # EUR/kg
    otherOPEX = MEA.get("Makeup")*cMEA *3600*duration *10**-6 


    # SYNTHESIZE AND PLOT
    cost_labels = ['aCAPEX', 'fixedOPEX', 'energyOPEX', 'otherOPEX']
    costs = [aCAPEX, fixedOPEX, energyOPEX, otherOPEX] #MEUR/yr
    costs_specific = [x*10**6 / (CHP.mCO2*duration) for x in costs]     #EUR/tCO2

    emission_intensity = 0.355*CHP.Qfuel / (CHP.Qdh+CHP.Qfgc+CHP.P)     #tCO2/MWhoutput, NOTE: these are total emissions produced, only 90% of these are captured.
    consumer_cost = emission_intensity * 0.9 * sum(costs_specific)      #EUR/MWhoutput
    print(consumer_cost)


    # ADD COST ESCALATION (READ ELIASON: IS THIS ABSOLUTE CAPEX=TPC? SO WE ESCALATE IT?)
    # capture plant: he summarises EIC of individual components into a TotalInstalledCost. But EIC includes condingencies already!(Ali) And in Ali, TIC=TDC, but misses TOC/TASC escalation
    # liquefaction plant: estimated TotalDirectCost by scaling from Deng. Then multiply by contingeiny(Deng) to get TIC.
    # issue: they already include congingencies, but I want this as an uncertainty!
    # solution: open ApepndixA => to see the equipment, AppendixB => their cost functions, open Ali2019 => equipment factors

    # We can add TOC: up to +20%of TDC (NETL, https://www.osti.gov/servlets/purl/1567736). This includes start-up, working/inventory capital, land, securing financing
    # => introduce 1 uncertainty 0-20%
    # We can add TASC, but this requires WACC etc. This tells us how the capital is incurred during the expenditure period, and how it escalates.
    # => introduce 3 uncertainties: i=capital escalation rate, dy=years of capital expenditure, WACC=capital distribution
    # This could work... but maybe this is incompatible with the annualization?
    
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
