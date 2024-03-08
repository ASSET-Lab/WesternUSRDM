import sys, os, csv, operator, copy, time, random, warnings, numpy as np, datetime as dt, pandas as pd
from os import path; from gams import *
from SetupGeneratorFleet import setupGeneratorFleet,compressAndAddSizeDependentParams
from AddCoolingTypes import addCoolingTypes
from ProcessHydro import processHydro
from UpdateFuelPriceFuncs import updateFuelPricesAndCosts
from ImportDemand import importDemand
from DemandFuncsCE import getHoursForCE
from IsolateDataForCE import isolateDataInCEHours,isolateDataInCEBlocks
from ImportNewTechs import getNewTechs
from RetireUnitsCFPriorCE import retireUnitsCFPriorCE
from CreateFleetForCELoop import createFleetForCurrentCELoop
from GetRenewableCFs import getREGen
from GetNewRenewableCFs import getNewRenewableCFs
from AddWSSitesToNewTechs import addWSSitesToNewTechs
from ProcessCEResults import saveCEBuilds,addNewGensToFleet,addNewLineCapToLimits,retireUnitsByCF
from ScaleRegResForAddedWind import scaleRegResForAddedWind
from CombinePlants import combineWindSolarStoPlants
from GAMSAddSetToDatabaseFuncs import *
from GAMSAddParamToDatabaseFuncs import *
from InitializeOnOffExistingGensCE import initializeOnOffExistingGens
from ReservesWWSIS import calcWWSISReserves
from GetIncResForAddedRE import getIncResForAddedRE
from SaveCEOperationalResults import saveCapacExpOperationalData
from WriteTimeDependentConstraints import writeTimeDependentConstraints
from WriteBuildVariable import writeBuildVariable
from CreateEmptyReserveDfs import createEmptyReserveDfs
from SetupTransmissionAndZones import setupTransmissionAndZones, defineTransmissionRegions
from DefineReserveParameters import defineReserveParameters
from CalculateDerates import calculateLineThermalDerates,calculatePlantCapacityDerates
from ImportPRMCapacityAdjustments import importPRMCapacityAdjustments

# SET OPTIONS
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

# SCALARS
mwToGW = 1000
lbToShortTon = 2000

# ##############################################################################
# ##### UNIVERSAL PARAMETERS ###################################################
# ##############################################################################
def setKeyParameters(climateChange):
    # ### RUNNING ON SC OR LOCAL
    runOnSC = True                                     # whether running on supercomputer

    # ### START YEAR, END YEAR, AND STEPS FOR CE
    startYear, endYear, yearStepCE = 2022,2041,2

    # ### RE UPSAMPLING
    reDownFactor = 0                     # FRACTION of wind & solar sites per region dropped (0 = no sites dropped; .7 = 70% sites dropped); sites dropped with worst CFs
    
    # ### BUILD LIMITS
    yearIncDACS,yearIncHydrogen,yearIncCCS,yearIncNuclear = 2041,2041,2031,2021  #year to allow investment in certain technologies

    # ### CE OPTIONS
    if climateChange: numBlocks, daysPerBlock, daysPerPeak = 1, 365*yearStepCE, 1    # num rep time blocks, days per rep block, and days per peak block in CE
    else: numBlocks, daysPerBlock, daysPerPeak = 4,7,1
    removeHydro = False                                  #whether to remove hydropower from fleet & subtract generation from demand, or to include hydro as dispatchable in CE w/ gen limit
    stoInCE,seasStoInCE = False,False                    # whether to allow new storage,new seasonal storage in CE model
             
    return (runOnSC,startYear,endYear,yearStepCE,reDownFactor,yearIncDACS,yearIncHydrogen,yearIncCCS,yearIncNuclear,
        numBlocks,daysPerBlock,daysPerPeak,removeHydro,stoInCE,seasStoInCE)

def setNonCCWeatherData():
    demandScen = 'REFERENCE'                        # NREL EFS demand scenario: 'REFERENCE','HIGH','MEDIUM' (ref is lower than med)
    nonCCReanalysis = True                                # == True: use reanalysis as renewable data source, == False: use NSDRB and WTK
    nonCCWeatherYear = [2012] if nonCCReanalysis else [2012]  #EFS, WTK, NSRDB: 2012 weather year. Should use 2012 if using EFS.
    return nonCCWeatherYear,demandScen,nonCCReanalysis

def stateAssumptions(interconn,yearStepCE,cesmMembers,reBuildRateMultiplier=1,thermalBuildRateMultiplier=4): 
    # ### MAX BUILDS
    #Max builds (MW) per region or, for wind & solar, grid point
    areaPerLatLongBox = 9745 #km^2 per degree lat x long (https://www.usgs.gov/faqs/how-much-distance-does-a-degree-minute-and-second-cover-your-maps?qt-news_science_products=0#qt-news_science_products)
    windDens,solarDens = .9,5.7 #W/m^2 (equiv to MW/km^2); https://www.seas.harvard.edu/news/2018/10/large-scale-wind-power-would-require-more-land-and-cause-more-environmental-impact
    reDensity = {'Wind':windDens,'Solar':solarDens}
    #Set max capacity per tech. To turn off storage, use (stoInCE,seasStoInCE) flags above. 
    maxCapPerTech = {'Wind': areaPerLatLongBox * windDens, 'Solar': areaPerLatLongBox * solarDens, 
        'Combined Cycle': 1500*thermalBuildRateMultiplier*yearStepCE,'Storage': 0, 'Dac': -0, 'CCS': 1500*thermalBuildRateMultiplier*yearStepCE, 
        'Nuclear': 0, 'Battery Storage': .1, 'Hydrogen': .1, 'Transmission': 99999} #across WECC, 2400 MW from 2019-2021 & 4100 MW from 2015-2021 of new NGCC + NGCT. 2020 annual max (1500 MW)
    #Max wind & solar builds (MW) per interconnection & region
    reCAGR = .3 #compounded annual growth rate for max RE builds - applied to histMaxREBuildInterconn
    if interconn == 'WECC': #see SetupTransmissionAndZones for list of zones
        maxREPerZone = {'Wind': {'NWPP_NE':99999,'CAMX':99999,'Desert_Southwest':99999,'NWPP_Central':99999,'NWPP_NW':99999},
                        'Solar':{'NWPP_NE':99999,'CAMX':99999,'Desert_Southwest':99999,'NWPP_Central':99999,'NWPP_NW':99999}} 
        # histMaxREBuildInterconn = {'Wind':6700/2*yearStepCE*reBuildRateMultiplier,'Solar':7500/1*yearStepCE*reBuildRateMultiplier} #max build per interconnection per CE run. WECC 2022 WARA: 7.5 GW solar expected to be added in 2023; EIA 860: 6.7 GW & 5.8 GW wind & solar added in 2020 & 2021 across WECC
        histMaxREBuildInterconn = {'Wind':2600*reBuildRateMultiplier,'Solar':3400*reBuildRateMultiplier} #historic WECC max build - max of 2020-2022 annual max builds per EIA 860 across WECC. Wind & solar MW adds 2020/2021/2022: 1900/2600/870 & 2700/3100/3400. So use 2600 & 3400. Annual growth rate applied later.
    else:
        maxREPerZone = {'Wind': {'SERC':99999,'NY':99999,'NE':99999,'MISO':99999,'PJM':99999,'SPP':99999},
                        'Solar':{'SERC':99999,'NY':99999,'NE':99999,'MISO':99999,'PJM':99999,'SPP':99999}} 
        histMaxREBuildInterconn = {'Wind':21000/3*reBuildRateMultiplier,'Solar':15000/3*reBuildRateMultiplier} #max build per interconnection per CE run. EI using 860 data: 15 GW solar 2019-2021; 21 GW 2019-2021 
    
    # ### CO2 EMISSION CAPS [https://www.eia.gov/environment/emissions/state/, table 3]
    if interconn == 'ERCOT': co2EmsInitial =  130594820     #METRIC TONS. Initial emission for ERCOT: 130594820.
    elif interconn == 'EI': 
        co2EmsInitial,wsGenFracOfDemandInitial,wGenFracOfDemandInitial =  1043526617,0,0 #1043526617 is from CEM output (2023 w/out investments); 1274060000 is from EIA data
        print('if running EI w/ renewable generation requirements, need to populate wsGenFracOfDemandInitial & wGenFracOfDemandInitial')
    elif interconn == 'WECC': co2EmsInitial,wsGenFracOfDemandInitial,wGenFracOfDemandInitial =  172788598,7,2   #using output from CEM for CESM2 runs #2019 emissions: 248800000 METRIC TONS. wa,or,ca,nm,az,nv,ut,co,wy,id,mt

    # ### CE AND UCED/ED OPTIONS
    balAuths = 'full'                                   # full: run for all BAs in interconn. TODO: add selection of a subset of BAs. [10/31 MC note: Papa Yaw has this code]
    compressFleet = True                                                # whether to compress fleet
    tzAnalysis = {'ERCOT':'CST','EI':'EST','WECC':'PST'}[interconn]     # timezone for analysis
    fuelPrices = importFuelPrices('Reference case')                     # import fuel price time series
    transmissionEff = 0.95                                              # efficiency of transmission between zones (https://ars.els-cdn.com/content/image/1-s2.0-S2542435120305572-mmc1.pdf)
           #efficiency of storage
    greenField = False      #whether to run CE greenfield (no existing units) or not

    # ### CE OPTIONS
    runCE,ceOps = True,'ED'                           # ops are 'ED' or 'UC' (econ disp or unit comm constraints)
    includeRes = False                                  # whether to include reserves in CE & dispatch models (if False, multiplies reserve timeseries by 0)
    retireByAge = False                                  # whether to retire by age or not
    prmBasis = 'demand'                      # whether basis for planning reserve margin is peak demand ('demand') or net demand ('netdemand')

    retirementCFCutoff = .3                             # retire units w/ CF lower than given value
    ptEligRetCF = ['Coal Steam']                        # which plant types retire based on capacity factor (economics)
    discountRate = 0.07 

    # ### DEMAND FLEXIBILITY PARAMETERS
    demandShifter = 0                                   # Percentage of hourly demand that can be shifted
    demandShiftingBlock = 4                             # moving shifting demand window (hours)

    # ### WARNINGS OR ERRORS
    if ceOps == 'UC': sys.exit('CEwithUC.gms needs to be updated for DACS operations - add DACS constraints and include gentechs set')

    return (balAuths,co2EmsInitial,compressFleet,tzAnalysis,fuelPrices,transmissionEff,
        runCE,ceOps,includeRes,retireByAge,prmBasis,retirementCFCutoff,ptEligRetCF,discountRate,maxCapPerTech,
        maxREPerZone,histMaxREBuildInterconn,reCAGR,wsGenFracOfDemandInitial,wGenFracOfDemandInitial,reDensity,greenField,demandShifter,demandShiftingBlock)

def storageAssumptions():
    stoMkts = 'energy'                            # energy,res,energyAndRes - whether storage participates in energy, reserve, or energy and reserve markets
    stoFTLabels = ['Energy Storage','Pumped Storage']
    stoDuration = {'Energy Storage':'st','Hydrogen':'lt','Battery Storage':'st','Flywheels':'st','Batteries':'st','Pumped Storage':'st'} # mapping plant types to short-term (st) or long-term (lt) storage
    stoPTLabels = [pt for pt in stoDuration ]
    initSOCFraction = {pt:{'st':0,'lt':0}[dur] for pt,dur in stoDuration.items()} # get initial SOC fraction per st or lt storage
    stoMinSOC = 0     # min SOC
    stoEff = 0.81
    return stoMkts,stoFTLabels,stoDuration,stoPTLabels,initSOCFraction,stoMinSOC,stoEff

def importFuelPrices(fuelPriceScenario):
    fuelPrices = pd.read_csv(os.path.join('Data', 'Energy_Prices_Electric_Power.csv'), skiprows=4, index_col=0)
    fuelPrices = fuelPrices[[col for col in fuelPrices if fuelPriceScenario in col]]
    fuelPrices.columns = [col.split(':')[0] for col in fuelPrices.columns]
    return fuelPrices    
# ###############################################################################
# ###############################################################################
# ###############################################################################

# ###############################################################################
# ##### MASTER FUNCTION #########################################################
# ###############################################################################
#Main function to call. 
#Inputs: interconnection (EI, WECC, ERCOT); CO2 emissions in final year as fraction
#of initial CO2 emissions.
def macroCEM(interconn,co2EndPercent,wsGenFracOfDemand,windGenFracOfDemand,cesmMembers,climateChange,prm=0): #prm: integer giving % planning reserve margin
    # Set key parameters
    (runOnSC,startYear,endYear,yearStepCE,reDownFactor,yearIncDACS,yearIncHydrogen,yearIncCCS,yearIncNuclear,
        numBlocks,daysPerBlock,daysPerPeak,removeHydro,stoInCE,seasStoInCE) = setKeyParameters(climateChange)

    # Set assumptions
    nonCCWeatherYear,demandScen,nonCCReanalysis = setNonCCWeatherData()
    (balAuths,co2EmsInitial,compressFleet,tzAnalysis,fuelPrices,transmissionEff,
        runCE,ceOps,includeRes,retireByAge,prmBasis,retirementCFCutoff,ptEligRetCF,discountRate,
        maxCapPerTechOrig,maxREPerZone,histMaxREBuildInterconn,reCAGR,wsGenFracOfDemandInitial,wGenFracOfDemandInitial,
        reDensity,greenField,demandShifter,demandShiftingBlock) = stateAssumptions(interconn,yearStepCE,cesmMembers)
    stoMkts,stoFTLabels,stoDuration,stoPTLabels,initSOCFraction,stoMinSOC,stoEff = storageAssumptions()
    (regLoadFrac, contLoadFrac, regErrorPercentile, flexErrorPercentile, regElig, contFlexInelig, regCostFrac,
        rrToRegTime, rrToFlexTime, rrToContTime) = defineReserveParameters(stoMkts, stoFTLabels)

    # Create results directory
    resultsDirAll = interconn+'C'+str(co2EndPercent)+'RE'+str(wsGenFracOfDemand)+'W'+str(windGenFracOfDemand)+('EM'+cesmMembers[0] if climateChange else '') #+'PRM'+str(prm)
    if not os.path.exists(resultsDirAll): os.makedirs(resultsDirAll)
    pd.Series(co2EmsInitial).to_csv(os.path.join(resultsDirAll,'initialCO2Ems.csv'))

    # Setup initial fleet and demand
    (genFleet, compressedGens, transRegions, pRegionShapes, lineLimits, lineDists, lineCosts) = getInitialFleetAndTransmission(startYear, 
        fuelPrices, compressFleet, resultsDirAll, regElig, regCostFrac, stoMinSOC, greenField, interconn, balAuths, 
        contFlexInelig, stoFTLabels, stoPTLabels, stoEff, stoInCE, cesmMembers)

    # Run CE and/or ED/UCED
    for currYear in range(startYear, endYear, yearStepCE):
        # Get weather years
        weatherYears = list(range(currYear-yearStepCE+1,currYear+1)) if climateChange else nonCCWeatherYear #+1 so, e.g. 2021-2030

        # Set CO2 cap
        currCo2Cap = co2EmsInitial + (co2EndPercent/100*co2EmsInitial - co2EmsInitial)/((endYear-1) - startYear) * (currYear - startYear)
        currCo2Cap *= len(weatherYears) #adjust annual cap if running more than 1 year!
        currWSGenFracOfDemand = wsGenFracOfDemandInitial + wsGenFracOfDemand/(endYear-1 - startYear) * (currYear - startYear)
        currWindGenFracOfDemand = wGenFracOfDemandInitial + windGenFracOfDemand/(endYear-1 - startYear) * (currYear - startYear)
        print('Entering year ', currYear, ' with CO2 cap (million tons):', round(currCo2Cap/1e6),'\t and RE & wind requirement (%):',round(currWSGenFracOfDemand),round(currWindGenFracOfDemand))

        # Set maximum RE additions using annual CAGR from initial max historic builds while accounting for # years included in CE
        maxREInInterconn = dict()
        for re in histMaxREBuildInterconn: maxREInInterconn[re] = sum([histMaxREBuildInterconn[re] * (1 + reCAGR) ** (currYear - yr - startYear) for yr in range(yearStepCE)])

        # Create results directory
        resultsDir = os.path.join(resultsDirAll,str(currYear))
        if not os.path.exists(resultsDir): os.makedirs(resultsDir)
        
        # Get electricity demand profile
        demand = importDemand(weatherYears,demandScen,currYear,transRegions,cesmMembers)
        demand.to_csv(os.path.join(resultsDir,'demandInitial'+str(currYear)+'.csv'))

        # Run CE
        if currYear > startYear and runCE:
            print('Starting CE')
            #Initialize results & inputs
            if currYear == startYear + yearStepCE: priorCEModel, priorHoursCE, genFleetPriorCE = None, None, None
            (genFleet, genFleetPriorCE, lineLimits,priorCEModel, priorHoursCE) = runCapacityExpansion(genFleet, demand, currYear, weatherYears, prm, prmBasis,
                                        discountRate, fuelPrices, currCo2Cap, numBlocks, daysPerBlock, daysPerPeak,
                                        retirementCFCutoff, retireByAge, tzAnalysis, resultsDir,
                                        maxCapPerTechOrig, regLoadFrac, contLoadFrac, regErrorPercentile, flexErrorPercentile,
                                        rrToRegTime, rrToFlexTime, rrToContTime, regElig, regCostFrac, ptEligRetCF,
                                        genFleetPriorCE, priorCEModel, priorHoursCE, stoInCE, seasStoInCE,
                                        ceOps, stoMkts, initSOCFraction, includeRes, reDownFactor, demandShifter,
                                        demandShiftingBlock, runOnSC, interconn, yearIncDACS, yearIncHydrogen,yearIncCCS, yearIncNuclear, transRegions,pRegionShapes,
                                        lineLimits, lineDists, lineCosts, contFlexInelig, 
                                        nonCCReanalysis, stoFTLabels, transmissionEff, removeHydro, climateChange, cesmMembers,
                                        maxREPerZone, maxREInInterconn, compressedGens, currWSGenFracOfDemand, currWindGenFracOfDemand, reDensity,
                                        co2EmsInitial)
# ###############################################################################
# ###############################################################################
# ###############################################################################

# ###############################################################################
# ###### SET UP INITIAL FLEET AND DEMAND ########################################
# ###############################################################################
def getInitialFleetAndTransmission(startYear, fuelPrices, compressFleet, resultsDir, regElig, regCostFrac, 
        stoMinSOC, greenField, interconn, balAuths, contFlexInelig, stoFTLabels, stoPTLabels, stoEff, stoInCE, cesmMembers):
    # GENERATORS
    genFleet = setupGeneratorFleet(interconn, startYear, fuelPrices, stoEff, stoMinSOC, stoFTLabels, stoInCE, cesmMembers)
    #Modify retirement year for Diablo Canyon given recent extension
    if 'Diablo Canyon' in genFleet['Plant Name'].unique():
        print('Extending Diablo Canyon lifetime to 2035')
        genFleet.loc[genFleet['Plant Name']=='Diablo Canyon','Retirement Year'] = 2035

    # ADD COOLING TYPES TO GENERATORS TO CAPTURE DERATINGS WHEN RUNNING CLIMATE SCENARIOS
    genFleet = addCoolingTypes(genFleet, interconn)

    # DEFINE TRANSMISSION REGIONS
    transRegions = defineTransmissionRegions(interconn, balAuths)

    # TRANSMISSION
    genFleet, transRegions, limits, dists, costs, pRegionShapes = setupTransmissionAndZones(genFleet, transRegions, interconn, balAuths)
    for df, l in zip([limits, dists, costs],['Limits', 'Dists', 'Costs']): df.to_csv(os.path.join(resultsDir, 'transmission' + l + 'Initial.csv'))
    genFleet.to_csv(os.path.join(resultsDir, 'genFleetInitialPreCompression.csv'))

    # COMBINE GENERATORS FOR SMALLER GEN FLEET AND ADD SIZE DEPENDENT PARAMS (COST, REG OFFERS, UC PARAMS)
    genFleet,compressedGens = compressAndAddSizeDependentParams(genFleet, compressFleet, regElig, contFlexInelig, regCostFrac, stoFTLabels, stoPTLabels)

    # IF GREENFIELD, ELIMINATE EXISTING GENERATORS EXCEPT TINY NG, WIND, & SOLAR PLANT (TO AVOID CRASH IN LATER FUNCTIONS)
    if greenField: genFleet = stripDownGenFleet(genFleet, greenField)

    # IF RUNNING CESM, CONVERT LONGITUDE TO 0-360 FROM -180-180
    if cesmMembers != None: genFleet['Longitude'] = genFleet['Longitude']%360

    # SAVE FILES
    genFleet.to_csv(os.path.join(resultsDir,'genFleetInitial.csv')),compressedGens.to_csv(os.path.join(resultsDir,'compressedUnitsFromGenFleet.csv'))
    return genFleet, compressedGens, transRegions, pRegionShapes, limits, dists, costs
# ###############################################################################
# ###############################################################################
# ###############################################################################

# ###############################################################################
# ###### RUN CAPACITY EXPANSION #################################################
# ###############################################################################
def runCapacityExpansion(genFleet, demand, currYear, weatherYears, prm, prmBasis, discountRate, fuelPrices, currCo2Cap, numBlocks,
                         daysPerBlock, daysPerPeak, retirementCFCutoff, retireByAge, tzAnalysis, resultsDirOrig, maxCapPerTechOrig,
                         regLoadFrac,contLoadFrac, regErrorPercentile, flexErrorPercentile, rrToRegTime, rrToFlexTime,  rrToContTime,
                         regElig, regCostFrac, ptEligRetCF, genFleetPriorCE, priorCEModel, priorHoursCE, stoInCE, seasStoInCE,
                         ceOps, stoMkts, initSOCFraction, includeRes, reDownFactor, demandShifter, demandShiftingBlock, runOnSC,
                         interconn, yearIncDACS, yearIncHydrogen,yearIncCCS, yearIncNuclear, transRegions, pRegionShapes, lineLimits, lineDists, lineCosts, contFlexInelig,
                         nonCCReanalysis, stoFTLabels, transmissionEff, removeHydro, climateChange, cesmMembers,
                         maxREPerZone, maxREInInterconn, compressedGens, currWSGenFracOfDemand, currWindGenFracOfDemand, reDensity, co2EmsInitial):
    # ###############CREATE RESULTS DIRECTORY FOR CE RUN AND SAVE INITIAL INPUTS
    resultsDir = os.path.join(resultsDirOrig, 'CE')
    if not os.path.exists(resultsDir): os.makedirs(resultsDir)
    print('Entering CE loop for year ' + str(currYear))
    lineLimits.to_csv(os.path.join(resultsDir,'lineLimitsForCE' + str(currYear) + '.csv'))
    pd.Series(currCo2Cap).to_csv(os.path.join(resultsDir,'co2CapCE' + str(currYear) + '.csv'))
    pd.Series(currWSGenFracOfDemand).to_csv(os.path.join(resultsDir,'wsGenFracOfDemandCE' + str(currYear) + '.csv'))
    pd.Series(currWindGenFracOfDemand).to_csv(os.path.join(resultsDir,'windGenFracOfDemandCE' + str(currYear) + '.csv'))
    pd.Series(maxREInInterconn).to_csv(os.path.join(resultsDir,'maxREInInterconn' + str(currYear) + '.csv'))
    maxCapPerTech = maxCapPerTechOrig.copy()

    # ###############PREPARE INPUTS FOR CEM
    # Whether IRA tax credits are still in effect
    print('set IRA to false for hari runs for cesm paper see macrocem towards top of runcapacityexpansion function')
    #ira = (extract0dVarResultsFromGAMSModel(priorCEModel,'vCO2emsannual') > co2EmsInitial*.25) if priorCEModel != None else True #IRA ends when emissions reach 75% reducion relative to 2022
    #pd.Series(ira).to_csv(os.path.join(resultsDir,'iraInEffectCE' + str(currYear) + '.csv'))
    ira = False
    if priorCEModel != None:
        print(extract0dVarResultsFromGAMSModel(priorCEModel,'vCO2emsannual'),co2EmsInitial*.25,extract0dVarResultsFromGAMSModel(priorCEModel,'vCO2emsannual') > co2EmsInitial*.25)
    print('********************IRA:',ira)

    # Update new technology and fuel price data    
    newTechsCE = getNewTechs(regElig, regCostFrac, currYear, stoInCE, seasStoInCE,
            fuelPrices, yearIncDACS, yearIncHydrogen,yearIncCCS, yearIncNuclear, transRegions, contFlexInelig, weatherYears,cesmMembers, genFleet, ira, lbToShortTon)
    genFleet = updateFuelPricesAndCosts(genFleet, currYear, fuelPrices, regCostFrac)

    # Retire units and create fleet for current CE loop
    if priorCEModel != None:                    # if not in first CE loop
        genFleet = retireUnitsCFPriorCE(genFleet, genFleetPriorCE, retirementCFCutoff,
            priorCEModel, priorHoursCE, ptEligRetCF, currYear)
    genFleet, genFleetForCE = createFleetForCurrentCELoop(genFleet, currYear, retireByAge)
    genFleet.to_csv(os.path.join(resultsDir, 'genFleetPreCEPostRetirements' + str(currYear) + '.csv'))
    genFleetForCE.to_csv(os.path.join(resultsDir, 'genFleetForCEPreRECombine' + str(currYear) + '.csv'))
    
    # Combine wind, solar, and storage plants by region
    genFleetForCE = combineWindSolarStoPlants(genFleetForCE)
    
    # Get renewable CFs by plant and region and calculate net demand by region
    print('Loading RE data')
    windGen, solarGen, windGenRegion, solarGenRegion = getREGen(genFleet, 
        tzAnalysis, weatherYears, currYear, pRegionShapes, nonCCReanalysis, climateChange, cesmMembers, interconn)
    netDemand = demand - windGenRegion - solarGenRegion

    # Remove hydropower generation from demand using net-demand-based heuristic
    genFleetForCE,hydroGen,demand = processHydro(genFleetForCE, demand, netDemand, weatherYears, removeHydro, cesmMembers) 
    genFleetForCE.to_csv(os.path.join(resultsDir, 'genFleetForCE' + str(currYear) + '.csv'))

    # Get hours included in CE model (representative + special blocks)
    (hoursForCE, planningReserve, blockWeights, socScalars, planningReserveHour, blockNamesChronoList, 
        lastRepBlockNames, specialBlocksPrior) = getHoursForCE(demand, netDemand, windGenRegion, solarGenRegion,
        daysPerBlock, daysPerPeak, currYear, resultsDir, numBlocks, prm, prmBasis, climateChange)

    # Get CFs for new wind and solar sites and add wind & solar sites to newTechs
    newCfs,maxCapPerTech = getNewRenewableCFs(genFleet, tzAnalysis, weatherYears, currYear, 
        pRegionShapes, nonCCReanalysis, climateChange, cesmMembers, interconn, maxCapPerTech, reDensity)
    newTechsCE,newCfs,maxCapPerTech = addWSSitesToNewTechs(newCfs, newTechsCE, pRegionShapes, reDownFactor, maxCapPerTech)
    pd.Series(maxCapPerTech).to_csv(os.path.join(resultsDir,'buildLimitsForCE' + str(currYear) + '.csv'))

    # Calculating thermal power plant & thermal line capacity deratings, FORs, and capacity eligibilities towards PRM
    print('Calculating deratings and capacity adjustments')
    capDerates,capDeratesTechs,newTechsCE = calculatePlantCapacityDerates(genFleetForCE, newTechsCE, demand, weatherYears, cesmMembers, compressedGens)
    lineDerates = calculateLineThermalDerates(lineLimits, demand) 
    fors,windFOR,solarFOR,forsTechs,prmEligWindSolar = importPRMCapacityAdjustments(genFleetForCE, 
                            newTechsCE, demand, prmBasis, interconn, nonCCReanalysis, weatherYears, compressedGens, cesmMembers)

    # Initialize which generators are on or off at start of each block of hours (useful if CE has UC constraints)
    onOffInitialEachPeriod = initializeOnOffExistingGens(genFleetForCE, hoursForCE, netDemand)

    # Set reserves for existing and incremental reserves for new generators
    print('Calculating reserves')
    if includeRes:
        cont, regUp, flex, regDemand, regUpSolar, regUpWind, flexSolar, flexWind = calcWWSISReserves(windGenRegion, solarGenRegion, demand, regLoadFrac,
                                                                                                     contLoadFrac, regErrorPercentile, flexErrorPercentile)
        regUpInc, flexInc = getIncResForAddedRE(newCfs, regErrorPercentile, flexErrorPercentile)
    else:
        cont, regUp, flex, regDemand, regUpSolar, regUpWind, flexSolar, flexWind, regUpInc, flexInc = createEmptyReserveDfs(windGenRegion, newCfs)

    # Get timeseries hours for CE (demand, wind, solar, new wind, new solar, reserves) & save dfs
    (demandCE, windGenCE, solarGenCE, newCfsCE, contCE, regUpCE, flexCE, regUpIncCE, 
        flexIncCE, forsCE, forsTechsCE, capDeratesCE, capDeratesTechsCE, lineDeratesCE) = isolateDataInCEHours(hoursForCE, 
        demand, windGenRegion, solarGenRegion, newCfs, cont, regUp, flex, regUpInc, flexInc, fors, forsTechs, capDerates, capDeratesTechs, lineDerates)
    
    # Get total hydropower generation potential by block for CE and, if running CESM, get monthly hydropower generation to populate monthly generation constraints
    [hydroGenCE] = isolateDataInCEBlocks(hoursForCE,hydroGen)
    if cesmMembers is not None: 
        hydroGenMonthlyCE = hydroGen.groupby(pd.Grouper(freq="M")).sum()    
        hydroGenMonthlyCE['dt'] = hydroGenMonthlyCE.index
        hydroGenMonthlyCE.index = ['month'+str(c)+'h' for c in range(hydroGenMonthlyCE.shape[0])]
        hydroGenMonthlyCE.to_csv(os.path.join(resultsDir,'hydroGenMonthlyCE'+str(currYear)+'.csv'))
    else:
        hydroGenMonthlyCE = 'NA'

    # Save CE inputs
    for df, n in zip([windGen, solarGen, windGenRegion, solarGenRegion, newCfs, demand, netDemand, cont, regUp, flex, regUpInc, flexInc, regDemand, regUpSolar, regUpWind, flexSolar, flexWind, hydroGen, fors, forsTechs, capDerates, capDeratesTechs, lineDerates],
                     ['windGen','solarGen','windGenRegion','solarGenRegion','windSolarNewCFs','demand','netDemand','contRes','regUpRes','flexRes','regUpInc','flexInc','regUpDemComp','regUpSolComp','regUpWinComp','flexSolComp','flexWinComp','hydroGen','fors','forTechs','capDerates','capDeratesTechs','lineDerates']):
        df.to_csv(os.path.join(resultsDir, n + 'FullYr' + str(currYear) + '.csv'))
    for df, n in zip([demandCE, windGenCE, solarGenCE, newCfsCE, newTechsCE, contCE, regUpCE, flexCE, regUpIncCE, flexIncCE, hydroGenCE, forsCE, forsTechsCE, capDeratesCE, capDeratesTechsCE, lineDeratesCE, hoursForCE],
                     ['demand', 'windGen', 'solarGen','windAndSolarNewCFs','newTechs','contRes','regUpRes','flexRes','regUpInc','flexInc','hydroGen','fors','forTechs','capDerates','capDeratesTechs','lineDerates','hoursByBlock']):
        df.to_csv(os.path.join(resultsDir, n + 'CE' + str(currYear) + '.csv'))
    for scalar, n in zip([windFOR,solarFOR,prmEligWindSolar,planningReserve,planningReserveHour],['windFOR','solarFOR','windSolarPRMElig','planningReserveCE','planningReserveHour']): 
        pd.Series(scalar).to_csv(os.path.join(resultsDir,n + str(currYear) + '.csv'))   
    pd.DataFrame([[k, v] for k, v in socScalars.items()],columns=['block','scalar']).to_csv(os.path.join(resultsDir,'socScalarsCE' + str(currYear) + '.csv'))

    # ###############SET UP CAPACITY EXPANSION
    # Create GAMS workspace and database. Parameters and sets are put into database below.
    ws, db, gamsFileDir = createGAMSWorkspaceAndDatabase(runOnSC)

    # Write .gms files for inclusion in CE. These files vary based on CE parameters, e.g. treatment of time.
    writeTimeDependentConstraints(blockNamesChronoList, stoInCE, seasStoInCE, gamsFileDir, ceOps, lastRepBlockNames, specialBlocksPrior, removeHydro, hydroGenMonthlyCE)
    writeBuildVariable(ceOps, gamsFileDir)

    # Enter sets and parameters into database
    genSet, hourSet, hourSymbols, zoneOrder, lineSet, zoneSet = edAndUCSharedFeatures(db, genFleetForCE, hoursForCE, demandCE, contCE,regUpCE,flexCE,
                                                                             demandShifter, demandShiftingBlock, rrToRegTime, rrToFlexTime, rrToContTime,
                                                                             solarGenCE, windGenCE, transRegions, lineLimits, transmissionEff, capDeratesCE,
                                                                             lineDeratesCE)  
    stoGenSet, stoGenSymbols = storageSetsParamsVariables(db, genFleetForCE, stoMkts, stoFTLabels)
    stoTechSet, stoTechSymbols = ceSharedFeatures(db, planningReserveHour, genFleetForCE, newTechsCE, planningReserve, discountRate, currCo2Cap,
                                      genSet, hourSet, hourSymbols, newCfsCE, maxCapPerTech, maxREPerZone, maxREInInterconn, regUpIncCE, flexIncCE, 
                                      stoMkts, lineDists, lineCosts, lineSet, zoneOrder, ceOps, interconn, stoFTLabels, zoneSet, 
                                      forsCE, forsTechsCE, capDeratesTechsCE, windFOR, solarFOR, prmEligWindSolar, currWSGenFracOfDemand, currWindGenFracOfDemand)
    if ceOps == 'UC': ucFeatures(db, genFleetForCE, genSet),
    ceTimeDependentConstraints(db, hoursForCE, blockWeights, socScalars, ceOps, onOffInitialEachPeriod, 
                genSet, genFleetForCE, stoGenSet,stoGenSymbols, newTechsCE, stoTechSet, stoTechSymbols, initSOCFraction,
                hydroGenCE, zoneSet, hydroGenMonthlyCE)

    # Run CE model
    print('Running CE for ' + str(currYear))
    capacExpModel, ms, ss = runGAMS('CEWith{o}.gms'.format(o=ceOps), ws, db)

    # ########## SAVE AND PROCESS CE RESULTS
    pd.Series([ms,ss],index=['ms','ss']).to_csv(os.path.join(resultsDir, 'msAndSsCE' + str(currYear) + '.csv'))
    saveCapacExpOperationalData(capacExpModel, genFleetForCE, newTechsCE, hoursForCE, transRegions, lineLimits, resultsDir, 'CE', currYear)
    newGens,newStoECap,newStoPCap,newLines = saveCEBuilds(capacExpModel, resultsDir, currYear)
    genFleet = addNewGensToFleet(genFleet, newGens, newStoECap, newStoPCap, newTechsCE, currYear)
    genFleet = retireUnitsByCF(genFleet,hoursForCE,capacExpModel,currYear,planningReserve,planningReserveHour)
    lineLimits = addNewLineCapToLimits(lineLimits, newLines)
    genFleet.to_csv(os.path.join(resultsDir, 'genFleetAfterCE' + str(currYear) + '.csv'))
    lineLimits.to_csv(os.path.join(resultsDir, 'lineLimitsAfterCE' + str(currYear) + '.csv'))

    return (genFleet, genFleetForCE, lineLimits, capacExpModel, hoursForCE)

# ###############################################################################
# ###############################################################################
# ###############################################################################

# ###############################################################################
# ################## GAMS FUNCTIONS #############################################
# ###############################################################################
def createGAMSWorkspaceAndDatabase(runOnSC):
    # currDir = os.getcwd()
    if runOnSC:
        gamsFileDir = 'GAMS'
        gamsSysDir = '/home/mtcraig/gams40_3'
    else:
        gamsFileDir = 'C:\\Users\\mtcraig\\Desktop\\Research\\Models\\MacroCEM\\GAMS'
        gamsSysDir = 'C:\\GAMS\\43'
        # gamsFileDir = r"C:\Users\atpha\Documents\Postdocs\Projects\NETs\Model\EI-CE\GAMS"
        # gamsSysDir = r"C:\GAMS\win64\30.2"
    ws = GamsWorkspace(working_directory=gamsFileDir, system_directory=gamsSysDir)
    db = ws.add_database()
    return ws, db, gamsFileDir

def runGAMS(gamsFilename, ws, db):
    t0 = time.time()
    model = ws.add_job_from_file(gamsFilename)
    opts = GamsOptions(ws)
    opts.defines['gdxincname'] = db.name
    model.run(opts, databases=db)
    ms, ss = model.out_db['pModelstat'].find_record().value, model.out_db['pSolvestat'].find_record().value
    if (int(ms) != 8 and int(ms) != 1) or int(ss) != 1: print('***********************Modelstat & solvestat:', ms, ' & ', ss, ' (ms1 global opt, ms8 int soln, ss1 normal)')
    print('Time (mins) for GAMS run: ' + str(round((time.time()-t0)/60)))
    return model, ms, ss

def edAndUCSharedFeatures(db, genFleet, hours, demand, contRes, regUpRes, flexRes, demandShifter, demandShiftingBlock, rrToRegTime, rrToFlexTime,
                          rrToContTime, hourlySolarGen, hourlyWindGen, transRegions, lineLimits, transmissionEff, capDeratesCE, lineDeratesCE, cnse=10000, co2Price=0):
    # SETS
    genSet = addGeneratorSets(db, genFleet)
    hourSet, hourSymbols = addHourSet(db, hours)
    zoneSet,zoneSymbols,zoneOrder = addZoneSet(db, transRegions)
    lineSet,lineSymbols = addLineSet(db, lineLimits)

    # PARAMETERS
    # Demand and reserves
    addDemandParam(db, demand, hourSet, zoneSet, demandShifter, demandShiftingBlock, mwToGW)
    addReserveParameters(db, contRes, regUpRes, flexRes, rrToRegTime, rrToFlexTime, rrToContTime, hourSet, zoneSet, mwToGW)

    # CO2 cap or price
    addCo2Price(db, co2Price)

    # Generators
    addGenParams(db, genFleet, genSet, mwToGW, lbToShortTon, zoneOrder)
    addCapacityDerates(db, genSet, hourSet, capDeratesCE)
    addExistingRenewableMaxGenParams(db, hourSet, zoneSet, hourlySolarGen, hourlyWindGen, mwToGW)
    addSpinReserveEligibility(db, genFleet, genSet)
    addCostNonservedEnergy(db, cnse)

    # Transmission lines
    addLineParams(db,lineLimits, transmissionEff, lineSet, zoneOrder, mwToGW)
    addLineDerates(db, lineSet, hourSet, lineDeratesCE)
    return genSet, hourSet, hourSymbols, zoneOrder, lineSet, zoneSet

def storageSetsParamsVariables(db, genFleet, stoMkts, stoFTLabels):
    (stoGenSet, stoGenSymbols) = addStoGenSets(db, genFleet, stoFTLabels)
    addStorageParams(db, genFleet, stoGenSet, stoGenSymbols, mwToGW, stoMkts)
    return stoGenSet, stoGenSymbols

def ed(db, socInitial, stoGenSet):
    addStorageInitSOC(db, socInitial, stoGenSet, mwToGW)

def ucFeatures(db, genFleet, genSet):
    addGenUCParams(db, genFleet, genSet, mwToGW)
    
def uc(db, stoGenSet, genSet, socInitial, onOffInitial, genAboveMinInitial, mdtCarriedInitial):
    addStorageInitSOC(db, socInitial, stoGenSet, mwToGW)
    addEguInitialConditions(db, genSet, onOffInitial, genAboveMinInitial, mdtCarriedInitial, mwToGW)

def ceSharedFeatures(db, planningReserveHour, genFleet, newTechs, planningReserve, discountRate, co2Cap, 
        genSet, hourSet, hourSymbols, newCfs, maxCapPerTech, maxREPerZone, maxREInInterconn, regUpInc, 
        flexInc, stoMkts, lineDists, lineCosts, lineSet, zoneOrder, ceOps, interconn, stoFTLabels, zoneSet,
        forsCE, forsTechsCE, capDeratesTechsCE, windFOR, solarFOR, prmEligWindSolar, currWSGenFracOfDemand, currWindGenFracOfDemand):
    # Sets
    addPlanningReserveHourSubset(db, planningReserveHour)
    addStorageSubsets(db, genFleet, stoFTLabels)
    (techSet, renewTechSet, stoTechSet, stoTechSymbols, thermalSet, dacsSet, CCSSet) = addNewTechsSets(db, newTechs)

    # Long-term planning parameters
    addPlanningReserveParam(db, planningReserve, mwToGW)
    addDiscountRateParam(db, discountRate)
    addCO2Cap(db, co2Cap)

    # New tech parameters
    addGenParams(db, newTechs, techSet, mwToGW, lbToShortTon, zoneOrder, True)
    addCapacityDerates(db, techSet, hourSet, capDeratesTechsCE, True)
    addHourlyFORs(db, forsCE, genSet, hourSet)
    addHourlyFORs(db, forsTechsCE, techSet, hourSet, True)
    addFORScalars(db, windFOR, solarFOR, prmEligWindSolar)
    addTechCostParams(db, newTechs, techSet, stoTechSet, mwToGW)
    addRenewTechCFParams(db, renewTechSet, hourSet, newCfs)
    addMaxNewBuilds(db, newTechs, thermalSet, renewTechSet, stoTechSet, dacsSet, CCSSet, zoneSet, maxCapPerTech, maxREPerZone, maxREInInterconn, mwToGW)
    addWSMinGen(db, currWSGenFracOfDemand, currWindGenFracOfDemand)
    if ceOps == 'UC': addGenUCParams(db, newTechs, techSet, mwToGW, True)
    addResIncParams(db, regUpInc, flexInc, renewTechSet, hourSet)
    addSpinReserveEligibility(db, newTechs, techSet, True)
    addStorageParams(db, newTechs, stoTechSet, stoTechSymbols, mwToGW, stoMkts, True)
    addNewLineParams(db, lineDists, lineCosts, lineSet, maxCapPerTech, zoneOrder, interconn, mwToGW)
    return stoTechSet, stoTechSymbols

def ceTimeDependentConstraints(db, hoursForCE, blockWeights, socScalars, ceOps, onOffInitialEachPeriod,
        genSet, genFleet, stoGenSet, stoGenSymbols, newTechs, stoTechSet, stoTechSymbols, 
        initSOCFraction, hydroGenCE, zoneSet, hydroGenMonthlyCE):
    addHourSubsets(db, hoursForCE, hydroGenMonthlyCE)
    addSeasonDemandWeights(db, blockWeights)
    addBlockSOCScalars(db, socScalars)
    if ceOps == 'UC': addInitialOnOffForEachBlock(db, onOffInitialEachPeriod, genSet)
    addStoInitSOCCE(db, genFleet, stoGenSet, stoGenSymbols, mwToGW, initSOCFraction)
    addStoInitSOCCE(db, newTechs, stoTechSet, stoTechSymbols, mwToGW, initSOCFraction, True)
    addHydroGenLimits(db, hydroGenCE, hydroGenMonthlyCE, zoneSet, mwToGW)

# ###############################################################################
# ###############################################################################
# ###############################################################################
