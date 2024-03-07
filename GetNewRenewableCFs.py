import os, copy, datetime, pandas as pd, datetime as dt, numpy as np
from os import path
from GetRenewableCFs import *

#Output: dfs of wind and solar generation (8760 dt rows, arbitrary cols)
def getNewRenewableCFs(genFleet, tgtTz, weatherYears, currYear, pRegionShapes, 
            nonCCReanalysis, climateChange, cesmMembers, interconn, maxCapPerTech, reDensity):
    #Importing single RE timeseries
    if not climateChange or len(cesmMembers)==1: 
        newCfs,maxCapPerTech = getSingleNewRenewableCFsTimeseries(genFleet, tgtTz, weatherYears, currYear, 
            pRegionShapes, nonCCReanalysis, climateChange, cesmMembers, interconn, maxCapPerTech, reDensity)
    #Importing multiple RE timeseries for different ensemble members
    else: 
        newCfsAll = list()
        for cesmMember in cesmMembers:
            newCfs,maxCapPerTech = getSingleNewRenewableCFsTimeseries(genFleet, tgtTz, weatherYears, currYear, 
                pRegionShapes, nonCCReanalysis, climateChange, cesmMember, interconn, maxCapPerTech, reDensity)
            newCfs.columns = pd.MultiIndex.from_product([[cesmMember], newCfs.columns], names=['ensembleMember', 'locs'])
            newCfsAll.append(newCfs)
        #Combine into 1 array
        newCfs = pd.concat(newCfsAll,axis=1)
    return newCfs,maxCapPerTech

#Output: dfs of wind and solar generation (8760 dt rows, arbitrary cols)
def getSingleNewRenewableCFsTimeseries(genFleet, tgtTz, weatherYears, currYear, 
        pRegionShapes, nonCCReanalysis, climateChange, cesmMember, interconn, maxCapPerTech, reDensity):
    if currYear > 2050 and climateChange == False: currYear = 2050
    
    #Isolate wind & solar units
    windUnits, solarUnits = getREInFleet('Wind', genFleet), getREInFleet('Solar', genFleet)
    
    #Get list of wind / solar sites in region.
    lats,lons,cf,latlonRegion = loadData(weatherYears,pRegionShapes,cesmMember,interconn,climateChange,nonCCReanalysis)

    #Import available land per grid cell
    wArea,sArea = importWSLandAvailable(interconn,climateChange)

    #Match existing gens to CFs (use this for figuring our spare capacity available at each coordinate given existing generators)
    if not climateChange: get_cf_index(windUnits, lats, lons),get_cf_index(solarUnits, lats, lons)
    else: getCFIndexCC(windUnits,lats,lons,cf),getCFIndexCC(solarUnits,lats,lons,cf)

    #Calculate new CFs for given met year, but (in some cases) setting dt index to currYear
    yrForCFs = weatherYears if (climateChange or (nonCCReanalysis and interconn == 'WECC')) else [currYear] #if not CC, relabel fixed met year to future year; if CC, have many years
    
    #Filter out new CF locations to states being analyzed
    stateBounds = latlonRegion.reset_index(drop=True)
    stateBounds.columns = range(stateBounds.columns.size)
    cf = enforceStateBounds(cf, stateBounds)

    #Calculate new CF sites    
    windCfs,maxNewWindCaps = calcNewCfs(windUnits, lats, lons, cf, 'wind', yrForCFs, wArea, reDensity)
    solarCfs,maxNewSolarCaps = calcNewCfs(solarUnits, lats, lons, cf, 'solar', yrForCFs, sArea, reDensity)

    #Modify max capacity of wind and solar using cell-specific available capacity
    for re in ['Wind','Solar']: maxCapPerTech.pop(re)
    maxCaps = pd.concat([maxNewWindCaps,maxNewSolarCaps])
    for i in maxCaps.index: maxCapPerTech[i] = maxCaps[i]

    #Shift to target timezone
    if not climateChange: windCfs, solarCfs = shiftTz(windCfs, tgtTz, currYear, 'wind'), shiftTz(solarCfs, tgtTz, currYear, 'solar')

    return pd.concat([windCfs, solarCfs], axis=1),maxCapPerTech

#Import available land area per grid cell based on Grace WU SL work (see gis-work channel; https://www.pnas.org/doi/10.1073/pnas.2204098120)
def importWSLandAvailable(interconn,climateChange,wsSLScenario='sl1'):
    #Import available area per grid cell based on RE dataset and spatial resolution (WECC and CONUS are the same in WECC except at edges of WECC)
    #Used the following two lines for Hari paper 2 runs around July 29, 2023
    wArea = pd.read_csv(os.path.join('Data','WindSolarSiting',wsSLScenario + '_wind_cesm2_areaassessment_reproject_hariRuns.csv'),index_col=0,header=0)
    sArea = pd.read_csv(os.path.join('Data','WindSolarSiting',wsSLScenario + '_solar_cesm2_areaassessment_reproject_hariRuns.csv'),index_col=0,header=0)
    if climateChange:
        if interconn == 'WECC':
            print('using old wecc wind and solar siting files for hari runs see getnewrenewablescf file')
            #wArea = pd.read_csv(os.path.join('Data','WindSolarSiting','wind_'+wsSLScenario+'_cpa_area_cesm2_4326.csv'),index_col=0,header=0)
            #sArea = pd.read_csv(os.path.join('Data','WindSolarSiting','solar_'+wsSLScenario+'_cpa_area_cesm2_4326.csv'),index_col=0,header=0)
            #wArea = pd.read_csv(os.path.join('Data','WindSolarSiting',wsSLScenario + '_wind_cesm2_wecc.csv'),index_col=0,header=0)
            #sArea = pd.read_csv(os.path.join('Data','WindSolarSiting',wsSLScenario + '_solar_cesm2_wecc.csv'),index_col=0,header=0)
        else:
            wArea = pd.read_csv(os.path.join('Data','WindSolarSiting',wsSLScenario + '_wind_cesm2_conus.csv'),index_col=0,header=0)
            sArea = pd.read_csv(os.path.join('Data','WindSolarSiting',wsSLScenario + '_solar_cesm2_conus.csv'),index_col=0,header=0)
    else:
        if interconn == 'WECC': 
            wArea = pd.read_csv(os.path.join('Data','WindSolarSiting',wsSLScenario + '_wind_era5_wecc.csv'),index_col=0,header=0)
            sArea = pd.read_csv(os.path.join('Data','WindSolarSiting',wsSLScenario + '_solar_era5_wecc.csv'),index_col=0,header=0)
        else:
            wArea = pd.read_csv(os.path.join('Data','WindSolarSiting',wsSLScenario + '_wind_era5_conus.csv'),index_col=0,header=0)
            sArea = pd.read_csv(os.path.join('Data','WindSolarSiting',wsSLScenario + '_solar_era5_conus.csv'),index_col=0,header=0)

    #Fill NAs with 0
#    wArea['area'].fillna(0,inplace=True),sArea['area'].fillna(0,inplace=True)
    wArea['ia_area'].fillna(0,inplace=True),sArea['ia_area'].fillna(0,inplace=True)

    #Isolate lat & lon
    locs = wArea['centroid_text'].str.split(' ')
    lons,lats = [float(v[0].split('(')[1]) for v in locs.values],[float(v[1].split(')')[0]) for v in locs.values]
    lats,lons = pd.Series(lats,index=locs.index),pd.Series(lons,index=locs.index)
    wArea['Latitude'],wArea['Longitude'] = lats,lons+360

    locs = sArea['centroid_text'].str.split(' ')
    lons,lats = [float(v[0].split('(')[1]) for v in locs.values],[float(v[1].split(')')[0]) for v in locs.values]
    lats,lons = pd.Series(lats,index=locs.index),pd.Series(lons,index=locs.index)
    sArea['Latitude'],sArea['Longitude'] = lats,lons+360

    return wArea,sArea

def calcNewCfs(existingGens, lats, lons, cf, re, yrForCFs, availableArea, reDensity): 
    #Pull number of time steps from CF array (indexed by lat/lon/time, so time is idx 2)
    tSteps = cf[re].shape[2] 
    f = 'H' if tSteps >=8760 else 'D' #f = 'H' if tSteps == 8760 else 'D' #2/1/23
    #For each lat/lon, check existing capacity and, if spare room for more renewables, add CFs
    cfs,maxNewCaps,maxedLocs = dict(),dict(),list()
    latDiff,lonDiff = abs(lats[1]-lats[0]),abs(lons[1]-lons[0])
    for latIdx in range(len(lats)):
        for lonIdx in range(len(lons)):
            lat, lon = lats[latIdx], lons[lonIdx]

            #Get max available area for RE at location
            dists = calcHaversine(lat,lon,availableArea['Latitude'],availableArea['Longitude'])
            maxArea = availableArea.iloc[dists.argmin()]['ia_area'] #in square meters!
            maxCapacity = maxArea*reDensity[re.capitalize()]/1e6 #MW [=m^2 * W/m^2 * MW/1e6W]

            #Get generators & total capacity at location
            gensAtLoc = existingGens.loc[(existingGens['lat idx'] == latIdx) & (existingGens['lon idx'] == lonIdx)]
            existingCap = gensAtLoc['Capacity (MW)'].astype(float).sum()

            #Get CFs @ coordinates
            coordCfs = cf[re][latIdx, lonIdx, :]

            #Filter out any coordinates with all NANs (these are sites not in EI)
            if not np.isnan(coordCfs).all():
                #Filter out coords w/ no gen
                if coordCfs.sum() > 0:
                    #If can fit more capacity in cell, save CFs and update max capacity
                    if existingCap < maxCapacity: 
                        cfs[re + 'lat' + str(lat) + 'lon' + str(lon)] = coordCfs
                        maxNewCaps[re + 'lat' + str(lat) + 'lon' + str(lon)] = maxCapacity - existingCap
                    #Zero out CFs if current capacity exceeds max possible capacity in that cell
                    else:
                        cfs[re + 'lat' + str(lat) + 'lon' + str(lon)] = 0
                        maxNewCaps[re + 'lat' + str(lat) + 'lon' + str(lon)] = 0
                        maxedLocs.append((lat,lon))
    print('No new ' + re + ' capacity at:', maxedLocs)
    #Create df of CFs and series of max capacities
    cfs,maxNewCaps = pd.DataFrame(cfs),pd.Series(maxNewCaps)
    #Create dt index
    idx = pd.date_range('1/1/' + str(yrForCFs[0]) + ' 0:00','12/31/' + str(yrForCFs[-1]) + ' 23:00', freq=f)
    #Drop leap year extra time steps from index and df
    idx = idx.drop(idx[(idx.month == 2) & (idx.day == 29)]) #2/1/23
    cfs = cfs.iloc[:len(idx)]
    #Add datetime index to cfs df
    cfs.index = idx
    return cfs,maxNewCaps

def calcHaversine(lat1,lon1,lat2,lon2,R=6371): #R = Earth radius (km)
    lat1,lon1,lat2,lon2 = lat1*np.pi/180,lon1*np.pi/180,lat2*np.pi/180,lon2*np.pi/180
    a = np.sin((lat1-lat2)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon1-lon2)/2)**2
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c

def enforceStateBounds(cf, stateBounds):
   for re in cf:
        for row in stateBounds.index:
            for col in stateBounds.columns:
                cf[re][row,col] *= stateBounds.loc[row,col]
    # plotCFs(cf)
   return cf



# import matplotlib.pyplot as plt
# def plotCFs(cf):
#     avgCfs,lats,lons = np.zeros((23,23)),np.zeros(23),np.zeros(23)
#     for re in cf:
#         cfs = cf[re]
#         for lat in range(cfs.shape[0]):
#             for lon in range(cfs.shape[1]):
#                 avgCfs[lat,lon] = cfs[lat,lon].mean()
#                 # lats[lat] = cfs['lat'][lat]
#                 # lons[lon] = cfs['lon'][lon]

#         plt.figure()
#         ax = plt.subplot(111)
#         im = ax.contourf(avgCfs,cmap='plasma')#,extent = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)])
#         cbar = ax.figure.colorbar(im, ax=ax)#, ticks=np.arange(vmin,vmax,int((vmax-vmin)/5)))
#         plt.title(re)
#     plt.show()
