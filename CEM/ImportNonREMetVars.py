import os, xarray as xr

#Import met vars for derates & TDFORs of thermal units (T, rh, pressure)
def importNonREMet(weatherYears,cesmMembers,nonCCReanalysis):
    if cesmMembers != None: metVars = importCESMMetVars(weatherYears,cesmMembers[0])
    elif nonCCReanalysis: metVars = importERA5MetVars(weatherYears)
    else: metVars = None
    return metVars

def importERA5MetVars(weatherYears):
    #Load FORs file that contains temperatures (in C) and slice down to weather years
    temps = xr.open_dataset(os.path.join('Data','ERA5','wecc_FOR_ERA5_hourly_PST.nc'))
    temps = temps.sel(time=slice(str(weatherYears[0])+"-01-01", str(weatherYears[-1])+"-12-31"))
    return temps

def importCESMMetVars(weatherYears,cesmMember):
    print(cesmMember)
    temps = xr.open_dataset(os.path.join('Data','CESM','derate_fields_' + cesmMember + '.nc'))
    temps = temps.sel(time=slice(str(weatherYears[0])+"-01-01", str(weatherYears[-1])+"-12-31")) #T in K    
    return temps

