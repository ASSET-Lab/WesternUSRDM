import pandas as pd
import xarray as xr
import numpy as np
import os

import CalculateDerates as calc_derates
import transmission as tr
import economicdispatch as ed

from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.environ as pyo

def get_reqdfleet_info(baseDir,CE_fleet_name,
                       all_regions,
                       year_all=[str(yr) for yr in range(2022,2042,2)],                       
                       return_transmission=True                      
                      ):
    # Get the required columns, and the generator in required for, from the 
    # fleets coming out of CEM
    reqd_plants = ['NU','HD','ST','CT','CC','CCCCS','CTCCS','Solar','Wind','Storage']
    
    fleet_ds = xr.Dataset(coords={'time':np.empty(0,dtype='datetime64[ns]'),
                                  'GAMS Symbol':np.empty(0)                                 
                                 }
                         )
    transmission_dict = dict()
    
    for year in year_all:
        if year == '2022':
            CE_fleet_path = os.path.join(baseDir,CE_fleet_name)
            gens = pd.read_csv(CE_fleet_path+'/genFleetInitial.csv')
        else:    
            CE_fleet_path = os.path.join(baseDir,CE_fleet_name,str(year),'CE')
            gens = pd.read_csv(CE_fleet_path+'/genFleetAfterCE'+str(year)+'.csv')
        
        reqd_cols = ['PlantType','Latitude','Longitude','Nameplate Energy Capacity (MWh)',
                     'Capacity (MW)','region','GAMS Symbol','CoolingType','OpCost($/MWh)',
                     'Retired','CO2EmRate(lb/MMBtu)','Heat Rate (Btu/kWh)'
                    ]

        #fleet_df = gens[gens[].isin([False])][reqd_cols]
        fleet_df = gens[reqd_cols]

        # Rename new RE plant types to same name as existing plants
        fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('windlat')]['PlantType'].index,'PlantType'] = 'Onshore Wind'
        fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('solarlat')]['PlantType'].index,'PlantType'] = 'Solar PV'

        # Rename plant types to match FOR dict
        fleet_df = rename_plant_types(fleet_df)
        
        fleet_df.loc[fleet_df[~(fleet_df['PlantType'].isin(reqd_plants))].index,'PlantType'] = 'Other'
        
        fleet_df = fleet_df.set_index('GAMS Symbol')
        fleet_df = fleet_df.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        
        fleet_ds_year = fleet_df.to_xarray()
        fleet_ds_year = fleet_ds_year.expand_dims(dim={'time':[np.datetime64(year)]})
        
        fleet_ds = xr.merge([fleet_ds,fleet_ds_year])
        
        if return_transmission:
            transmission_limits = pd.DataFrame(np.zeros((5,5)),index=all_regions,columns=all_regions)

            if year == '2022':
                transmission_fromfile = pd.read_csv(CE_fleet_path+'/transmissionLimitsInitial.csv',index_col=0)
            else:
                transmission_fromfile = pd.read_csv(CE_fleet_path+'/lineLimitsAfterCE'+str(year)+'.csv',index_col=0)

            for idx,row in transmission_fromfile.iterrows():
                transmission_limits.loc[row['r']][row['rr']] = row['TotalCapacity']

            max_flows = transmission_limits.to_numpy()
            max_flows = max_flows[~np.eye(max_flows.shape[0],dtype=bool)]

            transmission_dict[str(year)] = max_flows

    if return_transmission:
        return fleet_ds,transmission_dict
    else:
        return fleet_ds
    
def rename_plant_types(fleet_df):
    
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Solar')]['PlantType'].index,'PlantType'] = 'Solar'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Onshore')]['PlantType'].index,'PlantType'] = 'Wind'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Combined Cycle CCS')]['PlantType'].index,'PlantType'] = 'CCCCS'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Coal Steam CCS')]['PlantType'].index,'PlantType'] = 'CTCCS'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Combined')]['PlantType'].index,'PlantType'] = 'CC'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Combust')]['PlantType'].index,'PlantType'] = 'CT'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Coal')]['PlantType'].index,'PlantType'] = 'ST'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Hydro')]['PlantType'].index,'PlantType'] = 'HD'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Nuclear')]['PlantType'].index,'PlantType'] = 'NU'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Pumped Storage')]['PlantType'].index,'PlantType'] = 'Storage'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Energy Storage')]['PlantType'].index,'PlantType'] = 'Storage'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Battery')]['PlantType'].index,'PlantType'] = 'Storage'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Batteries')]['PlantType'].index,'PlantType'] = 'Storage'
    fleet_df.loc[fleet_df[fleet_df['PlantType'].str.match('Cell')]['PlantType'].index,'PlantType'] = 'Storage'
    
    return fleet_df
    
def get_subfleets(fleet_df:pd.DataFrame,types_reqd=None):
    
    if types_reqd is None:       
        fleet_df_RE = fleet_df.loc[fleet_df['PlantType'].isin(['Solar','Wind'])]
        fleet_df_derate = fleet_df.loc[fleet_df['PlantType'].isin(['ST','CT','CC','CCCCS'])]
        fleet_df_nonRE_nonderate = fleet_df.loc[fleet_df['PlantType'].isin(['NU','Other'])]
        fleet_df_non_RE_non_HD = pd.concat([fleet_df_derate,fleet_df_nonRE_nonderate])
        fleet_df_HD = fleet_df.loc[fleet_df['PlantType'].isin(['HD'])]
        
        return fleet_df_RE,fleet_df_derate,fleet_df_nonRE_nonderate,\
               fleet_df_non_RE_non_HD,fleet_df_HD


##############################################################################################################
## Get plant generation

### Map closest lat/lon

def get_field_timeseries_latlon(dset:xr.Dataset, var_name:str, 
                                lat:float, lon:float,
                                lat_name:str="lat", lon_name:str="lon"
                               ) -> xr.DataArray:
    """
    Find the time series of a required field closest to a given lat,lon point.
    Required for edge cases where the dataset has been masked by a shapefile, but 
    xarray retains the rectilinear grid, so the given point might be close to a gridpoint
    that has NaN value. 

    Args:
        dset (xr.Dataset): Dataset containing required field. (lat,lon,time) dims
        var_name (str): Required field name
        lat (float): Required lat
        lon (float): Required lon
        lat_name (str, optional): Dataset might have latitude or lat or other name. 
                                  Defaults to "lat".
        lon_name (str, optional): Dataset might have longitude or lon or other name. 
                                  Defaults to "lon".
    Returns:
        xr.DataArray: Time series of required field
    """
   
    latCenter,lonCenter = dset[lat_name][int(dset.dims[lat_name]/2)].values,dset[lon_name][int(dset.dims[lon_name]/2)].values
    latStep,lonStep = abs(dset[lat_name][0].values-dset[lat_name][1].values),abs(dset[lon_name][0].values-dset[lon_name][1].values)

    while np.isnan(dset.isel(time=-1).sel({lat_name:lat,lon_name:lon},method='nearest').compute()[var_name]):
        lat = lat + ((latCenter-lat)/((latCenter-lat)**2+(lonCenter-lon)**2)**0.5)*latStep
        lon = lon + ((lonCenter-lon)/((latCenter-lat)**2+(lonCenter-lon)**2)**0.5)*lonStep

    reqd_field_ts = dset[var_name].sel({lat_name:lat,lon_name:lon},method='nearest')

    return reqd_field_ts

### Solar, wind capacity factors

def calc_generation_RE(CF_dset:xr.Dataset,caps_dframe:pd.DataFrame,var_name:str,
                       return_CF=False                      
                      ):
    """wind_CF_ens.sel(time=str(year)).sel({'lat':32.702107,
                                        'lon':360-116.354587},
                                       method='nearest')['Wind_CF']
                                       
    """
  
    arr = np.array([get_field_timeseries_latlon(CF_dset,var_name, 
                                                row['lat'],
                                                row['lon']
                                               ).values for index,row in caps_dframe.iterrows()])
    
    if len(arr) == 0:
        print('here')
        return pd.Series(np.zeros(len(CF_dset.time.values)),
                            index=CF_dset.time.values
                           )
                                     
    
    #CF_dframe = pd.DataFrame(arr.T,index=xr.CFTimeIndex.to_datetimeindex(CF_dset.time.values))
    CF_dframe = pd.DataFrame(arr.T,index=CF_dset.time.values)

    CF_dframe.columns = caps_dframe.index
    
    generation = (CF_dframe*caps_dframe['Capacity (MW)']).sum(axis=1)
    
    if return_CF:
        return CF_dframe
    else:
        return generation

### FOR

def calc_CF_nonRE(FOR_dset:xr.Dataset,caps_dframe:pd.DataFrame,var_name:str='',
                  return_CF=False
                 ):
    
    CF_dframe = pd.DataFrame(index=FOR_dset.time.values)
    
    for index,row in caps_dframe.iterrows():
        if np.isnan(row['lat']):
            print('here')
            CF_dframe[index] = np.repeat((1-0.05),len(CF_dframe))            
        else:
            if row['PlantType'] == 'CCCCS':
                row['PlantType'] = 'CC'
            CF_dframe[index] = 1-get_field_timeseries_latlon(FOR_dset,row['PlantType'],
                                                             row['lat'],
                                                             row['lon']
                                                            ).values
    CF_dframe.columns = caps_dframe.index
        
    if return_CF:
        return CF_dframe
    else:
        generation = (CF_dframe*caps_dframe['Capacity (MW)']).sum(axis=1)
        return generation

### Hydroelectric generation

def return_hydroelectric_generation(hydro_plants,monthly_limits_total,
                                    netload_wo_hydro,netload_RE,
                                    FORs,hydro=False
                                   ):

    """
    Everything for a subregion, so this code will work whether we have all regions
    or individual regions.
    hydro_plants: pd.Dataframe with individual plant capacities
    monthly_limits_total: Total monthly generation
    netload_wo_hydro: demand - generation from CC,RE,NU,CT etc etc
    netload_RE: demand - (solar+wind)
    """
    
    monthly_limits_dailyavg = monthly_limits_total/24

    days_in_month_dict = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    days_till_month = np.append([0],[(days_in_month_dict).cumsum()])

    region_hydro_nameplate = hydro_plants['Capacity (MW)'].sum() 

    hydro_plants = hydro_plants.set_index('GAMS Symbol') 
    hydro_multip = hydro_plants['Capacity (MW)']
    hydro_multip = hydro_multip/hydro_multip.sum()
    
    hydro_dispatch = np.zeros((365))
    
    netload_deficit = np.where(netload_wo_hydro>0,netload_wo_hydro,0)

    ### NEED TO COMEBACK AND VERIFY THIS WHEN LOTS OF RE
    netload_RE = np.where(netload_RE>0,netload_RE,0)

    if hydro=='annual':
        
        total_hydro_capacity = monthly_limits_dailyavg.sum()
        for day in range(days_in_month_dict.sum()):
            
            if total_hydro_capacity < netload_deficit[day]:
                print(day)
                hydro_dispatch[day] = total_hydro_capacity if total_hydro_capacity < region_hydro_nameplate \
                                    else region_hydro_nameplate
                break

            hydro_dispatch[day] = netload_deficit[day] if netload_deficit[day]< region_hydro_nameplate \
                                    else region_hydro_nameplate 
            
            total_hydro_capacity -= hydro_dispatch[day]

        if total_hydro_capacity >= 0:

            # smear the remaining hydro in the month proportional 
            # to the netload curve (demand - generation from RE)
            normalize_netload_RE = netload_RE
            normalize_netload_RE = normalize_netload_RE/normalize_netload_RE.sum()

            spread = total_hydro_capacity*normalize_netload_RE

            dispatch = np.where(spread+hydro_dispatch\
                                        <region_hydro_nameplate,
                                spread+hydro_dispatch,
                                region_hydro_nameplate)
            hydro_dispatch = dispatch
            
    elif hydro == 'monthly':
        for month in range(12):
            monthly_total_hydro_capacity = monthly_limits_dailyavg[month]

            for day in range(days_till_month[month],days_till_month[month+1]):

                if monthly_total_hydro_capacity < netload_deficit[day]:
                    print(day)
                    hydro_dispatch[day] = monthly_total_hydro_capacity if monthly_total_hydro_capacity < region_hydro_nameplate \
                                        else region_hydro_nameplate
                    break

                hydro_dispatch[day] = netload_deficit[day] if netload_deficit[day]< region_hydro_nameplate \
                                        else region_hydro_nameplate 

                monthly_total_hydro_capacity -= hydro_dispatch[day]

            if monthly_total_hydro_capacity >= 0:

                # smear the remaining hydro in the month proportional 
                # to the netload curve (demand - generation from RE)
                normalize_netload_RE = netload_RE[days_till_month[month]:days_till_month[month+1]]
                normalize_netload_RE = normalize_netload_RE/normalize_netload_RE.sum()

                spread = monthly_total_hydro_capacity*normalize_netload_RE

                dispatch = np.where(spread+hydro_dispatch[days_till_month[month]:days_till_month[month+1]]\
                                            <region_hydro_nameplate,
                                    spread+hydro_dispatch[days_till_month[month]:days_till_month[month+1]],
                                    region_hydro_nameplate)
                hydro_dispatch[days_till_month[month]:days_till_month[month+1]] = dispatch

    hydroelectric_generation_plants = pd.DataFrame(hydro_multip.to_numpy()*hydro_dispatch.reshape((-1,1)),
                                         columns=hydro_plants.index,                
                                         index=netload_wo_hydro.index)
    
    HD_CFs = calc_CF_nonRE(FORs,
                           hydro_plants,return_CF=True)

    # For plants which have a cooling type, multiply derate with forced outage
    for col in HD_CFs:
        hydroelectric_generation_plants[col] = hydroelectric_generation_plants[col]*HD_CFs[col]

    
    hydroelectric_generation = hydroelectric_generation_plants.sum(axis=1)
            
    return hydroelectric_generation

##############################################################################################################
## SAC

def get_region_SAC(fleet_year_region,compressed_units,
                   solar_cf_year_ens,wind_cf_year_ens,
                   FOR_ds_year_ens,derate_ds_year_ens,
                   monthly_limits_total_region,
                   demand_region_year,hydro=None,
                   return_HD=False
                  ):
    
    year = str(solar_cf_year_ens.time.dt.year.values[0])
    
    fleet_year_region_RE = fleet_year_region.loc[fleet_year_region['PlantType'].isin(['Solar','Wind'])]
    fleet_year_region_derate = fleet_year_region.loc[fleet_year_region['PlantType'].isin(['ST','CT','CC','CCCCS','CTCCS'])]
    fleet_year_region_nonRE_nonderate = fleet_year_region.loc[fleet_year_region['PlantType'].isin(['NU','Other'])]
    
    fleet_year_region_non_RE_non_HD = pd.concat([fleet_year_region_derate,fleet_year_region_nonRE_nonderate])
    
    # Get solar and wind generation in region, year
    solar_df = calc_generation_RE(solar_cf_year_ens,
                                  fleet_year_region_RE.loc[fleet_year_region_RE['PlantType'].isin(['Solar'])],
                                  'Solar_CF')

    wind_df = calc_generation_RE(wind_cf_year_ens,
                                 fleet_year_region_RE.loc[fleet_year_region_RE['PlantType'].isin(['Wind'])],
                                 'Wind_CF')
    RE_generation = solar_df+wind_df

    # Get CFs (1-FOR) for all plants other than Solar, Wind, HD
    nonRE_CFs = calc_CF_nonRE(FOR_ds_year_ens,
                              fleet_year_region_non_RE_non_HD,
                              return_CF=True,
                             )

    # Get derates for plants which have a cooling type
    if len(fleet_year_region_derate):
        capDerates = calc_derates.calculatePlantCapacityDerates(fleet_year_region_derate,[year],
                                                                derate_ds_year_ens,compressed_units)

        # For plants which have a cooling type, multiply derate with forced outage
        for col in capDerates:
            nonRE_CFs[col] = nonRE_CFs[col]*(1-capDerates[col])

    # Get generation for all plants other than Solar, Wind, HD
    non_RE_non_HD_generation = (nonRE_CFs*fleet_year_region_non_RE_non_HD['Capacity (MW)']).sum(axis=1)

    if hydro is None:
        
        #total_nonHD_generation = RE_generation+non_RE_non_HD_generation
        total_nonHD_generation = RE_generation
        
        region_SAC = total_nonHD_generation-demand_region_year

    else:
        fleet_year_region_HD = fleet_year_region.loc[fleet_year_region['PlantType'].isin(['HD'])]

        HD_generation = return_hydroelectric_generation(fleet_year_region_HD,
                                                        monthly_limits_total_region,
                                                        demand_region_year-(RE_generation+non_RE_non_HD_generation),
                                                        demand_region_year-(RE_generation),
                                                        FOR_ds_year_ens,
                                                        hydro=hydro
                                                       )

        total_generation = RE_generation+non_RE_non_HD_generation+HD_generation

        region_SAC = total_generation-demand_region_year
    
    if return_HD:
        return region_SAC,non_RE_non_HD_generation,demand_region_year-(RE_generation+non_RE_non_HD_generation),demand_region_year-(RE_generation),solar_df,wind_df 
    else:
        return region_SAC
    
##############################################################################################################
## Transmission

def get_SAC_delta(flows,flow_indices,num_regions):
    
    delta = np.zeros(num_regions)
    
    for idx,elem in enumerate(flow_indices):
        delta[elem[0]] -= flows[idx]         
        delta[elem[1]] += flows[idx]
        
    return delta

def transmission_after_regional_SAC(SAC_before,transmission_flow,
                                    all_regions,
                                    flow_cost=5,ens_cost=100,                                 
                                   ):
    
    num_regions = len(all_regions)
        
    model = tr.init_model(transmission_flow,flow_cost,ens_cost)

    deficits = SAC_before[:,np.any(SAC_before<0,axis=0)]
    deficits = deficits.T
    residual_samples = np.array(deficits)

    deficit_locs = np.unique(np.argwhere(np.any(SAC_before<0,axis=0)),axis=1)
    #return deficit_locs

    output = [tr.create_instance_getoutput(model,row) for row in residual_samples]

    SAC_after = SAC_before.copy()

    flow_indices = np.array([(reg2,reg1) if reg1 != reg2 else 0 for reg2 in range(num_regions) for reg1 in range(num_regions)])
    flow_indices = np.delete(flow_indices, np.where(flow_indices == 0))
    
    SAC_deltas = np.array([get_SAC_delta(row[0],flow_indices,len(all_regions)) for row in output])

    SAC_after[:,deficit_locs[:,0]] = SAC_after[:,deficit_locs[:,0]] \
                                                       + SAC_deltas.T
    
    return SAC_after

def transmission_with_hydro(SAC_before,transmission_flow,all_regions,
                            hydro_plants,monthly_limits_allregions,
                            FORs,
                            flow_cost=5,ens_cost=100,                              
                           ):

    """
    hydro_plants: pd.Dataframe with individual plant capacities
    monthly_limits_total: Total monthly generation
    """
    
    monthly_limits_dailyavg = monthly_limits_allregions/24
    time_steps = (1, 365)
    
    days_in_month_dict = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    days_till_month = np.append([0],[(days_in_month_dict).cumsum()])
    daylims_month = trhy.get_dicts_from_numpy(np.vstack([(days_till_month[:-1] + 1),days_till_month[1:]]).T,)

    hydro_nameplate_allregions = hydro_plants.groupby('region').sum()['Capacity (MW)']
    hydro_daily_CF_allregions = pd.DataFrame(index=FORs.time.values)

    for region in all_regions:
        region_hydro = hydro_plants.loc[hydro_plants['region'].isin([region])]

        region_hydro = region_hydro.set_index('GAMS Symbol') 
        
        hydro_multip = region_hydro['Capacity (MW)']
        hydro_multip = hydro_multip/hydro_multip.sum()
        
        HD_CFs = calc_CF_nonRE(FORs,region_hydro,return_CF=True)

        # For plants which have a cooling type, multiply derate with forced outage
        hydro_daily_CF_allregions[region] = (hydro_multip*HD_CFs).sum(axis=1)
    
    num_regions = len(all_regions)
        
    hydro_params = {'hydro_capacitycap':trhy.get_dicts_from_numpy(hydro_nameplate_allregions),
                    'hydro_gen_cap':trhy.get_dicts_from_dataframe(monthly_limits_dailyavg)[1],
                    'hydro_CF':trhy.get_dicts_from_dataframe(hydro_daily_CF_allregions)[1]
                   }
    
    initial_residuals = trhy.get_dicts_from_numpy(SAC_before)
    
    model = trhy.init_model(transmission_flow,time_steps,daylims_month,
                            initial_residuals,
                            hydro_params,
                            flow_cost=flow_cost,ens_cost=ens_cost)
    
    trhy.set_model_hydro_constraints(model,daylims_month)
    trhy.nodal_residual_constraint(model)
    trhy.flow_constraint_rule(model)

    model.obj = pyo.Objective(rule=trhy.o_rule, sense=pyo.minimize)
    
    result = pyo.SolverFactory('glpk').solve(model)
    
    flow_indices = np.array([(reg2,reg1) if reg1 != reg2 else 0 for reg2 in range(num_regions) 
                                                                for reg1 in range(num_regions)])    
    flow_indices = np.delete(flow_indices, np.where(flow_indices == 0))

    
    if (result.solver.status == SolverStatus.ok) and \
       (TerminationCondition.optimal == result.solver.termination_condition):

        flow = np.array([[model.flows[j,t]() for t in model.time]
                         for j in model.edges]).T
        
        hydro = np.array([[model.hydro_generation[i,t]() for t in model.time]
                         for i in model.nodes]).T
                
        ens = np.array([[model.unserved_energy[i,t]() for t in model.time]
                         for i in model.nodes]).T
        
        SAC_after = SAC_before.copy()

        SAC_deltas = np.array([get_SAC_delta(row,flow_indices,len(all_regions)) for row in flow])
        SAC_after = SAC_after + SAC_deltas.T + hydro.T

    return SAC_after,hydro.T,ens.T


def run_economicdispatch(all_regions,fleet_year,
                         compressed_units,demand_year,
                         solar_cf_year_ens,wind_cf_year_ens,
                         FOR_ds_year_ens,derate_ds_year_ens,
                         monthly_limits_allregions,
                         transmission_flow,
                         flow_cost,ens_cost,
                         co2_cost=None,co2_cap=None,
                         return_everything=False,
                         return_generations_damand_only=False
                        ):
    
    year = str(solar_cf_year_ens.time.dt.year.values[0])
    
    RE_generation_allregions = pd.DataFrame(index=demand_year.index)
    solar_allregions = pd.DataFrame(index=demand_year.index)
    wind_allregions = pd.DataFrame(index=demand_year.index)
    
    hydro_daily_CF_allregions = pd.DataFrame(index=demand_year.index)
    
    nonRE_nonHD_generation_allregions = pd.DataFrame(index=demand_year.index)
    nonRE_nonHD_generation_regionaltotals = pd.DataFrame(index=demand_year.index)
    
    non_RE_non_HD_cost_all_regions = np.empty(0)
    non_RE_non_HD_co2emis_all_regions = np.empty(0)
    non_RE_non_HD_index_all_regions = np.empty(0)
    
    nonRE_nonHD_plant_indexrange = np.empty((0,2))
    count = 1

    for region in all_regions:
    
        fleet_year_region = fleet_year.loc[fleet_year['region'].isin([region])]
    
        ###########
        # RE generation
        fleet_year_region_RE = fleet_year_region.loc[fleet_year_region['PlantType'].isin(['Solar','Wind'])]
        fleet_year_region_RE = fleet_year_region_RE[fleet_year_region_RE['Retired'].isin([False])]
        
        # Get solar and wind generation in region, year
        solar_df = calc_generation_RE(solar_cf_year_ens,
                                      fleet_year_region_RE.loc[fleet_year_region_RE['PlantType'].isin(['Solar'])],
                                      'Solar_CF')

        wind_df = calc_generation_RE(wind_cf_year_ens,
                                     fleet_year_region_RE.loc[fleet_year_region_RE['PlantType'].isin(['Wind'])],
                                     'Wind_CF')
        
        
        RE_generation_allregions[region] = solar_df+wind_df
        solar_allregions[region] = solar_df
        wind_allregions[region] = wind_df
        
        if return_generations_damand_only:
            continue
                
        ###########
        # nonRE_nonHD generation        
        fleet_year_region_derate = fleet_year_region.loc[fleet_year_region['PlantType'].isin(['ST','CT','CC','CCCCS'])]
        fleet_year_region_nonRE_nonderate = fleet_year_region.loc[fleet_year_region['PlantType'].isin(['NU','Other'])]

        fleet_year_region_non_RE_non_HD = pd.concat([fleet_year_region_derate,fleet_year_region_nonRE_nonderate])
        fleet_year_region_non_RE_non_HD = fleet_year_region_non_RE_non_HD[fleet_year_region_non_RE_non_HD['Retired'].isin([False])]
        
        # Get CFs (1-FOR) for all plants other than Solar, Wind, HD
        nonRE_CFs = calc_CF_nonRE(FOR_ds_year_ens,
                                  fleet_year_region_non_RE_non_HD,
                                  return_CF=True,
                                 )
        
        #nonRE_CFs = pd.DataFrame(np.ones(nonRE_CFs.shape),
        #                         columns=nonRE_CFs.columns,
        #                         index=nonRE_CFs.index)
        
        
        # Get derates for plants which have a cooling type
        if len(fleet_year_region_derate):
            capDerates = calc_derates.calculatePlantCapacityDerates(fleet_year_region_derate,[year],
                                                                    derate_ds_year_ens,compressed_units)
            
            #print(region,'derated')
            #capDerates.plot(legend=False)
            #plt.show()
            

            # For plants which have a cooling type, multiply derate with forced outage
            for col in np.intersect1d(nonRE_CFs.columns,capDerates.columns):
                nonRE_CFs[col] = nonRE_CFs[col]*(1-capDerates[col])

        # Get generation for all plants other than Solar, Wind, HD
        #nonRE_nonHD_generation_allregions[region] = (nonRE_CFs\
        #                                             *fleet_year_region_non_RE_non_HD['Capacity (MW)']
        #                                            ).sum(axis=1)
        
        nonRE_nonHD_generation_region = (nonRE_CFs\
                                                     *fleet_year_region_non_RE_non_HD['Capacity (MW)']
                                                    )
        nonRE_nonHD_generation_regionaltotals[region] = nonRE_nonHD_generation_region.sum(axis=1)
        
        nonRE_nonHD_generation_allregions = pd.concat([nonRE_nonHD_generation_allregions,nonRE_nonHD_generation_region],
                                              axis=1)
        
        count_region = len(nonRE_nonHD_generation_region.columns)
        nonRE_nonHD_plant_indexrange = np.vstack((nonRE_nonHD_plant_indexrange,np.array([count,count+count_region-1])))
        count = count+count_region

        
        # Cost of nonREnonHD generators should also include a carbon cost so that the CCS generators
        # are incentivized to participate. If not, the nat gas generators produce more
        # CO2 cost parameter here is $/ton
        
        if co2_cost is not None:
            non_RE_non_HD_cost_region = np.array([fleet_year_region_non_RE_non_HD.loc[col]['OpCost($/MWh)'] \
                                                  + (fleet_year_region_non_RE_non_HD.loc[col]['CO2EmRate(lb/MMBtu)']\
                                                     *fleet_year_region_non_RE_non_HD.loc[col]['Heat Rate (Btu/kWh)']/1000\
                                                     *(co2_cost/2000)) #convert $/US ton to $/lb
                                                  for col in nonRE_nonHD_generation_region.columns])
        else:
            non_RE_non_HD_cost_region = np.array([fleet_year_region_non_RE_non_HD.loc[col]['OpCost($/MWh)'] \
                                                  for col in nonRE_nonHD_generation_region.columns])                                      
            
        non_RE_non_HD_co2emis_region = np.array([(fleet_year_region_non_RE_non_HD.loc[col]['CO2EmRate(lb/MMBtu)']\
                                                 *fleet_year_region_non_RE_non_HD.loc[col]['Heat Rate (Btu/kWh)']/1000)\
                                              for col in nonRE_nonHD_generation_region.columns])

        non_RE_non_HD_co2emis_all_regions = np.append(non_RE_non_HD_co2emis_all_regions,non_RE_non_HD_co2emis_region)        

        non_RE_non_HD_cost_all_regions = np.append(non_RE_non_HD_cost_all_regions,non_RE_non_HD_cost_region)
        non_RE_non_HD_index_all_regions = np.append(non_RE_non_HD_index_all_regions,nonRE_nonHD_generation_region.columns)

        
        ###########
        # HD generation
        fleet_year_region_HD = fleet_year_region.loc[fleet_year_region['PlantType'].isin(['HD'])]
        fleet_year_region_HD = fleet_year_region_HD[fleet_year_region_HD['Retired'].isin([False])]

        fleet_year_region_HD = fleet_year_region_HD.loc[fleet_year_region_HD['region'].isin([region])]
        
        hydro_multip = fleet_year_region_HD['Capacity (MW)']
        hydro_multip = hydro_multip/hydro_multip.sum()
        
        HD_CFs = calc_CF_nonRE(FOR_ds_year_ens,fleet_year_region_HD,return_CF=True)

        hydro_daily_CF_allregions[region] = (hydro_multip*HD_CFs).sum(axis=1)
        
    if return_generations_damand_only:
        
        return solar_allregions,wind_allregions
    #######################
    # Init ED setup
    
    time_steps = (1, len(demand_year))
    
    days_in_month_dict = np.tile([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],len(demand_year)//365)
    days_till_month = np.append([0],[(days_in_month_dict).cumsum()])
    daylims_month = ed.get_dicts_from_numpy(np.vstack([(days_till_month[:-1] + 1),days_till_month[1:]]).T,)
    nonRE_nonHD_plant_indexrange_dict = ed.get_dicts_from_numpy(nonRE_nonHD_plant_indexrange)
    
    # Demand
    demand = ed.get_dicts_from_dataframe(demand_year)[1]

    # RE
    RE_params = {'re_generation':ed.get_dicts_from_dataframe(RE_generation_allregions)[1]}

    # Non RE non HD
    nonRE_nonHD_params = {'num_nonre_nonHD_plants':len(nonRE_nonHD_generation_allregions.columns),
                          'nonre_nonHD_costs':ed.get_dicts_from_numpy(non_RE_non_HD_cost_all_regions),
                          'nonre_nonHD_generation':ed.get_dicts_from_dataframe(nonRE_nonHD_generation_allregions)[1]}
    
    if co2_cap is not None:
        nonRE_nonHD_params['nonre_nonHD_co2emis'] = ed.get_dicts_from_numpy(non_RE_non_HD_co2emis_all_regions)
       
    # HD
    monthly_limits_dailyavg = monthly_limits_allregions/24
    
    hydro_plants = fleet_year.loc[fleet_year['PlantType'].isin(['HD'])]
    hydro_nameplate_allregions = hydro_plants.groupby('region').sum()['Capacity (MW)']

    hydro_params = {'hydro_capacitycap':ed.get_dicts_from_numpy(hydro_nameplate_allregions),
                    'hydro_gen_cap':ed.get_dicts_from_dataframe(monthly_limits_dailyavg)[1],
                    'hydro_CF':ed.get_dicts_from_dataframe(hydro_daily_CF_allregions)[1]
                   }
    
    cost_params = {'flow_cost':flow_cost,
                   'ens_cost':ens_cost,
                  }
    
    model = ed.init_model(transmission_flow,time_steps,daylims_month,
                          demand,
                          RE_params,nonRE_nonHD_params,
                          hydro_params,
                          cost_params,
                          co2_cap
                         )
    
    
    #######################
    # Set constratints
    ed.set_RE_constraints(model)
    ed.set_nonRE_nonHD_constraints(model,co2_cap)
    ed.set_hydro_constraints(model,daylims_month)
    ed.nodal_balance_constraint(model,nonRE_nonHD_plant_indexrange_dict)
    ed.flow_constraint_rule(model)

    model.obj = pyo.Objective(rule=ed.o_rule, sense=pyo.minimize)


    #######################
    # Solve and get results
    result = pyo.SolverFactory('glpk').solve(model)
    
    
    if (result.solver.status == SolverStatus.ok) and \
       (TerminationCondition.optimal == result.solver.termination_condition):

        flow = np.array([[sum(model.nodal_constraint_matrix[i,j]*model.flows[j,t]() for j in model.edges)
                          for t in model.time]
                          for i in model.nodes
                         ]).T
        
        flow_df = pd.DataFrame(flow,
                            index=nonRE_nonHD_generation_regionaltotals.index,
                            columns=nonRE_nonHD_generation_regionaltotals.columns)
    
        
        hydro_df = pd.DataFrame(np.array([[model.hydro_generation[i,t]() for i in model.nodes]
                                                  for t in model.time]
                                                ),
                                        index=nonRE_nonHD_generation_regionaltotals.index,
                                        columns=nonRE_nonHD_generation_regionaltotals.columns)
        
        RE_generation_df = pd.DataFrame(np.array([[model.RE_generation[i,t]() for i in model.nodes]
                                                  for t in model.time]
                                                ),
                                        index=nonRE_nonHD_generation_regionaltotals.index,
                                        columns=nonRE_nonHD_generation_regionaltotals.columns)
                                
        
        nonRE_nonHD_df = pd.DataFrame(np.array([[sum(model.nonRE_nonHD_generation[p,t]() 
                                                  for p in pyo.RangeSet(nonRE_nonHD_plant_indexrange_dict[(i, 1)],
                                                                        nonRE_nonHD_plant_indexrange_dict[(i, 2)]))
                                              for i in model.nodes]
                                             for t in model.time]
                                           ),
                                        index=nonRE_nonHD_generation_regionaltotals.index,
                                        columns=nonRE_nonHD_generation_regionaltotals.columns)
        
        annual_nonRE_generation_df = pd.Series(np.hstack([[sum(model.nonRE_nonHD_generation[p,t]() for t in model.time)
                                                            for p in pyo.RangeSet(nonRE_nonHD_plant_indexrange_dict[(i, 1)],
                                                                                  nonRE_nonHD_plant_indexrange_dict[(i, 2)])]
                                                            for i in model.nodes]
                                                          ),
                                        index=non_RE_non_HD_index_all_regions
                                        )
                
        ens_df = pd.DataFrame(np.array([[model.unserved_energy[i,t]() for i in model.nodes]
                                     for t in model.time]
                                   ),
                           index=nonRE_nonHD_generation_regionaltotals.index,
                           columns=nonRE_nonHD_generation_regionaltotals.columns)
        
        SAC_df = ((nonRE_nonHD_generation_regionaltotals - nonRE_nonHD_df)
                  +(RE_generation_allregions - RE_generation_df)
                  - ens_df)

    
    else:
        print("didn't solve")
        
    if return_everything:
        return SAC_df,ens_df,annual_nonRE_generation_df,hydro_df,flow_df,nonRE_nonHD_generation_regionaltotals,RE_generation_allregions,nonRE_CFs,nonRE_nonHD_generation_allregions
    #SAC_df,ens_df,flow.T,hydro.T,RE_generation_df,nonRE_nonHD_df,nonRE_nonHD_generation_regionaltotals,RE_generation_allregions,solar_allregions,wind_allregions,model,annual_nonRE_generation_df
    else:
        return SAC_df,ens_df,annual_nonRE_generation_df,hydro_df,flow_df,solar_allregions,wind_allregions


    
