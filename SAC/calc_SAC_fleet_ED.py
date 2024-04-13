import datetime
import xarray as xr
import pandas as pd
import numpy as np

import cartopy.feature as cfeature
import cartopy.crs as ccrs

import os, itertools, warnings, sys
warnings.filterwarnings("ignore")

import SAC as SAC

base_dir = '/glade/work/sriharis/MacroCEMResults/forpaper/'
out_dir = '/glade/work/sriharis/paper2_SAC_data/09102023_CO2cap/'

pathway = sys.argv[1]
invest_ens = sys.argv[2]

if len(sys.argv) >= 4:
    year = sys.argv[3]
    print(year)
    if int(year)%2:
        fleet_year = str(int(year)+1)
    else:
        fleet_year = year

if len(sys.argv) == 5:
    partition_reqd = int(sys.argv[4])
    
fleet_name = 'WECCC'+pathway+'EM'+invest_ens

all_gens = ['NU','HD','ST','CT','CC','CCCCS','Solar','Wind','Storage','Other']
all_regions = ['CAMX', 'Desert_Southwest', 'NWPP_Central', 'NWPP_NE', 'NWPP_NW']

ens_info = pd.read_csv('ensinfo.csv')

compressed_units = pd.read_csv(base_dir+fleet_name+'/compressedUnitsFromGenFleet.csv')
compressed_units = SAC.rename_plant_types(compressed_units)
compressed_units = compressed_units.rename(columns={"Latitude": "lat", "Longitude": "lon"})

fleet_ds,transmission_dict = SAC.get_reqdfleet_info(base_dir,fleet_name,all_regions)

met_inputs_source_dir = '/glade/work/sriharis/RDM_dsets/'

demand_ds = xr.open_dataset(met_inputs_source_dir+'demand_allmembers.nc')
solar_ds = xr.open_dataset(met_inputs_source_dir+'cesm_WECC_solar_CF.nc')
wind_ds = xr.open_dataset(met_inputs_source_dir+'cesm_WECC_wind_CF_noBC.nc')
FOR_ds = xr.open_dataset(met_inputs_source_dir+'cesm_WECC_FOR.nc')
derate_ds = xr.open_dataset(met_inputs_source_dir+'cesm_derate_fields.nc')
hydro_ds = xr.open_dataset(met_inputs_source_dir+'monthly_hydro_generation_plants.nc')

fleet_year_ds = fleet_ds.sel(time=fleet_year)
fleet_year_ds = fleet_year_ds.where(fleet_year_ds['Capacity (MW)'].notnull(),drop=True)
fleet_year_df = fleet_year_ds.to_dataframe().reset_index().set_index('GAMS Symbol')

#co2_cap = pd.read_csv(base_dir+'WECCC'+pathway+'EM'+invest_ens
#                      +'/'+fleet_year+'/CE/co2CapCE'\
#                      +fleet_year+'.csv',index_col=0).values.sum()*2000/2

co2_cap = 1e30

co2_cost = pd.read_csv(base_dir+'co2Prices.csv',index_col=0
                      ).loc['WECCC'+pathway+'EM'+invest_ens][fleet_year]

def get_SAC_reliability_ens(reliability_ens,year):
    print(reliability_ens)
    
    solar_cf_year_ens = solar_ds.sel(time=year).sel(member_id=reliability_ens)    
    wind_cf_year_ens = wind_ds.sel(time=year).sel(member_id=reliability_ens)
    FOR_ds_year_ens = FOR_ds.sel(time=year).sel(member_id=reliability_ens)
    derate_ds_year_ens = derate_ds.sel(time=year).sel(member_id=reliability_ens)
    hydro_df = hydro_ds.sel(time=year).sel(member_id=reliability_ens)\
                .groupby('region').sum().to_dataframe().reset_index(level=0)
    monthly_limits_total = hydro_df.pivot_table(values='mon_hydro', index=hydro_df.index, columns='region')

    demand_year = demand_ds.sel(time=year)\
                                  .sel(member_id=reliability_ens)\
                                  .to_dataframe().reset_index(level=1)
    demand_year = demand_year.pivot_table(values='demand',index=demand_year.index, 
                                                        columns='region')

    #flow,hydro,ens,RE_generation,nonRE_nonHD,nonRE_nonHD_generation_allregions,RE_generation_allregions,solar_allregions,wind_allregions,model = SAC.run_economicdispatch(all_regions,fleet_year_df,
    SAC_df,ens_df,annual_nonRE_generation_df,hydro_df,flow_df,solar_year,wind_year = SAC.run_economicdispatch(all_regions,fleet_year_df,
                                compressed_units,demand_year,
                             solar_cf_year_ens,wind_cf_year_ens,
                             FOR_ds_year_ens,derate_ds_year_ens,
                             monthly_limits_total,
                             transmission_dict[fleet_year],
                             flow_cost=0.01,ens_cost=1000,
                             co2_cap=co2_cap,co2_cost=co2_cost)

    SAC_df.index.name = 'time'
    SAC_df['reliability_ens'] = np.repeat(reliability_ens,len(SAC_df.index))
    SAC_df = SAC_df.reset_index()
    SAC_df.set_index(['time','reliability_ens'],inplace=True)

    ens_df.index.name = 'time'
    ens_df['reliability_ens'] = np.repeat(reliability_ens,len(ens_df.index))
    ens_df = ens_df.reset_index()
    ens_df.set_index(['time','reliability_ens'],inplace=True)
    
    hydro_df.index.name = 'time'
    hydro_df['reliability_ens'] = np.repeat(reliability_ens,len(hydro_df.index))
    hydro_df = hydro_df.reset_index()
    hydro_df.set_index(['time','reliability_ens'],inplace=True)
    
    flow_df.index.name = 'time'
    flow_df['reliability_ens'] = np.repeat(reliability_ens,len(flow_df.index))
    flow_df = flow_df.reset_index()
    flow_df.set_index(['time','reliability_ens'],inplace=True)
    
    solar_year.index.name = 'time'
    solar_year['reliability_ens'] = np.repeat(reliability_ens,len(solar_year.index))
    solar_year = solar_year.reset_index()
    solar_year.set_index(['time','reliability_ens'],inplace=True)

    wind_year.index.name = 'time'
    wind_year['reliability_ens'] = np.repeat(reliability_ens,len(wind_year.index))
    wind_year = wind_year.reset_index()
    wind_year.set_index(['time','reliability_ens'],inplace=True)
    
    return SAC_df.to_xarray(),ens_df.to_xarray(),\
           annual_nonRE_generation_df,hydro_df.to_xarray(),flow_df.to_xarray(),\
           solar_year.to_xarray(),wind_year.to_xarray()


##################################################################################################################
# Main

reqd_ens = ens_info['ensemble_forcing'].loc[partition_reqd*10:(partition_reqd+1)*10-1]

print(reqd_ens)
all_SAC_ens = [get_SAC_reliability_ens(reliability_ens,year) \
               for reliability_ens in reqd_ens
              ]

all_SAC_ds = xr.concat([elem[0] for elem in all_SAC_ens],dim='reliability_ens')
all_SAC_ds.attrs = {'Decarbonization pathway':pathway,
                     'Invest ensemble':invest_ens,
                     'Date':datetime.date.today().strftime("%b-%d-%Y"),
                     'Author':'Srihari Sundar, University of Michigan'
                   }

all_ens_ds = xr.concat([elem[1] for elem in all_SAC_ens],dim='reliability_ens')
all_ens_ds.attrs = {'Decarbonization pathway':pathway,
                     'Invest ensemble':invest_ens,
                     'Date':datetime.date.today().strftime("%b-%d-%Y"),
                     'Author':'Srihari Sundar, University of Michigan'
                   }

all_ens_nonREgen = pd.concat([elem[2] for elem in all_SAC_ens],axis=1)
all_ens_nonREgen.columns = reqd_ens

all_hydro_ds = xr.concat([elem[3] for elem in all_SAC_ens],dim='reliability_ens')
all_hydro_ds.attrs = {'Decarbonization pathway':pathway,
                     'Invest ensemble':invest_ens,
                     'Date':datetime.date.today().strftime("%b-%d-%Y"),
                     'Author':'Srihari Sundar, University of Michigan'
                   }

all_flow_ds = xr.concat([elem[4] for elem in all_SAC_ens],dim='reliability_ens')
all_flow_ds.attrs = {'Decarbonization pathway':pathway,
                     'Invest ensemble':invest_ens,
                     'Date':datetime.date.today().strftime("%b-%d-%Y"),
                     'Author':'Srihari Sundar, University of Michigan'
                   }

solar_ds = xr.concat([elem[5] for elem in all_SAC_ens],dim='reliability_ens')

solar_ds.attrs = {'Decarbonization pathway':pathway,
                     'Invest ensemble':invest_ens,
                     'Date':datetime.date.today().strftime("%b-%d-%Y"),
                     'Author':'Srihari Sundar, University of Michigan'
                   }

wind_ds = xr.concat([elem[6] for elem in all_SAC_ens],dim='reliability_ens')
wind_ds.attrs = {'Decarbonization pathway':pathway,
                     'Invest ensemble':invest_ens,
                     'Date':datetime.date.today().strftime("%b-%d-%Y"),
                     'Author':'Srihari Sundar, University of Michigan'
                   }

fleet_year_df_nonRE = fleet_year_df.loc[fleet_year_df['PlantType'].isin(['ST','CT','CC','CCCCS','NU','Other'])]
fleet_year_df_nonRE = fleet_year_df_nonRE.loc[fleet_year_df_nonRE['Retired'].isin([False])]

CO2_all = (all_ens_nonREgen.T.multiply(fleet_year_df_nonRE['CO2EmRate(lb/MMBtu)']\
                                       *fleet_year_df_nonRE['Heat Rate (Btu/kWh)']/1000*24)).sum(axis=1)

all_SAC_ds.to_netcdf(out_dir+'SAC_pway'+pathway+'_invens'+invest_ens+'_year'+year+'_'+str(partition_reqd)+'.nc')

all_ens_ds.to_netcdf(out_dir+'ENS_pway'+pathway+'_invens'+invest_ens+'_year'+year+'_'+str(partition_reqd)+'.nc')

all_hydro_ds.to_netcdf(out_dir+'hydro_gen_pway'+pathway+'_invens'+invest_ens+'_year'+year+'_'+str(partition_reqd)+'.nc')

all_flow_ds.to_netcdf(out_dir+'flow_gen_pway'+pathway+'_invens'+invest_ens+'_year'+year+'_'+str(partition_reqd)+'.nc')

all_ens_nonREgen.to_csv(out_dir+'nonRE_gen_pway'+pathway+'_invens'+invest_ens+'_year'+year+'_'+str(partition_reqd)+'.csv')

solar_ds.to_netcdf(out_dir+'solar_gen_pway'+pathway+'_invens'+invest_ens+'_year'+year+'_'+str(partition_reqd)+'.nc')
wind_ds.to_netcdf(out_dir+'wind_gen_pway'+pathway+'_invens'+invest_ens+'_year'+year+'_'+str(partition_reqd)+'.nc')

CO2_all.to_csv(out_dir+'CO2_emis_pway'+pathway+'_invens'+invest_ens+'_year'+year+'_'+str(partition_reqd)+'.csv')
