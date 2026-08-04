#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 09:14:14 2025

@author: andreaskarpasitis
"""
import numpy as np
import xarray as xr
import os
import copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

warnings.filterwarnings("ignore")

def convert_lon_to_minus180_180(da):
    """
    Convert longitudes from the range [0, 360] to [-180, 180].

    Parameters:
    da (xr.DataArray or xr.Dataset): Input data with longitude dimension.

    Returns:
    xr.DataArray or xr.Dataset: Data with longitudes converted to [-180, 180].
    """
    if da['lon'].max() > 180:
        #print('Chaning coordinates from [0,360] to [-180,180]')
        da = da.assign_coords(lon=(((da['lon'] + 180) % 360) - 180))
        da = da.sortby('lon')
    return da

def filter_files_by_year_range(filenames, start_year, end_year):
  """Filters NetCDF files based on the year range specified in the filename.

  Args:
    filenames: The path to the directory containing the NetCDF files.
    start_year: The starting year of the desired range.
    end_year: The ending year of the desired range.

  Returns:
    A list of filenames that fall within the specified year range.
  """

  filtered_files = []
  
  for file in filenames:
    range_years = file.split("_")[-1]
    first_year = int(range_years.split("-")[0][:4])  # Extract the first year from the filename
    last_year = int(range_years.split("-")[-1][:4])  # Extract the first year from the filename
    #print(first_year, last_year)

    if first_year <= start_year <= last_year or first_year <= end_year <= last_year or start_year <= first_year <= end_year or start_year <= last_year <= end_year:
      filtered_files.append(file)
  return filtered_files

def get_all_files(directory):
  """Retrieves all files in a directory, including subdirectories.

  Args:
    directory: The path to the directory.

  Returns:
    A list of file paths.
  """

  files = []
  for root, dirs, filenames in os.walk(directory):
    for filename in filenames:
        if filename[-3:] == '.nc':
            files.append(os.path.join(root, filename))
  return files   

def filter_files_by_var_name(filenames, var_name):
    filenames = [f for f in filenames if f.split("/")[-1][:len(var_name[0])] == var_name[0]]
    return filenames

def check_year_gaps(year_ranges):
    """
    Check for gaps in a list of [start_year, end_year] ranges.

    Parameters:
    - year_ranges (list of lists): Each element is [start_year, end_year]

    Returns:
    - missing_years (list): List of individual years that are missing
    """
    # Sort the ranges by the starting year
    sorted_ranges = sorted(year_ranges, key=lambda x: x[0])

    # Collect all years from the ranges
    covered_years = set()
    for start, end in sorted_ranges:
        covered_years.update(range(start, end + 1))

    # Determine the overall span of years
    overall_start = min(start for start, _ in sorted_ranges)
    overall_end = max(end for _, end in sorted_ranges)

    # Find missing years
    all_years = set(range(overall_start, overall_end + 1))
    missing_years = sorted(all_years - covered_years)

    return missing_years

def get_available_years_range(filenames):
    available_years = []
    individual_files_range = []
    for file in filenames:
        range_years = file.split("_")[-1]
        first_year = int(range_years.split("-")[0][:4])  # Extract the first year from the filename
        last_year = int(range_years.split("-")[-1][:4])  # Extract the first year from the filename
        available_years.append(first_year)
        available_years.append(last_year)
        individual_files_range.append([first_year,last_year])
    available_years = list(set(available_years))
    min_year_available = np.min(available_years)
    max_year_available = np.max(available_years)
    missing_years = check_year_gaps(individual_files_range)
    if len(missing_years)>0:
        print('Missing files for years:', missing_years)
    else:
        print('There are no gaps in the data')
    return min_year_available, max_year_available

def choose_years(dict_years, delay, years_range):
    first_years = []
    last_years = []
    for variant in dict_years.keys():
        first_years.append(dict_years[variant][0])
        last_years.append(dict_years[variant][1])
    first_year = np.max(first_years)
    last_year = np.min(last_years)

    first_year_use = first_year +1 + delay
    last_year_use = first_year + years_range - 1 + 1 + delay
    if last_year_use <= last_year:
        return first_year_use, last_year_use
    else:
        raise Exception("Last year to use exceeds the last year of the avilable data")

def categorize_files_by_year_range(filenames):   
    categorized_files = []
    ranges = []
    for file in filenames:
        ranges.append(file[-20:-3])
    separate_ranges = list(set(ranges))
    for range_year in separate_ranges:
        categorized_files.append([f for f in filenames if range_year in f])
    return categorized_files

def calculating_ensemble_mean(filenames, var, silent=True):
    """Calculates Ensemble mean of the NetCDF files using the rolling mean method"""
    ensemble_mean_array = np.array([])
    for i in range(len(filenames)):       
        file = filenames[i]
        if not silent:
            print('Calculating:', file.split("/")[-1])
        ncfile = xr.open_dataset(file)
        ncfile = renaming_dimensions(ncfile)
        ncfile['lat'] = ncfile['lat'].round(3)
        ncfile['lon'] = ncfile['lon'].round(3)
        if i==0:
            ensemble_mean_array = copy.deepcopy(ncfile[var])
        else:
            ensemble_mean_array = ensemble_mean_array*i/(i+1) + ncfile[var]*1/(i+1)
    return ensemble_mean_array

def levels_correction(ds, levels_list):
    try:
        ds['level']
    except:
        try:
            ds['plev']
            ds = ds.rename({'plev':'level'})
        except:
            try:
                ds['pressure']
                ds = ds.rename({'pressure':'level'})
            except:
                print('')
    if ds.level.max() > 10000:
        ds = ds.assign_coords(level=(ds.level/100))
    
    ds = ds.where((ds.level>=np.min(levels_list))&(ds.level<=np.max(levels_list)), drop=True)
    ds = ds.interp(level=levels_list, method='linear')
    return ds

def open_mask_file(path):
    ncfile = xr.open_dataset(path,engine = 'netcdf4')
    dataarray = ncfile['topo']
    dataarray = dataarray.where(dataarray.isnull(),0)
    dataarray = dataarray.where(dataarray.notnull(),1)
    dataarray = dataarray.where(dataarray!=0, np.nan)
    return dataarray

def ensemble_time_series(filenames_all,var,first_year,last_year):
    filtered_files = filter_files_by_year_range(filenames_all, first_year, last_year)
    print('All the filtered file are the following:')
    print(filtered_files)
    print("")
    categorized_files = categorize_files_by_year_range(filtered_files)
    for i,filenames2 in enumerate(categorized_files):  
        if i==0:
            ensemble_series = calculating_ensemble_mean(filenames2, var, silent = False)
        else:
            ensemble_series = xr.concat([ensemble_series,calculating_ensemble_mean(filenames2, var, silent=False)], dim='time').sortby('time') 
    ensemble_series = ensemble_series.where((ensemble_series.time.dt.year>=first_year)&(ensemble_series.time.dt.year<=last_year),drop=True)
    return ensemble_series

def time_series(filenames_all,var,first_year,last_year):
    filtered_files = filter_files_by_year_range(filenames_all, first_year, last_year)
    for i,file in enumerate(filtered_files):  
        ncfile = xr.open_dataset(file)
        ncfile = ncfile[var]
        ncfile['lat'] = ncfile['lat'].round(3)
        ncfile['lon'] = ncfile['lon'].round(3)
        if i==0:
            ensemble_series = ncfile
        else:
            ensemble_series = xr.concat([ensemble_series,ncfile], dim='time').sortby('time') 
    ensemble_series = ensemble_series.where((ensemble_series.time.dt.year>=first_year)&(ensemble_series.time.dt.year<=last_year),drop=True)
    return ensemble_series

def renaming_dimensions(ncfile):
    variables = ncfile.dims
    if 'latitude' in variables:
        ncfile = ncfile.rename({'latitude':'lat'})
    if 'longitude' in variables:
        ncfile = ncfile.rename({'longitude':'lon'})
    if 'plev' in variables:
        ncfile = ncfile.rename({'plev':'level'})
    elif 'pressure' in variables:
        ncfile = ncfile.rename({'pressure':'level'})
    elif 'pressure_level' in variables:
        ncfile = ncfile.rename({'pressure_level':'level'})
    if 'date' in variables: 
        ncfile = ncfile.rename({'date':'time'})
        ncfile['time'] = pd.to_datetime(ncfile['time'].astype(str), format='%Y%m%d')
    if 'valid_time' in variables: 
        ncfile = ncfile.rename({'valid_time':'time'})
    if 'expver' and 'number' in ncfile.coords:
        ncfile = ncfile.drop_vars(['expver', 'number'])
    return ncfile

def convert_lon_to_360(da):
    """
    Convert longitudes from the range [-180, 180] to [0, 360].

    Parameters:
    da (xr.DataArray or xr.Dataset): Input data with longitude dimension.

    Returns:
    xr.DataArray or xr.Dataset: Data with longitudes converted to [0, 360].
    """
    if da['lon'].min() < 0:
        #print('Chaning coordinates from [-180,180] to [0,360]')
        da = da.assign_coords(lon=(da['lon'] % 360))
        da = da.sortby('lon')
    return da

def preparing_pr_data(files,var_name,first_year,last_year,dict_lonlat):
    dataaaray = ensemble_time_series(files, var_name, first_year, last_year)
    dataaaray = renaming_dimensions(dataaaray)
    dataaaray = add_longitudes_circular(dataaaray)
    dataaaray = add_poles_latitudes(dataaaray)
    dataaaray = remapping(dataaaray, dict_lonlat)
    dataaaray = dataaaray.transpose('time', 'lat', 'lon')
    if dataaaray.max() < 1e-2:
        #print('Multiplying by 86400')
        dataaaray = dataaaray*86400  ### From flux per second, to daily averaged precipitation
    return dataaaray

def preparing_var_data(files,var_name,first_year,last_year):
    dataarray = time_series(files, var_name, first_year, last_year)
    return dataarray

def add_longitudes_circular(da):
    lons = da['lon'].values
    if 360 not in lons:
        # Extract the data corresponding to the closest at lon=0
        data_at_near_zero_lon = da.sel(lon=0, method='nearest')    
        # Create a new DataArray with lon=360, copying the data from lon=0
        new_data_at_near360 = data_at_near_zero_lon.expand_dims({'lon': [360+np.min(lons)]}, axis=-1)
        da = xr.concat([da , new_data_at_near360], dim='lon')
        da = da.sortby('lon')
    if 0 not in lons:
        # Extract the data corresponding to the closest at lon=360
        data_at_near_360_lon = da.sel(lon=360, method='nearest')
        new_data_at_near0 = data_at_near_360_lon.expand_dims({'lon': [np.max(lons)-360]}, axis=-1)
        # Combine the new data with the original DataArray
        da = xr.concat([new_data_at_near0 ,da], dim='lon')
        
        # Sort the longitude dimension if needed (optional)
        da = da.sortby('lon')
    return da

def add_poles_latitudes(da):
    lats = da['lat'].values
    if 90 not in lats:
        # Extract the data corresponding to the closest at lon=0
        data_at_near_north_pole = da.sel(lat=90, method='nearest')    
        # Create a new DataArray with lon=360, copying the data from lon=0
        new_data_at_north_pole = data_at_near_north_pole.expand_dims({'lat': [90]}, axis=-1)
        da = xr.concat([da , new_data_at_north_pole], dim='lat')
        da = da.sortby('lat')
    if -90 not in lats:
        # Extract the data corresponding to the closest at lon=360
        data_at_near_south_pole = da.sel(lat=-90, method='nearest')
        new_data_at_near_south_pole = data_at_near_south_pole.expand_dims({'lat': [-90]}, axis=-1)
        # Combine the new data with the original DataArray
        da = xr.concat([new_data_at_near_south_pole ,da], dim='lat')
        
        # Sort the longitude dimension if needed (optional)
        da = da.sortby('lat')
    return da

def remapping(array_var,dict_lonlat):
    lons = np.linspace(0, 360, dict_lonlat['lon'])
    lats = np.linspace(-90, 90, dict_lonlat['lat'])
    print('Remapping data from {}x{} to {}x{} resolution'.format(len(array_var.lon),len(array_var.lat),len(lons),len(lats)))
    array_var = array_var.interp(lon=lons,lat=lats,method='linear')
    return array_var

def compute_seasonal_mean(ds, season, first_year, last_year): 
    #### Computes the seasonal mean, in case of DJF season, it takes the December from the previous year #####
    ds = ds.sel(time=ds.time.dt.month.isin(season))

    if set([12,1]).issubset(season):
        # Shift December to next year
        year = ds.time.dt.year
        year = xr.where(ds.time.dt.month == 12, year + 1, year)

        ds = ds.assign_coords(season_year=year)

        # Compute mean
        ds_season = ds.groupby('season_year').mean(dim='time')

        # Ensure full DJF only
        counts = ds.groupby('season_year').count(dim='time')
        ds_season = ds_season.where(counts == 3, drop=True)
        ds_season = ds_season.rename({'season_year': 'year'})
    else:
        ds_season = ds.groupby('time.year').mean(dim='time')

        # Ensure full seasons (optional but good practice)
        counts = ds.groupby('time.year').count(dim='time')
        ds_season = ds_season.where(counts == len(season), drop=True)

    # Select years
    ds_season = ds_season.where(
        (ds_season.year >= first_year) & (ds_season.year <= last_year),
        drop=True
    )

    return ds_season

def zonal_mean_time_lat_cut_data(dataarray,season,first_year, last_year,
                                 min_lat,max_lat,lons_region):
    dataarray_season = compute_seasonal_mean(dataarray, season, first_year, last_year)
    dataarray_region = longitude_cut(dataarray_season,lons_region)
    dataarray_region = dataarray_region.mean(dim="lon")
    dataarray_region = dataarray_region.where((dataarray_region.lat> min_lat)&(dataarray_region.lat<max_lat), drop=True)
    return dataarray_region

def longitude_cut(P,lons):
    str_lons = check_longitude_system(lons)
    if str_lons == "180":
        P = convert_lon_to_minus180_180(P)
    elif str_lons == "360":
        P = convert_lon_to_360(P)
    else:
        raise ValueError("Unknown longitude system")

    P = P.sel(lon=slice(lons[0], lons[1]))
    return P


def check_longitude_system(longitudes):
    lon_left, lon_right = longitudes

    # 1. If any value is negative, it must be the -180 to 180 system.
    if lon_left < 0 or lon_right < 0:
        return "180"

    # 2. If any value exceeds 180, it must be the 0 to 360 system.
    if lon_left > 180 or lon_right > 180:
        return "360"

    # 3. If lon_left > lon_right, it suggests a wrap-around the 180 meridian
    # in a -180 to 180 system (e.g., [170, -170]).
    if lon_left > lon_right:
        return "180"

    # 4. Default case: If values are between 0 and 180, they work in both.
    # We usually default to -180 to 180 as it's the standard for GPS/WGS84.
    return "180"

def make_level_descending(file_nc):
    if file_nc.level.values[0] < file_nc.level.values[1]:
        file_nc = file_nc.isel(level=slice(None, None, -1))
    return file_nc

def make_level_ascending(file_nc):
    if file_nc.level.values[0] > file_nc.level.values[1]:
        file_nc = file_nc.isel(level=slice(None, None, -1))
    return file_nc

def mass_streamfunction2(va_array, psi_array, a ,g):
    ###### Calculates the mass stremafunction #######
    va_array = make_level_ascending(va_array)
    psi_array = make_level_ascending(psi_array)
    latitudes = va_array.lat.values
    levels = va_array.level.values
    years = va_array.year.values
    for year in years:
        for lat in latitudes:
            va_values = va_array.sel(lat = lat, year=year)
            press_values = np.append(0,levels * 100)
            dp = np.diff(press_values)
            psi_values = (2 * np.pi * a * np.cos(np.radians(lat)) / g) *np.cumsum(va_values.values*dp)

            psi_array.loc[dict(lat=lat, year=year)] = psi_values
    psi_array = make_level_descending(psi_array)
    return psi_array

def get_hatching_mask(da, threshold, agreement):
    """
    Creates a 0/1 mask where 1 indicates model disagreement.
    
    Parameters:
    da (xr.DataArray): Input data with a 'model' dimension.
    threshold (int): Number of models that must agree (e.g., 3 or 4).
    """
    # Count how many models are strictly positive
    pos_agreement = (da > 0).sum(dim='model')
    
    # Count how many models are strictly negative
    neg_agreement = (da < 0).sum(dim='model')
    
    # Create mask where either positive OR negative agreement meets the threshold
    # The .astype(int) converts True/False to 1/0
    if agreement:
        mask = (((pos_agreement >= threshold) | (neg_agreement >= threshold))).astype(int)
    else:
        mask = (~((pos_agreement >= threshold) | (neg_agreement >= threshold))).astype(int)
    
    return mask, da.mean(dim='model')

def calculate_model_mean_diff(dict_comp, dict_idealized, 
                              years_use_comp, years_use_idealized,
                              season, variable, min_lat, max_lat, lons_regions):
    a = 6371000  # Earth's radius in meters
    g = 9.81  # gravity in m/s^2
    variant_list = []
    for comp_variant in dict_comp.keys():
        variant = dict_comp[comp_variant]
        first_year, last_year = years_use_comp[comp_variant]
        variant_season = zonal_mean_time_lat_cut_data(variant, season, first_year, last_year,
                                                       min_lat, max_lat, lons_regions)

        if variable == 'va':
            variant_season = make_level_descending(variant_season)
            psi_values_variant = xr.zeros_like(variant_season)
            variant_season = mass_streamfunction2(variant_season, psi_values_variant, a, g)
        variant_season = variant_season.assign_coords(variant=comp_variant)
        variant_list.append(variant_season)
    combined_comp = xr.concat(variant_list, dim="variant")

    variant_list = []
    for idea_variant in dict_idealized.keys():
        variant_idea = dict_idealized[idea_variant]
        first_year, last_year = years_use_idealized[idea_variant]
        variant_idea_season = zonal_mean_time_lat_cut_data(variant_idea, season, first_year, last_year,
                                                            min_lat, max_lat, lons_regions)

        if variable == 'va':
            variant_idea_season = make_level_descending(variant_idea_season)
            psi_values_idea_variant = xr.zeros_like(variant_idea_season)
            variant_idea_season = mass_streamfunction2(variant_idea_season, psi_values_idea_variant, a, g)
        variant_idea_season = variant_idea_season.assign_coords(variant=idea_variant)
        variant_list.append(variant_idea_season)
    combined_idealized = xr.concat(variant_list, dim="variant")
    
    std_comp = combined_comp.std(dim=['variant', 'year'])
    combined_comp = combined_comp.mean(dim=['variant', 'year'])
    combined_idealized = combined_idealized.mean(dim=['variant', 'year'])
    

    difference = combined_idealized - combined_comp
    snr = np.abs(difference)/std_comp
    snr_bool = xr.where(snr>1,1,0)

    return difference, combined_idealized, combined_comp, snr_bool


def multi_model(dict_comp_all, dict_idealized_all, 
                dict_years_use_comp_all, dict_years_use_idealized_all,
                season, variable, min_lat, max_lat, lons_regions):
    ##### Calculates the multi-model mean of the difference between GWL and piControl, as well as multi-model mean climatology for each of piControl and GWL runs
    model_diff_list=[]
    model_idea_list=[]
    model_comp_list=[]
    snr_list = []
    for model_name in dict_comp_all.keys():
        dict_comp = dict_comp_all[model_name]
        dict_idealized = dict_idealized_all[model_name]
        years_use_comp = dict_years_use_comp_all[model_name]
        years_use_idealized = dict_years_use_idealized_all[model_name]
        mean_model_difference, mean_model_idea, mean_model_comp, snr = calculate_model_mean_diff(dict_comp, dict_idealized,
                                                                      years_use_comp, years_use_idealized,
                                                                      season, variable, min_lat, max_lat, lons_regions)
        mean_model_difference = mean_model_difference.assign_coords(model=model_name)
        mean_model_idea = mean_model_idea.assign_coords(model=model_name)
        mean_model_comp = mean_model_comp.assign_coords(model=model_name)
        snr = snr.assign_coords(model=model_name)
        model_diff_list.append(mean_model_difference)
        model_idea_list.append(mean_model_idea)
        model_comp_list.append(mean_model_comp)
        snr_list.append(snr)
    combined_model_diff = xr.concat(model_diff_list, dim="model")
    combined_model_idea = xr.concat(model_idea_list, dim="model")
    combined_model_comp = xr.concat(model_comp_list, dim="model")
    combined_snr_model = xr.concat(snr_list, dim="model")
    return combined_model_diff, combined_model_comp, combined_model_idea, combined_snr_model

def zero_crossing_models(da_model):
    # Step 1: Find indices where the sign changes
    # (np.sign returns -1, 0, or 1. np.diff finds where it jumps)
    list_lats = []
    for model in da_model.model:
        da = da_model.sel(model=model)
        sign_changes = np.diff(np.sign(da.values)) != 0
        idx = np.where(sign_changes)[0]

        # Step 2: Grab the values/latitudes right before and after the crossing
        lat1 = da.lat.values[idx]
        lat2 = da.lat.values[idx + 1]
        val1 = da.values[idx]
        val2 = da.values[idx + 1]
        
        # Step 3: Apply the linear interpolation formula
        zero_lats = lat1 - val1 * (lat2 - lat1) / (val2 - val1)
        zero_lat_arg = np.abs(zero_lats).argmin()
        list_lats.append(zero_lats[zero_lat_arg])
    return list_lats

def simple_metrics_models(da_psi_comp, da_psi_idea, da_wap_comp, da_wap_idea, season_n):
    ###### Calculates the simple circulation metrics and returns a Dataframe ######
    psi_comp = da_psi_comp
    psi_comp = psi_comp.where((psi_comp.level<=800)&(psi_comp.level>=300),drop=True).mean(dim='level', skipna=True)
    psi_idea = da_psi_idea
    psi_idea = psi_idea.where((psi_idea.level<=800)&(psi_idea.level>=300),drop=True).mean(dim='level', skipna=True)

    wap_comp = da_wap_comp
    wap_comp = wap_comp.where((wap_comp.level<=800)&(wap_comp.level>=300),drop=True).mean(dim='level', skipna=True)
    wap_idea = da_wap_idea
    wap_idea = wap_idea.where((wap_idea.level<=800)&(wap_idea.level>=300),drop=True).mean(dim='level', skipna=True)

    max_psi_comp = np.abs(psi_comp).max(dim='lat').values
    max_psi_idea = np.abs(psi_idea).max(dim='lat').values

    if season_n == 'DJF':
        min_lat_s = -30
        max_lat_s = 5
        min_lat_n = 10
        max_lat_n = 45
    else:
        min_lat_n = -5
        max_lat_n = 30
        min_lat_s = -45
        max_lat_s = -10
    psi_comp_zero_lat_s = zero_crossing_models(psi_comp.sel(lat=slice(min_lat_s, max_lat_s)))
    psi_idea_zero_lat_s = zero_crossing_models(psi_idea.sel(lat=slice(min_lat_s, max_lat_s)))

    psi_comp_zero_lat_n = zero_crossing_models(psi_comp.sel(lat=slice(min_lat_n, max_lat_n)))
    psi_idea_zero_lat_n = zero_crossing_models(psi_idea.sel(lat=slice(min_lat_n, max_lat_n)))

    if season_n == 'DJF':
        wap_min_comp = wap_comp.sel(lat=slice(min_lat_s, max_lat_s)).min(dim='lat').values
        wap_min_idea = wap_idea.sel(lat=slice(min_lat_s, max_lat_s)).min(dim='lat').values
    else:
        wap_min_comp = wap_comp.sel(lat=slice(min_lat_n, max_lat_n)).min(dim='lat').values
        wap_min_idea = wap_idea.sel(lat=slice(min_lat_n, max_lat_n)).min(dim='lat').values

    data = {
        'Season': [season_n, season_n],
        'run': ['Comp', 'Idealized'],
        'PSI max value': [np.mean(max_psi_comp), np.mean(max_psi_idea)],
        'PSI max value std': [np.std(max_psi_comp), np.std(max_psi_idea)],
        'Hadley cell southern edge': [np.mean(psi_comp_zero_lat_s), np.mean(psi_idea_zero_lat_s)],
        'Hadley cell southern edge std': [np.std(psi_comp_zero_lat_s), np.std(psi_idea_zero_lat_s)],
        'Hadley cell northern edge': [np.mean(psi_comp_zero_lat_n), np.mean(psi_idea_zero_lat_n)],
        'Hadley cell northern edge std': [np.std(psi_comp_zero_lat_n), np.std(psi_idea_zero_lat_n)],
        'Wap max updraft': [np.mean(wap_min_comp), np.mean(wap_min_idea)],
        'Wap max updraft std': [np.std(wap_min_comp), np.std(wap_min_idea)],\
    }   
    df = pd.DataFrame(data)
    return df 

def colorbar_contour(levels_original, cmap_name, vmin, vmax, mult_factor):
    vmin = vmin*mult_factor
    vmax = vmax*mult_factor
    levels_mult = [f*mult_factor for f in levels_original]
    base_cmap = plt.get_cmap(cmap_name)

    # Ensure original levels are a sorted numpy array
    levels_orig = np.sort(np.array(levels_mult))

    # 1. Sample the exact colors AT the specified levels
    colors = []
    for val in levels_orig:
        normalized_val = (val - vmin) / (vmax - vmin)
        normalized_val = np.clip(normalized_val, 0.0, 1.0)
        colors.append(base_cmap(normalized_val))

    # 2. Dynamically calculate 6 boundaries to enclose the 5 levels cleanly.
    # We find the midpoints between your levels to act as internal boundaries.
    midpoints = [
        (levels_orig[i] + levels_orig[i + 1]) / 2
        for i in range(len(levels_orig) - 1)
    ]

    # Extrapolate the outer edges so the first and last levels are perfectly centered
    left_edge = levels_orig[0] - (midpoints[0] - levels_orig[0])
    right_edge = levels_orig[-1] + (levels_orig[-1] - midpoints[-1])

    # Combine them into the final boundaries array (length N + 1)
    boundaries = [left_edge] + midpoints + [right_edge]
    boundaries = np.array(boundaries)

    # 3. Create discrete colormap and normalizer mapping colors to these new intervals
    custom_cmap, norm = mpl.colors.from_levels_and_colors(
        boundaries, colors, extend="neither"
    )

    # 4. Set up the ScalarMappable
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array([])

    # 5. Your original levels become the exact tick locations (centered in each block)
    tick_locs = list(levels_orig)

    # 6. Format the tick labels cleanly
    tick_labels = []
    for loc in tick_locs:
        if np.isclose(loc, 0.0, atol=1e-7):
            tick_labels.append("0")
        else:
            tick_labels.append(f"{loc:.4f}".rstrip("0").rstrip("."))

    return sm, tick_locs, tick_labels


def plot_streamfunctions_change(idealized_dict_va_all, comp_dict_va_all, idealized_dict_wap_all,
                                  comp_dict_wap_all, idealized_dict_pr_all, comp_dict_pr_all,
                                  dict_years_use_comp_all, dict_years_use_idealized_all,
                                  area, bottom_level, top_level, min_lat, max_lat, 
                                  seasons, lons_regions, out_file, name_of_plot, name_runs):

    fig, axes = plt.subplots(3,4,figsize=(30,18),
                                    gridspec_kw={'width_ratios': [15,15, 1, 1]},  # Ensure equal heights
                                    )
    # Unpack the axes
    ax11, ax12, ax_cb11, ax_cb12 = axes[0]
    ax21, ax22, ax_cb21, ax_cb22 = axes[1]
    ax31, ax32, ax_cb31, ax_cb32 = axes[2]  # No colorbar for the third plot, so leave it empty
    dfs_to_combine = []
    for season_n in list(seasons.keys()):
        if season_n == 'JFM' or season_n == 'DJF':
            ax1 = ax11
            ax2 = ax21
            ax3 = ax31
        else:
            ax1 = ax12
            ax2 = ax22
            ax3 = ax32
        print(area, season_n)
        season = seasons[season_n]

        print('psi')
        change_psi_multimodel, multimodel_psi_comp, multimodel_psi_idea, snr_psi = multi_model(comp_dict_va_all, idealized_dict_va_all, 
                                                                 dict_years_use_comp_all, dict_years_use_idealized_all,
                                                                 season, 'va', min_lat, max_lat, lons_regions)
        print('wap')
        change_wap_multimodel, multimodel_wap_comp, multimodel_wap_idea, snr_wap = multi_model(comp_dict_wap_all, idealized_dict_wap_all, 
                                                                 dict_years_use_comp_all, dict_years_use_idealized_all,
                                                                 season, 'wap', min_lat, max_lat, lons_regions)
        print('pr')
        change_pr_multimodel, multimodel_pr_comp, _,_ = multi_model(comp_dict_pr_all, idealized_dict_pr_all, 
                                                               dict_years_use_comp_all, dict_years_use_idealized_all,
                                                                 season, 'pr', min_lat, max_lat, lons_regions)
        df_season = simple_metrics_models(multimodel_psi_comp, multimodel_psi_idea, multimodel_wap_comp, multimodel_wap_idea, season_n)
        dfs_to_combine.append(df_season)

        plot_in_contours_psi = multimodel_psi_comp
        plot_in_contours_wap = multimodel_wap_comp

        levels_ps=np.arange(-1.2e11, 1.6e11, 4e10)
        levels2_ps=np.arange(-3e10, 3.5e10, 0.5e10)
        
        c11 = ax1.contourf(change_psi_multimodel.lat, change_psi_multimodel.level, change_psi_multimodel.mean(dim='model').values, 
                        levels=levels2_ps, cmap='BrBG_r', extend='both', alpha=0.7)
        

        ax1.set_ylim(bottom_level,top_level)

        ax1.contour(plot_in_contours_psi.lat, plot_in_contours_psi.level, plot_in_contours_psi.mean(dim='model').values, 
                    levels=levels_ps, vmin=-1.4e11, vmax= 1.4e11, colors='black', linewidths=4, linestyles='-')
        
        ax1.contour(plot_in_contours_psi.lat, plot_in_contours_psi.level, plot_in_contours_psi.mean(dim='model').values, 
                    levels=levels_ps, vmin=-1.4e11, vmax= 1.4e11, cmap='bwr', linewidths=3)
        print(np.max(np.abs(plot_in_contours_psi.mean(dim='model'))))
        hatch_model_agreement_sign, mean_change = get_hatching_mask(change_psi_multimodel, 5, agreement=True)
        hatch_model_agreement_robust, _ = get_hatching_mask(snr_psi, 4, agreement=True)
        hatch = hatch_model_agreement_sign * hatch_model_agreement_robust
    
        density=1
        ax1.contourf(hatch.lat,hatch.level,hatch,levels=[0.5,1.5],
                        colors='none', alpha=0,hatches=[density*'/',2*density*'/'], 
                        label='Agreement')
        

        ax1.set_ylabel('Pressure (hPa)', fontsize=25)
        #ax1.set_xlabel('Latitude', fontsize=15)
        ax1.tick_params(axis='x', which='major', labelsize=20)
        ax1.tick_params(axis='y', which='major', labelsize=20)

        ax1.set_title('{}\n Mass Streamfunction'.format(season_n), fontsize=25)
        ax2.set_title('Pressure Velocity', fontsize=25)
        ax3.set_title('Precipitation Change', fontsize=25)

        #####################
        #####################
        #####################

        levels2_wap = np.arange(-0.008,0.010,0.002)
        levels_wap = np.arange(-0.05,0.06,0.01)

        c22 = ax2.contourf(change_wap_multimodel.lat, change_wap_multimodel.level, change_wap_multimodel.mean(dim='model').values,
                        vmin=-0.008,vmax=0.008,cmap='BrBG_r',levels=levels2_wap,extend='both', alpha=0.7)
        ax2.set_ylim(bottom_level,top_level)

        ax2.contour(plot_in_contours_wap.lat, plot_in_contours_wap.level, plot_in_contours_wap.mean(dim='model').values,
                    vmin=-0.05,vmax=0.05,colors='black',levels=levels_wap, linewidths=4, linestyles='-')
        ax2.contour(plot_in_contours_wap.lat, plot_in_contours_wap.level, plot_in_contours_wap.mean(dim='model').values,
                    vmin=-0.06,vmax=0.06,cmap='bwr_r',levels=levels_wap, linewidths=3, linestyles='-')
        
        hatch_model_agreement_sign, mean_change = get_hatching_mask(change_wap_multimodel, 5, agreement=True)
        hatch_model_agreement_robust, _ = get_hatching_mask(snr_wap, 4, agreement=True)
        hatch = hatch_model_agreement_sign * hatch_model_agreement_robust

        density=1
        ax2.contourf(hatch.lat,hatch.level,hatch,levels=[0.5,1.5],colors='none', alpha=0,hatches=[density*'/',2*density*'/'], label='Strengthening')
        
        ax2.set_ylabel('Pressure (hPa)', fontsize=25)
        
        ax2.tick_params(axis='x', which='major', labelsize=20)
        ax2.tick_params(axis='y', which='major', labelsize=20)
        

        ax3.plot(change_pr_multimodel.lat.values, change_pr_multimodel.mean(dim='model').values,color='black')
        ax3.set_ylim(-2,2)
        ax3.set_xlim(np.min(change_pr_multimodel.lat.values),np.max(change_pr_multimodel.lat.values))
        ax3.axhline(y=0, color='grey', linestyle='-',alpha=0.7)
        # Fill the area above the line with blue
        ax3.fill_between(change_pr_multimodel.lat.values, change_pr_multimodel.mean(dim='model').values, 0, where=change_pr_multimodel.mean(dim='model').values >= 0, color='blue', alpha=0.5)
        
        # Fill the area below the line with red
        ax3.fill_between(change_pr_multimodel.lat.values, change_pr_multimodel.mean(dim='model').values, 0, where=change_pr_multimodel.mean(dim='model').values < 0, color='red', alpha=0.5)
        ax3.tick_params(axis='x', which='major', labelsize=20)
        ax3.tick_params(axis='y', which='major', labelsize=20)
        ax3.set_ylabel('Precipitation difference (mm)', fontsize=25)
        ax3.set_xlabel('Latitude', fontsize=25)
        


    cbar11 = plt.colorbar(c11,ax=ax_cb11,fraction=0.95)
    cbar11.ax.tick_params(labelsize=15)
    cbar11.set_label('Mass Streamfunction Difference ($10^{10}$ kg $s^{-1}$)',fontsize=20)
    cbar11.ax.yaxis.get_offset_text().set_visible(False)
    
    sm_ps, tick_locs_ps, tick_labels_ps = colorbar_contour(levels_ps, 'bwr', -1.4e11, 1.4e11, 1/1e11)

    cbar12 = plt.colorbar(sm_ps, ticks=tick_locs_ps, ax=ax_cb12, fraction=0.95)
    cbar12.ax.set_yticklabels(tick_labels_ps)
    cbar12.ax.tick_params(labelsize=15)
    cbar12.set_label('Mass Streamfunction ($10^{11}$ kg $s^{-1}$)',fontsize=20)
    cbar12.ax.yaxis.get_offset_text().set_visible(False)

    cbar21 = plt.colorbar(c22,ax=ax_cb21,fraction=0.95)
    cbar21.ax.tick_params(labelsize=15)
    cbar21.set_label('Pressure Velocity Difference (Pa $s^{-1}$)',fontsize=20)
    cbar21.ax.yaxis.get_offset_text().set_visible(False)
    
    sm_wap, tick_locs_wap, tick_labels_wap = colorbar_contour(levels_wap, 'bwr_r', -0.06, 0.06, 1)

    cbar22 = plt.colorbar(sm_wap, ticks=tick_locs_wap, ax=ax_cb22, fraction=0.95)
    cbar22.ax.set_yticklabels(tick_labels_wap)
    cbar22.ax.tick_params(labelsize=15)
    cbar22.set_label('Pressure Velocity (Pa $s^{-1}$)',fontsize=20)
    cbar22.ax.yaxis.get_offset_text().set_visible(False)
    
    ax11.annotate(
        '(a)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax12.annotate(
        '(b)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax21.annotate(
        '(c)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax22.annotate(
        '(d)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax31.annotate(
        '(e)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax32.annotate(
        '(f)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    
    for ax_cb in [ax_cb11, ax_cb21, ax_cb31, ax_cb12, ax_cb22, ax_cb32]:
        ax_cb.spines[['top', 'bottom', 'left', 'right']].set_visible(False)  # Remove axes borders
        ax_cb.set_xticks([])  # Remove x ticks
        ax_cb.set_yticks([])  # Remove y ticks
        ax_cb.tick_params(which='both', bottom=False, top=False, left=False, right=False)  # Remove tick marks
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(out_file,name_of_plot+'_{}.pdf'.format(area)), dpi=300)

    df_final = pd.concat(dfs_to_combine, ignore_index=True)
    df_final.to_csv(os.path.join(out_file, 'Metrics_criculation_models_{}.csv'.format(name_runs)))
    print(df_final)


def streamfunction_plotting(idealized_dict_va_all,idealized_dict_wap_all,idealized_dict_pr_all,
                            comp_dict_va_all, comp_dict_wap_all, comp_dict_pr_all, 
                            dict_years_use_comp_all, dict_years_use_idealized_all,
                            out_file, name):    
    ##### Plots a figure for each region specified
    ##### Also choose min and max latitude of plotting, as well as top and bottom pressure level
    name_of_plot = 'Streamfunction_change_omega_{}'.format(name)

    seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    regions = {'World':[0,360]}
    
    min_lat = -60.0
    max_lat = 60.0
    bottom_level = 1000
    top_level = 100
    
    for area in list(regions.keys()):   
        lons_regions = regions[area]
        print(lons_regions)
        plot_streamfunctions_change(idealized_dict_va_all, comp_dict_va_all, idealized_dict_wap_all,
                                comp_dict_wap_all, idealized_dict_pr_all, comp_dict_pr_all, 
                                dict_years_use_comp_all, dict_years_use_idealized_all,
                                area, bottom_level, top_level, min_lat, max_lat, 
                                seasons, lons_regions, out_file, name_of_plot, name)

if __name__ == '__main__':
    #########################################################################################################
    #### This code plots the the multimodel mean of the change of the zonal mean of mass stremafunction #####
    #### pressure velocity and precipitation, for the DJF and JJA seasons for a specific GWL            #####
    #########################################################################################################
    model_names = ['GFDL-ESM2M', 'NorESM2-LM', 'MIROC-ES2L', 'UKESM1-2', 'EC-Earth3', 'IPSL', 'CNRM']

    out_path = '/path/to/output/'  ### Directory to output the figures and csv files
    gwl_runs_do = ['esm-up2p0-gwl2p0', 'esm-up2p0-gwl4p0']  ### Which GWL stabilzation runs to compare to piControl runs

    var_name_wap = 'wap'
    var_name_va = 'va'
    var_name_pr = 'pr'
    

    delay = 100   # Years to skip at the beginning of each run
    years_range = 50 # How many years the analysis will span for each run

    for gwl_run in gwl_runs_do:
        comp_dict_va_all = {}
        comp_dict_pr_all = {}
        comp_dict_wap_all = {}

        idealized_dict_va_all = {}
        idealized_dict_pr_all = {}
        idealized_dict_wap_all = {}

        dict_years_use_comp_all = {}
        dict_years_use_idealized_all = {}


        for model_name in model_names:
            print(model_name)
            #### The path to GWL stabilzation runs should be in the form of 
            #### /path/to/data/MODEL_NAME/TIPMIP/GWL_RUN/
            path_idealized = '/path/to/data/{}/TIPMIP/{}/'.format(model_name, gwl_run)   
            #### The path to Pre-Industrial Control runs should be in the form of 
            #### /path/to/data/MODEL_NAME/CMIP6Plus/piControl/
            path_hists = '/path/to/data/{}/CMIP6Plus/piControl/'.format(model_name)
            
            #### In each directory with data for instance: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/
            #### The next directory will be the variant number as r1, r2, r3 etc
            #### E.g.: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/r1/
            #### And then the variable name:
            #### E.g.: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/r1/pr/
            #### This Python code, needs the variables: 'pr', 'calculated_wap' and 'calculated_va', to work
            #### where 'pr' is the raw pr data from the run, and the 'calculated_wap' and 'calculated_va' is the 
            #### remapped 'wap' and 'va' variables respectively at 1 degree resolution, with terrain masking
                
        
            if path_hists[-1] == '/':
                run_comp = path_hists.split('/')[-2]
            else:
                run_comp = path_hists.split('/')[-1]

            if path_idealized[-1] == '/':
                run_gwl = path_idealized.split('/')[-2]
                model_name = path_idealized.split('/')[-4]
            else:
                run_gwl = path_idealized.split('/')[-1]
                model_name = path_idealized.split('/')[-3]

            

            comp_dict_va = {}
            comp_dict_pr = {}
            comp_dict_wap = {}

            idealized_dict_va = {}
            idealized_dict_pr = {}
            idealized_dict_wap = {}

            dict_years_use_comp = {}
            dict_years_use_idealized = {}

            lons=360
            lats=180
            
            dict_lonlat = {'lon':lons, 
                        'lat':lats}
                

            #### Check available year ranges for the piControl runs
            avail_variants_comp = [f.path for f in os.scandir(path_hists) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
            dict_pi_years = {}
            for i, pi_variant in enumerate(avail_variants_comp):
                try:
                    files = get_all_files(os.path.join(path_hists,pi_variant,'pr'))
                    files = filter_files_by_var_name(files, 'pr')
                    first_year_pi, last_year_pi = get_available_years_range(files)
                    dict_pi_years[pi_variant] = [first_year_pi, last_year_pi]
                except:
                    continue
            first_year_pi_use, last_year_pi_use = choose_years(dict_pi_years,delay,years_range)
            print('For the piControl the years between {} and {} will be used'.format(first_year_pi_use, last_year_pi_use))
            comp_years = [first_year_pi_use, last_year_pi_use]

            #### Check available year ranges for the GWL stabilization runs
            avail_variants_idealized = [f.path for f in os.scandir(path_idealized) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
            dict_gwl_years = {}
            
            for i, idealized_variant in enumerate(avail_variants_idealized):
                try:
                    files = get_all_files(os.path.join(path_idealized, idealized_variant,'pr'))
                    files = filter_files_by_var_name(files, 'pr')
                    first_year_gwl, last_year_gwl = get_available_years_range(files)
                    dict_gwl_years[idealized_variant] = [first_year_gwl, last_year_gwl]
                except:
                    continue
            first_year_gwl_use, last_year_gwl_use = choose_years(dict_gwl_years,delay,years_range)
            print('For the GWL the years between {} and {} will be used'.format(first_year_gwl_use, last_year_gwl_use))
            idea_years = [first_year_gwl_use, last_year_gwl_use]

            #################################################################
            #### Reading Monthly va data 
            #################################################################
            for i, pi_variant in enumerate(avail_variants_comp):
                files = get_all_files(os.path.join(path_hists,pi_variant,'calculated_{}'.format(var_name_va)))
                files = filter_files_by_var_name(files, var_name_va)
                print('Calculating {} historical variant'.format(pi_variant))
                ncfile = preparing_var_data(files,'va',first_year_pi_use -1 ,last_year_pi_use +1)

                comp_dict_va[pi_variant] = ncfile 
                dict_years_use_comp[pi_variant] = [first_year_pi_use, last_year_pi_use]

            for i, idealized_variant in enumerate(avail_variants_idealized):
                files = get_all_files(os.path.join(path_hists,idealized_variant,'calculated_{}'.format(var_name_va)))
                files = filter_files_by_var_name(files, var_name_va)
                print('Calculating {} historical variant'.format(idealized_variant))
                ncfile = preparing_var_data(files,var_name_va,first_year_gwl_use -1,last_year_gwl_use +1)

                idealized_dict_va[idealized_variant] = ncfile
                dict_years_use_idealized[idealized_variant] = [first_year_gwl_use, last_year_gwl_use]                


            #################################################################
            #### Reading Monthly wap data 
            #################################################################
            for i, pi_variant in enumerate(avail_variants_comp):
                files = get_all_files(os.path.join(path_hists,pi_variant,'calculated_{}'.format(var_name_wap)))
                files = filter_files_by_var_name(files, var_name_wap)
                print('Calculating {} historical variant'.format(pi_variant))
                ncfile = preparing_var_data(files,var_name_wap,first_year_pi_use - 1,last_year_pi_use + 1)

                comp_dict_wap[pi_variant] = ncfile   


            for i, pi_variant in enumerate(avail_variants_idealized):
                files = get_all_files(os.path.join(path_hists,pi_variant,'calculated_{}'.format(var_name_wap)))
                files = filter_files_by_var_name(files, var_name_wap)
                print('Calculating {} historical variant'.format(pi_variant))
                ncfile = preparing_var_data(files,var_name_wap,first_year_gwl_use - 1,last_year_gwl_use + 1)

                idealized_dict_wap[pi_variant] = ncfile    
            print(idealized_dict_wap.keys())

            #################################################################
            #### Reading Monthly pr data 
            #################################################################
            for i, pi_variant in enumerate(avail_variants_comp):
                files = get_all_files(os.path.join(path_hists,pi_variant,var_name_pr))
                files = filter_files_by_var_name(files, var_name_pr)
                print('Calculating {} historical variant'.format(pi_variant))
                ncfile = preparing_pr_data(files,var_name_pr,first_year_pi_use - 1,last_year_pi_use + 1,dict_lonlat)
                comp_dict_pr[pi_variant] = ncfile   


            for i, pi_variant in enumerate(avail_variants_idealized):
                files = get_all_files(os.path.join(path_hists,pi_variant,var_name_pr))
                files = filter_files_by_var_name(files, var_name_pr)
                print('Calculating {} historical variant'.format(pi_variant))
                ncfile = preparing_pr_data(files,var_name_pr,first_year_gwl_use - 1,last_year_gwl_use + 1,dict_lonlat)
                idealized_dict_pr[pi_variant] = ncfile     

            comp_dict_va_all[model_name] = comp_dict_va
            comp_dict_pr_all[model_name] = comp_dict_pr
            comp_dict_wap_all[model_name] = comp_dict_wap

            idealized_dict_va_all[model_name] = idealized_dict_va
            idealized_dict_pr_all[model_name] = idealized_dict_pr
            idealized_dict_wap_all[model_name] = idealized_dict_wap

            dict_years_use_comp_all[model_name] = dict_years_use_comp
            dict_years_use_idealized_all[model_name] = dict_years_use_idealized

        ##############################
        ####### Start Plotting #######
        ##############################
        streamfunction_plotting(idealized_dict_va_all, idealized_dict_wap_all, idealized_dict_pr_all, 
                                comp_dict_va_all, comp_dict_wap_all, comp_dict_pr_all, 
                                dict_years_use_comp_all, dict_years_use_idealized_all,
                                out_path, 'multimodel_{}_{}'.format(run_comp, run_gwl.split('-')[-1]))