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
import matplotlib.colors as mcolors
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
        all_variables = ncfile.data_vars

        variable = [f for f in all_variables if f.startswith(var)][0]
        ncfile = ncfile.rename_vars({variable:var})
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

def preparing_var_data(files,var_name,first_year,last_year):
    dataarray = time_series(files, var_name, first_year, last_year)
    return dataarray

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
    variant_list = []
    for comp_variant in dict_comp.keys():
        variant = dict_comp[comp_variant]
        first_year, last_year = years_use_comp[comp_variant]
        variant_season = zonal_mean_time_lat_cut_data(variant, season, first_year, last_year,
                                                       min_lat, max_lat, lons_regions)
        #variant_season = variant_season.mean(dim='year')
        variant_season = variant_season.assign_coords(variant=comp_variant)
        variant_list.append(variant_season)
    combined_comp = xr.concat(variant_list, dim="variant")

    variant_list = []
    for idea_variant in dict_idealized.keys():
        variant_idea = dict_idealized[idea_variant]
        first_year, last_year = years_use_idealized[idea_variant]
        variant_idea_season = zonal_mean_time_lat_cut_data(variant_idea, season, first_year, last_year,
                                                            min_lat, max_lat, lons_regions)
        #variant_idea_season = variant_idea_season.mean(dim='year')
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
    model_diff_list=[]
    #model_idea_list=[]
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
        #model_idea_list.append(mean_model_idea)
        model_comp_list.append(mean_model_comp)
        snr_list.append(snr)
    combined_model_diff = xr.concat(model_diff_list, dim="model")
    #combined_model_idea = xr.concat(model_idea_list, dim="model")
    combined_model_comp = xr.concat(model_comp_list, dim="model")
    combined_snr_model = xr.concat(snr_list, dim="model")
    return combined_model_diff, combined_model_comp, combined_snr_model

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

def reverse_binary_array(data_array):
    """
    Takes an array of 0s and 1s and flips them.
    """
    return 1 - data_array

def plot_moistureflux_change(idealized1_dict_waphus_all, idealized2_dict_waphus_all,
                                  comp_dict_waphus_all, dict_years_use_comp_all, 
                                  dict_years_use_idealized1_all, dict_years_use_idealized2_all,
                                  area, bottom_level, top_level, min_lat, max_lat, 
                                  seasons, lons_regions, out_file, labels, name_of_plot):
    g = 9.81  # gravity in m/s^2
    fig, axes = plt.subplots(2,3,figsize=(25,15),
                                    gridspec_kw={'width_ratios': [15,15, 2]},  # Ensure equal heights
                                    )
    # Unpack the axes
    ax11, ax12, ax13 = axes[0]
    ax21, ax22, ax23 = axes[1]
    for season_n in list(seasons.keys()):
        if season_n == 'JFM' or season_n == 'DJF':
            ax1 = ax11
            ax2 = ax21
        else:
            ax1 = ax12
            ax2 = ax22
        print(area, season_n)
        season = seasons[season_n]
        ####### Comparison piControl model
        print('waphus')
        change_waphus_multimodel1, multimodel_waphus_comp, snr_waphus1 = multi_model(comp_dict_waphus_all, idealized1_dict_waphus_all, 
                                                                 dict_years_use_comp_all, dict_years_use_idealized1_all,
                                                                 season, 'wap', min_lat, max_lat, lons_regions)
        
        change_waphus_multimodel2, _, snr_waphus2 = multi_model(comp_dict_waphus_all, idealized2_dict_waphus_all, 
                                                                 dict_years_use_comp_all, dict_years_use_idealized2_all,
                                                                 season, 'wap', min_lat, max_lat, lons_regions)

        change_waphus_multimodel1 = - change_waphus_multimodel1/g
        change_waphus_multimodel2 = - change_waphus_multimodel2/g
        multimodel_waphus_comp = - multimodel_waphus_comp/g

        plot_in_contours_waphus = multimodel_waphus_comp

        levels_wh=np.arange(-5e-5, 6e-5, 1e-5)
        levels2_wh=np.array([-2e-5, -1e-5, -0.5e-5, -0.2e-5, -0.1e-5, -0.05e-5, 0 , 0.05e-5, 0.1e-5, 0.2e-5, 0.5e-5, 1e-5, 2e-5])

        # 1. Get the colormap
        cmap = plt.get_cmap('BrBG')

        # 2. Create the BoundaryNorm
        # ncolors=cmap.N ensures it uses the full resolution of the colormap
        norm = mcolors.BoundaryNorm(boundaries=levels2_wh, ncolors=cmap.N)
        
        c11 = ax1.contourf(change_waphus_multimodel1.lat, change_waphus_multimodel1.level, change_waphus_multimodel1.mean(dim='model').values, 
                        levels=levels2_wh, cmap=cmap, norm=norm, extend='both', alpha=0.7)
        

        ax1.set_ylim(bottom_level,top_level)

        ax1.contour(plot_in_contours_waphus.lat, plot_in_contours_waphus.level, plot_in_contours_waphus.mean(dim='model').values, 
                    levels=levels_wh, vmin=-6e-5, vmax= 6e-5, colors='black', linewidths=4, linestyles='-')
        
        ax1.contour(plot_in_contours_waphus.lat, plot_in_contours_waphus.level, plot_in_contours_waphus.mean(dim='model').values, 
                    levels=levels_wh, vmin=-6e-5, vmax= 6e-5, cmap='bwr_r', linewidths=3)

        hatch_model_agreement_sign, mean_change = get_hatching_mask(change_waphus_multimodel1, 5, agreement=True)
        hatch_model_agreement_robust, _ = get_hatching_mask(snr_waphus1, 4, agreement=True)
        hatch = hatch_model_agreement_sign * hatch_model_agreement_robust
        #hatch = reverse_binary_array(hatch)
    
        density=1
        ax1.contourf(hatch.lat,hatch.level,hatch,levels=[0.5,1.5],
                        colors='none', alpha=0,hatches=[density*'/',2*density*'/'], 
                        label='Agreement')
        
        if season_n == 'JFM' or season_n == 'DJF':
            ax1.set_ylabel('{}\nPressure (hPa)'.format(labels[0]), fontsize=25)
        else:
            ax1.set_ylabel('Pressure (hPa)', fontsize=25)
        #ax1.set_xlabel('Latitude', fontsize=15)
        ax1.tick_params(axis='x', which='major', labelsize=20)
        ax1.tick_params(axis='y', which='major', labelsize=20)

        ax1.set_title('{}\n Vertical Moisture Flux'.format(season_n), fontsize=25)

        #####################
        #####################
        #####################

        plot_in_contours_waphus = multimodel_waphus_comp

        levels_wh=np.arange(-5e-5, 6e-5, 1e-5)
        levels2_wh=np.array([-2e-5, -1e-5, -0.5e-5, -0.2e-5, -0.1e-5, -0.05e-5, 0 , 0.05e-5, 0.1e-5, 0.2e-5, 0.5e-5, 1e-5, 2e-5])

        # 1. Get the colormap
        cmap = plt.get_cmap('BrBG')

        # 2. Create the BoundaryNorm
        # ncolors=cmap.N ensures it uses the full resolution of the colormap
        norm = mcolors.BoundaryNorm(boundaries=levels2_wh, ncolors=cmap.N, extend='both')
        
        ax2.contourf(change_waphus_multimodel2.lat, change_waphus_multimodel2.level, change_waphus_multimodel2.mean(dim='model').values, 
                        levels=levels2_wh, cmap=cmap, norm=norm, extend='both', alpha=0.7)
        

        ax2.set_ylim(bottom_level,top_level)

        ax2.contour(plot_in_contours_waphus.lat, plot_in_contours_waphus.level, plot_in_contours_waphus.mean(dim='model').values, 
                    levels=levels_wh, vmin=-6e-5, vmax= 6e-5, colors='black', linewidths=4, linestyles='-')
        
        ax2.contour(plot_in_contours_waphus.lat, plot_in_contours_waphus.level, plot_in_contours_waphus.mean(dim='model').values, 
                    levels=levels_wh, vmin=-6e-5, vmax= 6e-5, cmap='bwr_r', linewidths=3)

        hatch_model_agreement_sign, mean_change = get_hatching_mask(change_waphus_multimodel2, 5, agreement=True)
        hatch_model_agreement_robust, _ = get_hatching_mask(snr_waphus2, 4, agreement=True)
        hatch = hatch_model_agreement_sign * hatch_model_agreement_robust
        #hatch = reverse_binary_array(hatch)
    
        density=1
        ax2.contourf(hatch.lat,hatch.level,hatch,levels=[0.5,1.5],
                        colors='none', alpha=0,hatches=[density*'/',2*density*'/'], 
                        label='Agreement')
        
        if season_n == 'JFM' or season_n == 'DJF':
            ax2.set_ylabel('{}\nPressure (hPa)'.format(labels[1]), fontsize=25)
        else:
            ax2.set_ylabel('Pressure (hPa)', fontsize=25)
        #ax2.set_xlabel('Latitude', fontsize=15)
        ax2.tick_params(axis='x', which='major', labelsize=20)
        ax2.tick_params(axis='y', which='major', labelsize=20)

        #ax2.set_title('Vertical Moisture Flux', fontsize=25)
    ax13.set_axis_off()
    ax23.set_axis_off()

    fig.subplots_adjust(right=0.9)
    ax_cb11 = fig.add_axes([0.92, 0.55, 0.015, 0.4])
    ax_cb12 = fig.add_axes([0.92, 0.10, 0.015, 0.4])

    cbar11 = plt.colorbar(c11,ax=ax_cb11,fraction=0.8)
    cbar11.ax.tick_params(labelsize=15)
    cbar11.set_label('Moisture Mass Flux Difference ($10^{-5}$ kg $m^{-2}s^{-1}$)',fontsize=20)
    cbar11.ax.yaxis.get_offset_text().set_visible(False)

    sm_ps, tick_locs_ps, tick_labels_ps = colorbar_contour(levels_wh, 'bwr_r', -6e-5, 6e-5, 1/1e-5)

    cbar22 = plt.colorbar(sm_ps, ticks=tick_locs_ps, ax=ax_cb12, fraction=0.8)
    cbar22.ax.set_yticklabels(tick_labels_ps)
    cbar22.ax.tick_params(labelsize=15)
    cbar22.set_label('Moisture Mass Flux ($10^{-5}$ kg $m^{-2}s^{-1}$)',fontsize=20)
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

    
    for ax_cb in [ax_cb11, ax_cb12]:
        ax_cb.spines[['top', 'bottom', 'left', 'right']].set_visible(False)  # Remove axes borders
        ax_cb.set_xticks([])  # Remove x ticks
        ax_cb.set_yticks([])  # Remove y ticks
        ax_cb.tick_params(which='both', bottom=False, top=False, left=False, right=False)  # Remove tick marks
    plt.tight_layout()
    plt.savefig(os.path.join(out_file,name_of_plot+'_{}.pdf'.format(area)), dpi=300)


def moistureflux_plotting(idealized1_dict_waphus_all, idealized2_dict_waphus_all,
                          comp_dict_waphus_all, dict_years_use_comp_all, 
                          dict_years_use_idealized1_all, dict_years_use_idealized2_all,
                            out_file, labels, name):    
    name_of_plot = 'Moisture_mass_flux_change_{}'.format(name)   ### Name of output figure

    seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    regions = {'World':[0,360]}
    
    min_lat = -60.0
    max_lat = 60.0
    bottom_level = 1000
    top_level = 100
    
    for area in list(regions.keys()):   
        lons_regions = regions[area]
        print(lons_regions)
        plot_moistureflux_change(idealized1_dict_waphus_all, idealized2_dict_waphus_all,
                          comp_dict_waphus_all, dict_years_use_comp_all, 
                          dict_years_use_idealized1_all, dict_years_use_idealized2_all,
                                area, bottom_level, top_level, min_lat, max_lat, 
                                seasons, lons_regions, out_file, labels, name_of_plot)

if __name__ == '__main__':
    #########################################################################################################
    #### This code plots the the multimodel mean of the change of the zonal mean of vertical moisture   #####
    #### flux, for the DJF and JJA seasons for both GWLs, as well as the multi-model mean of the        #####
    #### piControl climatology                                                                          #####
    #########################################################################################################
    model_names = ['GFDL-ESM2M', 'NorESM2-LM', 'MIROC-ES2L', 'UKESM1-2', 'EC-Earth3', 'IPSL', 'CNRM']

    out_path = '/path/to/output/'  ### Directory to output the figures and csv files

    var_name_wap = 'wap'
    var_name_hus = 'hus'

    delay = 100   # Years to skip at the beginning of each run
    years_range = 50 # How many years the analysis will span for each run

    comp_dict_waphus_all = {}
    idealized1_dict_waphus_all = {}
    idealized2_dict_waphus_all = {}

    dict_years_use_comp_all = {}
    dict_years_use_idealized1_all = {}
    dict_years_use_idealized2_all = {}


    for model_name in model_names:
        print(model_name)
        #### The path to GWL stabilzation runs should be in the form of 
        #### /path/to/data/MODEL_NAME/TIPMIP/GWL_RUN/
        path_idealized1 = '/path/to/data/{}/TIPMIP/esm-up2p0-gwl2p0/'.format(model_name)
        path_idealized2 = '/path/to/data/{}/TIPMIP/esm-up2p0-gwl4p0/'.format(model_name)
        #### The path to Pre-Industrial Control runs should be in the form of 
        #### /path/to/data/MODEL_NAME/CMIP6Plus/piControl/
        path_hists = '/path/to/data/{}/CMIP6Plus/piControl/'.format(model_name)
        
        #### In each directory with data for instance: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/
        #### The next directory will be the variant number as r1, r2, r3 etc
        #### E.g.: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/r1/
        #### And then the variable name:
        #### E.g.: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/r1/pr/
        #### This Python code, needs the variables: calculated_wap' and 'calculated_hus', to work
        #### where 'calculated_wap' and 'calculated_hus' are the 
        #### remapped 'wap' and 'hus' variables respectively at 1 degree resolution, with terrain masking
            
    
        if path_hists[-1] == '/':
            run_comp = path_hists.split('/')[-2]
        else:
            run_comp = path_hists.split('/')[-1]

        if path_idealized1[-1] == '/':
            run_gwl1 = path_idealized1.split('/')[-2]
            model_name = path_idealized1.split('/')[-4]
        else:
            run_gwl1 = path_idealized1.split('/')[-1]
            model_name = path_idealized1.split('/')[-3]

        if path_idealized1[-1] == '/':
            run_gwl2 = path_idealized2.split('/')[-2]
            model_name = path_idealized2.split('/')[-4]
        else:
            run_gwl2 = path_idealized2.split('/')[-1]
            model_name = path_idealized2.split('/')[-3]

        comp_dict_waphus = {}
        idealized1_dict_waphus = {}
        idealized2_dict_waphus = {}

        dict_years_use_comp = {}
        dict_years_use_idealized1 = {}
        dict_years_use_idealized2 = {}

        lons=360
        lats=180
        
        dict_lonlat = {'lon':lons, 
                    'lat':lats}
            

        #### Check available year ranges
        avail_variants_comp = [f.path for f in os.scandir(path_hists) if f.is_dir() and f.path.split('/')[-1].startswith('r')]

        
        dict_pi_years = {}
        for i, pi_variant in enumerate(avail_variants_comp):
            try:
                files = get_all_files(os.path.join(pi_variant,'hus'))
                files = filter_files_by_var_name(files, 'hus')
                first_year_pi, last_year_pi = get_available_years_range(files)
                dict_pi_years[pi_variant] = [first_year_pi, last_year_pi]
            except:
                continue
        first_year_pi_use, last_year_pi_use = choose_years(dict_pi_years,delay,years_range)
        print('For the piControl the years between {} and {} will be used'.format(first_year_pi_use, last_year_pi_use))
        comp_years = [first_year_pi_use, last_year_pi_use]


        avail_variants_idealized1 = [f.path for f in os.scandir(path_idealized1) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
        #### Check available year ranges
        dict_gwl1_years = {}
        dict_gwl2_years = {}
        
        for i, idealized_variant in enumerate(avail_variants_idealized1):
            try:
                files = get_all_files(os.path.join(idealized_variant,'hus'))
                files = filter_files_by_var_name(files, 'hus')
                first_year_gwl, last_year_gwl = get_available_years_range(files)
                dict_gwl1_years[idealized_variant] = [first_year_gwl, last_year_gwl]
            except:
                continue
        first_year_gwl1_use, last_year_gwl1_use = choose_years(dict_gwl1_years,delay,years_range)
        print('For the GWL the years between {} and {} will be used'.format(first_year_gwl1_use, last_year_gwl1_use))
        idea1_years = [first_year_gwl1_use, last_year_gwl1_use]

        avail_variants_idealized2 = [f.path for f in os.scandir(path_idealized2) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
        #### Check available year ranges
        dict_gwl1_years = {}
        
        for i, idealized_variant in enumerate(avail_variants_idealized2):
            try:
                files = get_all_files(os.path.join(idealized_variant,'hus'))
                files = filter_files_by_var_name(files, 'hus')
                first_year_gwl, last_year_gwl = get_available_years_range(files)
                dict_gwl2_years[idealized_variant] = [first_year_gwl, last_year_gwl]
            except:
                continue
        first_year_gwl2_use, last_year_gwl2_use = choose_years(dict_gwl2_years,delay,years_range)
        print('For the GWL the years between {} and {} will be used'.format(first_year_gwl2_use, last_year_gwl2_use))
        idea2_years = [first_year_gwl2_use, last_year_gwl2_use]



        #################################################################
        #### Reading Monthly wap and hus data 
        #################################################################
        for i, pi_variant in enumerate(avail_variants_comp):
            files = get_all_files(os.path.join(pi_variant,'calculated_{}'.format(var_name_wap)))
            files = filter_files_by_var_name(files, var_name_wap)
            print('Calculating {} comparison variant'.format(pi_variant))
            ncfile_wap = preparing_var_data(files,'wap',first_year_pi_use -1 ,last_year_pi_use +1)

            files = get_all_files(os.path.join(pi_variant,'calculated_{}'.format(var_name_hus)))
            files = filter_files_by_var_name(files, var_name_hus)
            ncfile_hus = preparing_var_data(files,'hus',first_year_pi_use -1 ,last_year_pi_use +1)

            comp_dict_waphus[pi_variant] = ncfile_wap*ncfile_hus 
            dict_years_use_comp[pi_variant] = [first_year_pi_use, last_year_pi_use]

        for i, idealized_variant in enumerate(avail_variants_idealized1):
            files = get_all_files(os.path.join(idealized_variant,'calculated_{}'.format(var_name_wap)))
            files = filter_files_by_var_name(files, var_name_wap)
            print('Calculating {} idealized variant'.format(idealized_variant))
            ncfile_wap = preparing_var_data(files,var_name_wap,first_year_gwl1_use -1,last_year_gwl1_use +1)

            files = get_all_files(os.path.join(idealized_variant,'calculated_{}'.format(var_name_hus)))
            files = filter_files_by_var_name(files, var_name_hus)
            ncfile_hus = preparing_var_data(files,var_name_hus,first_year_gwl1_use -1,last_year_gwl1_use +1)

            idealized1_dict_waphus[idealized_variant] = ncfile_wap * ncfile_hus
            dict_years_use_idealized1[idealized_variant] = [first_year_gwl1_use, last_year_gwl1_use]     

        for i, idealized_variant in enumerate(avail_variants_idealized2):
            files = get_all_files(os.path.join(idealized_variant,'calculated_{}'.format(var_name_wap)))
            files = filter_files_by_var_name(files, var_name_wap)
            print('Calculating {} idealized variant'.format(idealized_variant))
            ncfile_wap = preparing_var_data(files,var_name_wap,first_year_gwl2_use -1,last_year_gwl2_use +1)

            files = get_all_files(os.path.join(idealized_variant,'calculated_{}'.format(var_name_hus)))
            files = filter_files_by_var_name(files, var_name_hus)
            ncfile_hus = preparing_var_data(files,var_name_hus,first_year_gwl2_use -1,last_year_gwl2_use +1)

            idealized2_dict_waphus[idealized_variant] = ncfile_wap * ncfile_hus
            dict_years_use_idealized2[idealized_variant] = [first_year_gwl2_use, last_year_gwl2_use]             

        comp_dict_waphus_all[model_name] = comp_dict_waphus
        idealized1_dict_waphus_all[model_name] = idealized1_dict_waphus
        idealized2_dict_waphus_all[model_name] = idealized2_dict_waphus

        dict_years_use_comp_all[model_name] = dict_years_use_comp
        dict_years_use_idealized1_all[model_name] = dict_years_use_idealized1
        dict_years_use_idealized2_all[model_name] = dict_years_use_idealized2

    ##############################
    ####### Start Plotting #######
    ##############################
    labels = [run_gwl1.split('-')[-1], run_gwl2.split('-')[-1]]
    labels_better = []
    for label in labels:
        if label == 'gwl2p0':
            new_label = '$2^oC$ GWL'
        elif label == 'gwl4p0':
            new_label = '$4^oC$ GWL'
        else:
            new_label = 'GWL'
        labels_better.append(new_label)
    moistureflux_plotting(idealized1_dict_waphus_all, idealized2_dict_waphus_all, 
                          comp_dict_waphus_all, dict_years_use_comp_all, 
                          dict_years_use_idealized1_all, dict_years_use_idealized2_all,
                            out_path, labels_better, 'multimodel_{}_{}_{}'.format(run_comp, labels[0], labels[1]))