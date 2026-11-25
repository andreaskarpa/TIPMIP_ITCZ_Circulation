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
import warnings
import matplotlib as mpl
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
        ds_season = ds_season.where(counts == len(season), drop=True)
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

def pvelocity_plotting(idealized_dict_wap_all,comp_dict_wap_all, dict_years_use_comp_all, dict_years_use_idealized_all, 
                       run_gwl, out_file):    
    name_of_plot = 'MMA_omega_change_regions_pi_{}'.format(run_gwl) #### Name of output plot
    
    regions = {'America':[-90,-40], 'Atlantic':[-40,-10],
               'Africa':[-10,40],'Indian Ocean':[40,120],
               'Western Pacific':[120,170], 'Central Pacific':[170,230], 
               'Eastern Pacific':[230,270]}
    seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    
    min_lat = -60.0
    max_lat = 60.0
    bottom_level = 1000
    top_level = 100
    fig, axes = plt.subplots(7,3, figsize=(25,30),
                                    gridspec_kw={'width_ratios': [15,15,3]},  # Ensure equal heights
                                    )
    
    for k, area in enumerate(list(regions.keys())): 
        ax11, ax12, ax_empty = axes[k]  

        lons_region = regions[area]
        for season_n in list(seasons.keys()):
            if season_n == 'DJF':
                ax1 = ax11
            else:
                ax1 = ax12
            print(area, season_n)
            season = seasons[season_n]
            ### wap
            ####### Comparison piControl model
            change_wap_multimodel, multimodel_wap_comp, snr_wap = multi_model(comp_dict_wap_all, idealized_dict_wap_all, 
                                                                 dict_years_use_comp_all, dict_years_use_idealized_all,
                                                                 season, 'wap', min_lat, max_lat, lons_region)

            #####################
            #####################
            #####################
            levels2 = np.arange(-0.01,0.012,0.002)

            levels_simple = [-0.03,-0.01,0,0.01,0.03]

            
            #### The normal
            plot_in_contours = multimodel_wap_comp
            c12 = ax1.contourf(change_wap_multimodel.lat, change_wap_multimodel.level, change_wap_multimodel.mean(dim='model').values,
                            vmin=-0.01,vmax=0.01,cmap='BrBG_r',levels=levels2,extend='both', alpha=0.7)
            ax1.set_ylim(bottom_level,top_level)   #'RdBu'
            
            ax1.contour(plot_in_contours.lat, plot_in_contours.level,plot_in_contours.mean(dim='model').values,
                        vmin=-0.04,vmax=0.04,colors='black',levels=levels_simple, linewidths=4, linestyles='-')
            ax1.contour(plot_in_contours.lat, plot_in_contours.level,plot_in_contours.mean(dim='model').values,
                        vmin=-0.04,vmax=0.04,cmap='bwr_r',levels=levels_simple, linewidths=3, linestyles='-')   

            hatch_model_agreement_sign, mean_change = get_hatching_mask(change_wap_multimodel, 5, agreement=True)
            hatch_model_agreement_robust, _ = get_hatching_mask(snr_wap, 4, agreement=True)
            hatch = hatch_model_agreement_sign * hatch_model_agreement_robust

            density=1
            ax1.contourf(hatch.lat,hatch.level,hatch,levels=[0.5,1.5],colors='none', alpha=0,hatches=[density*'/',2*density*'/'], label='Strengthening')
                        

            if season_n == 'DJF':
                ax1.set_ylabel('{}\nPressure (hPa)'.format(area), fontsize=25)
            else:
                ax1.set_ylabel('Pressure (hPa)', fontsize=25)

            
            ax1.tick_params(axis='x', which='major', labelsize=20)
            ax1.tick_params(axis='y', which='major', labelsize=20)
            if k==0:
                ax1.set_title('{}'.format(season_n), fontsize=30)
            ax_empty.set_axis_off()

    fig.subplots_adjust(right=0.9)
    ax_cb11 = fig.add_axes([0.92, 0.55, 0.015, 0.45])
    ax_cb12 = fig.add_axes([0.92, 0.10, 0.015, 0.45])
    cbar11 = plt.colorbar(c12,ax=ax_cb11,fraction=1.5)
    cbar11.ax.tick_params(labelsize=20)
    cbar11.set_label('Pressure Velocity Difference (Pa $s^{-1}$)',fontsize=25)
    cbar11.ax.yaxis.get_offset_text().set_visible(False)
    
    sm, tick_locs, tick_labels = colorbar_contour(levels_simple, "bwr_r", -0.04, 0.04, 1)
    cbar12 = plt.colorbar(sm, ticks=tick_locs, ax=ax_cb12, fraction=1.5)
    cbar12.ax.set_yticklabels(tick_labels)


    cbar12.ax.tick_params(labelsize=20)
    cbar12.set_label('Pressure Velocity (Pa $s^{-1}$)',fontsize=25)
    cbar12.ax.yaxis.get_offset_text().set_visible(False)
    for ax_cb in [ax_cb11, ax_cb12]:
        ax_cb.spines[['top', 'bottom', 'left', 'right']].set_visible(False)  # Remove axes borders
        ax_cb.set_xticks([])  # Remove x ticks
        ax_cb.set_yticks([])  # Remove y ticks
        ax_cb.tick_params(which='both', bottom=False, top=False, left=False, right=False)  # Remove tick marks

    plt.tight_layout()
    plt.savefig(os.path.join(out_file,name_of_plot+'.pdf'), dpi=300)


            
if __name__ == '__main__':
    #########################################################################################################
    #### This code plots the the multimodel mean of the change of the zonal mean of pressure velocity   #####
    #### for all longitudinal regions for the DJF and JJA seasons for a specific GWL                    #####
    #########################################################################################################
    model_names = ['GFDL-ESM2M', 'NorESM2-LM', 'MIROC-ES2L', 'UKESM1-2', 'EC-Earth3', 'IPSL', 'CNRM']

    out_path = '/path/to/output/'  ### Directory to output the figures

    var_name_wap = 'wap'

    gwl_runs_do = ['esm-up2p0-gwl2p0', 'esm-up2p0-gwl4p0']   ### Which GWL stabilzation runs to compare to piControl runs
    
    delay = 100   # Years to skip at the beginning of each run
    years_range = 50 # How many years the analysis will span for each run

    for gwl_run in gwl_runs_do:
        comp_dict_wap_all = {}

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
            if gwl_run == 'esm-up2p0-gwl4p0':
                path_hists = '/path/to/data/{}/TIPMIP/{}/'.format(model_name, 'esm-up2p0-gwl2p0')  #### To compare the GWL4 to GWL2
            else:
                path_hists = '/path/to/data/{}/CMIP6Plus/piControl/'.format(model_name)            #### To compare the GWL2 to piControl
            path_hists = '/path/to/data/{}/CMIP6Plus/piControl/'.format(model_name)
            
            #### In each directory with data for instance: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/
            #### The next directory will be the variant number as r1, r2, r3 etc
            #### E.g.: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/r1/
            #### And then the variable name:
            #### E.g.: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/r1/pr/
            #### This Python code, needs the variables:'calculated_wap' to work
            #### wher'calculated_wap'  is the remapped 'wap' variable at 1 degree resolution, with terrain masking
        
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

            comp_dict_wap = {}

            idealized_dict_wap = {}

            dict_years_use_comp = {}
            dict_years_use_idealized = {}

            lons=360
            lats=180
            
            dict_lonlat = {'lon':lons, 
                        'lat':lats}
                

            #### Check available year ranges
            avail_variants_comp = [f.path for f in os.scandir(path_hists) if f.is_dir() and f.path.split('/')[-1].startswith('r')]

            
            dict_pi_years = {}
            for i, pi_variant in enumerate(avail_variants_comp):
                files = get_all_files(os.path.join(path_hists,pi_variant,'wap'))
                files = filter_files_by_var_name(files, 'wap')
                first_year_pi, last_year_pi = get_available_years_range(files)
                dict_pi_years[pi_variant] = [first_year_pi, last_year_pi]
    
            first_year_pi_use, last_year_pi_use = choose_years(dict_pi_years,delay,years_range)
            print('For the piControl the years between {} and {} will be used'.format(first_year_pi_use, last_year_pi_use))
            comp_years = [first_year_pi_use, last_year_pi_use]


            avail_variants_idealized = [f.path for f in os.scandir(path_idealized) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
            #### Check available year ranges
            dict_gwl_years = {}
            
            for i, idealized_variant in enumerate(avail_variants_idealized):
                files = get_all_files(os.path.join(path_idealized, idealized_variant,'wap'))
                files = filter_files_by_var_name(files, 'wap')
                first_year_gwl, last_year_gwl = get_available_years_range(files)
                dict_gwl_years[idealized_variant] = [first_year_gwl, last_year_gwl]

            first_year_gwl_use, last_year_gwl_use = choose_years(dict_gwl_years,delay,years_range)
            print('For the GWL the years between {} and {} will be used'.format(first_year_gwl_use, last_year_gwl_use))
            idea_years = [first_year_gwl_use, last_year_gwl_use]

            #################################################################
            #### Reading Monthly wap data 
            #################################################################
            for i, pi_variant in enumerate(avail_variants_comp):
                files = get_all_files(os.path.join(path_hists,pi_variant,'calculated_{}'.format(var_name_wap)))
                files = filter_files_by_var_name(files, var_name_wap)
                print('Calculating {} historical variant'.format(pi_variant))
                ncfile = preparing_var_data(files,var_name_wap,first_year_pi_use - 1,last_year_pi_use + 1)

                comp_dict_wap[pi_variant] = ncfile   
                dict_years_use_comp[pi_variant] = [first_year_pi_use, last_year_pi_use]
            print(comp_dict_wap.keys())

            for i, idealized_variant in enumerate(avail_variants_idealized):
                files = get_all_files(os.path.join(path_hists,idealized_variant,'calculated_{}'.format(var_name_wap)))
                files = filter_files_by_var_name(files, var_name_wap)
                print('Calculating {} idealized variant'.format(idealized_variant))
                ncfile = preparing_var_data(files,var_name_wap,first_year_gwl_use - 1,last_year_gwl_use + 1)

                idealized_dict_wap[idealized_variant] = ncfile  
                dict_years_use_idealized[idealized_variant] = [first_year_gwl_use, last_year_gwl_use]   
            print(idealized_dict_wap.keys())

            comp_dict_wap_all[model_name] = comp_dict_wap
            idealized_dict_wap_all[model_name] = idealized_dict_wap

            dict_years_use_comp_all[model_name] = dict_years_use_comp
            dict_years_use_idealized_all[model_name] = dict_years_use_idealized

        ##############################
        ####### Start Plotting #######
        ##############################
        pvelocity_plotting(idealized_dict_wap_all, comp_dict_wap_all, dict_years_use_comp_all, dict_years_use_idealized_all,
                        run_gwl, out_path)
