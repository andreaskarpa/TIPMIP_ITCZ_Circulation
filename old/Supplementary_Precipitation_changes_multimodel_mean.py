import numpy as np
import xarray as xr
import os
import copy
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import cartopy.crs as ccrs
from tqdm import tqdm
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

def categorize_files_by_year_range(filenames):
    
    categorized_files = []
    ranges = []
    for file in filenames:
        ranges.append(file[-20:-3])
    separate_ranges = list(set(ranges))
    for range_year in separate_ranges:
        categorized_files.append([f for f in filenames if range_year in f])
    return categorized_files

def filter_files_by_var_name(filenames, var_name):
    filenames = [f for f in filenames if f.split("/")[-1][:len(var_name)] == var_name]
    return filenames

def calculating_ensemble_mean(filenames, vars, silent=True):
    """Calculates Ensemble mean of the NetCDF files using the rolling mean method"""
    ensemble_mean_array = np.array([])
    for i in range(len(filenames)):       
        file = filenames[i]
        if not True:
            print('Calculating:', file.split("/")[-1])
        ncfile = xr.open_dataset(file, use_cftime=True)
        ncfile = renaming_dimensions(ncfile)
        ncfile['lat'] = ncfile['lat'].round(3)
        ncfile['lon'] = ncfile['lon'].round(3)
        var_available = ncfile.data_vars.keys()
        var = [f for f in var_available if f in vars][0]
        if i==0:
            ensemble_mean_array = copy.deepcopy(ncfile[var])
        else:
            ensemble_mean_array = ensemble_mean_array*i/(i+1) + ncfile[var]*1/(i+1)
    return ensemble_mean_array

def ensemble_time_series(filenames_all,var,first_year,last_year):
    filtered_files = filter_files_by_year_range(filenames_all, first_year, last_year)
    '''
    print('All the filtered file are the following:')
    print(filtered_files)
    print("")'''
    categorized_files = categorize_files_by_year_range(filtered_files)
    for i,filenames2 in enumerate(tqdm(categorized_files)):  
        ds_file = calculating_ensemble_mean(filenames2, var, silent = True)
        if i==0:
            ensemble_series = ds_file
        else:
            ensemble_series = xr.concat([ensemble_series,ds_file], dim='time').sortby('time') 
    ensemble_series = ensemble_series.where((ensemble_series.time.dt.year>=first_year)&(ensemble_series.time.dt.year<=last_year),drop=True)
    return ensemble_series

def renaming_dimensions(ncfile):
    variables = ncfile.dims
    if 'latitude' in variables:
        ncfile = ncfile.rename({'latitude':'lat'})
    if 'longitude' in variables:
        ncfile = ncfile.rename({'longitude':'lon'})
    if 'nav_lat' in variables:
        ncfile = ncfile.rename({'nav_lat':'lat'})
    if 'nav_lon' in variables:
        ncfile = ncfile.rename({'nav_lon':'lon'})
    if 'time_counter' in variables:
        ncfile = ncfile.rename({'time_counter':'time'})
    if 'valid_time' in variables:
        ncfile = ncfile.rename({'valid_time':'time'})
    if 'plev' in variables:
        ncfile = ncfile.rename({'plev':'level'})
    elif 'pressure' in variables:
        ncfile = ncfile.rename({'pressure':'level'})
    elif 'pressure_level' in variables:
        ncfile = ncfile.rename({'pressure_level':'level'})
    if 'date' in variables: 
        ncfile = ncfile.rename({'date':'time'})
        ncfile['time'] = pd.to_datetime(ncfile['time'].astype(str), format='%Y%m%d')
    if 'expver' and 'number' in ncfile.coords:
        ncfile = ncfile.drop_vars(['expver', 'number'])   
    return ncfile

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

def remapping(array_var,dict_lonlat):
    lons = np.linspace(-180, 180, dict_lonlat['lon'])
    lats = np.linspace(-90, 90, dict_lonlat['lat'])
    print('Remapping data from {}x{} to {}x{} resolution'.format(len(array_var.lon),len(array_var.lat),len(lons),len(lats)))
    array_var = array_var.interp(lon=lons,lat=lats,method='linear')
    return array_var

def preparing_pr_data(files,var_name,first_year,last_year, dict_lonlat):
    dataaaray = ensemble_time_series(files, var_name, first_year, last_year)
    dataaaray = renaming_dimensions(dataaaray)
    dataaaray = convert_lon_to_minus180_180(dataaaray)
    dataaaray = remapping(dataaaray, dict_lonlat)
    
    dataaaray = dataaaray.transpose('time', 'lat', 'lon')
    if dataaaray.max() < 1e-2:
        #print('Multiplying by 86400')
        dataaaray = dataaaray*86400  ### From flux per second, to daily averaged precipitation
    return dataaaray

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

def compute_seasonal_mean(ds, season, first_year, last_year): 
    #### Computes the seasonal mean, in case of DJF season, it takes the December from the previous year #####
    ds = ds.isel(time=ds.time.dt.month.isin(season))

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

        ds_season = ds_season.rename({'season_year': 'time'})

    else:
        ds_season = ds.groupby('time.year').mean(dim='time')

        # Ensure full seasons (optional but good practice)
        counts = ds.groupby('time.year').count(dim='time')
        ds_season = ds_season.where(counts == len(season), drop=True)

        ds_season = ds_season.rename({'year': 'time'})

    # Select years
    ds_season = ds_season.where(
        (ds_season.time >= first_year) & (ds_season.time <= last_year),
        drop=True
    )

    return ds_season

def calculate_model_mean_diff(dict_pr_comp, dict_pr_idealized,season,
                              years_use_comp, years_use_idealized):
    variant_list = []
    for comp_variant in dict_pr_comp.keys():
        P_trop = dict_pr_comp[comp_variant]
        first_year, last_year = years_use_comp[comp_variant]
        p_trop_season = compute_seasonal_mean(P_trop,season, first_year, last_year)
        #p_trop_season = p_trop_season.mean(dim='time')
        p_trop_season = p_trop_season.assign_coords(variant=comp_variant)
        variant_list.append(p_trop_season)
    combined_pr_comp = xr.concat(variant_list, dim="variant")

    variant_list = []
    for idea_variant in dict_pr_idealized.keys():
        P_trop_idea = dict_pr_idealized[idea_variant]
        first_year, last_year = years_use_idealized[idea_variant]
        p_trop_idea_season = compute_seasonal_mean(P_trop_idea,season, first_year, last_year)
        #p_trop_idea_season = p_trop_idea_season.mean(dim='time')
        p_trop_idea_season = p_trop_idea_season.assign_coords(variant=idea_variant)
        variant_list.append(p_trop_idea_season)
    combined_pr_idealized = xr.concat(variant_list, dim="variant")

    std_pr_comp = combined_pr_comp.std(dim=['variant', 'time'])
    combined_pr_comp = combined_pr_comp.mean(dim=['variant', 'time'])
    combined_pr_idealized = combined_pr_idealized.mean(dim=['variant', 'time'])
    
    
    difference_pr = combined_pr_idealized - combined_pr_comp
    snr = np.abs(difference_pr)/std_pr_comp
    snr_bool = xr.where(snr>1,1,0)

    return difference_pr, snr_bool


def multi_model(dict_pr_comp_all, dict_pr_idealized_all, season, 
                years_use_comp_all, years_use_idealized_all):
    model_list=[]
    snr_list = []
    for model_name in dict_pr_comp_all.keys():
        dict_pr_comp = dict_pr_comp_all[model_name]
        dict_pr_idealized = dict_pr_idealized_all[model_name]
        years_use_comp = years_use_comp_all[model_name]
        years_use_idealized = years_use_idealized_all[model_name]
        mean_model_pr_difference, snr = calculate_model_mean_diff(dict_pr_comp, dict_pr_idealized,season,
                                                             years_use_comp, years_use_idealized)
        mean_model_pr_difference = mean_model_pr_difference.assign_coords(model=model_name)
        snr = snr.assign_coords(model=model_name)
        model_list.append(mean_model_pr_difference)
        snr_list.append(snr)
    combined_pr_model_diff = xr.concat(model_list, dim="model")
    combined_snr_model = xr.concat(snr_list, dim="model")
    return combined_pr_model_diff, combined_snr_model

def plot_changes(dict_pr_comp_all, dict_pr_idealized1_all, dict_pr_idealized2_all, seasons, years_use_comp_all, 
                 years_use_idealized1_all, years_use_idealized2_all, labels, out_path, out_name):
    fig, axs = plt.subplots(
        2, len(seasons.keys()), figsize=(17, 9), constrained_layout=True, 
        subplot_kw=dict(projection=ccrs.Mollweide())
    )
    for j, season_n in enumerate(seasons.keys()):
        season = seasons[season_n]
        change_multimodel1, snr1 = multi_model(dict_pr_comp_all, dict_pr_idealized1_all, season, years_use_comp_all, years_use_idealized1_all,)
        change_multimodel2, snr2 = multi_model(dict_pr_idealized1_all, dict_pr_idealized2_all, season, years_use_idealized1_all, years_use_idealized2_all,)
        
        ### Hatching where at least 5/7 models agree on the sign of change and at least 4/7 are robust (SNR>1) ###
        hatch_model_agreement_sign_1, mean_change_1 = get_hatching_mask(change_multimodel1, 5, agreement=True)
        hatch_model_agreement_robust_1, _ = get_hatching_mask(snr1, 4, agreement=True)
        hatch_1 = hatch_model_agreement_sign_1 * hatch_model_agreement_robust_1

        levels = np.arange(-1.6,1.8,0.2)
        c1 = axs[0,j].contourf(mean_change_1.lon.values,mean_change_1.lat.values, mean_change_1,
                                 vmin=-1.6,vmax=1.6,levels=levels,cmap = 'BrBG', alpha = 0.85,
                                 transform = ccrs.PlateCarree(),extend='both')
        axs[0,j].coastlines()
        density=2
        cs2_r = axs[0,j].contourf(hatch_1.lon,hatch_1.lat,hatch_1,levels=[0.5,1.5],colors='none', alpha=0,hatches=[2*density*'/',4*density*'/'], transform = ccrs.PlateCarree())
        axs[0,j].set_title(season_n, fontsize=25)

        hatch_model_agreement_sign_2, mean_change_2 = get_hatching_mask(change_multimodel2, 5, agreement=True)
        hatch_model_agreement_robust_2, _ = get_hatching_mask(snr2, 4, agreement=True)
        hatch_2 = hatch_model_agreement_sign_2 * hatch_model_agreement_robust_2

        c1 = axs[1,j].contourf(mean_change_2.lon.values,mean_change_2.lat.values, mean_change_2,
                                 vmin=-1.6,vmax=1.6,levels=levels,cmap = 'BrBG', alpha = 0.85,
                                 transform = ccrs.PlateCarree(),extend='both')
        axs[1,j].coastlines()
        axs[1,j].contourf(hatch_2.lon,hatch_2.lat,hatch_2,levels=[0.5,1.5],colors='none', alpha=0,hatches=[2*density*'/',4*density*'/'], transform = ccrs.PlateCarree())
    axs[0,0].text(-0.08, 0.5, labels[0],
                transform=axs[0,0].transAxes,
                rotation=90, va='center', ha='center', fontsize=25)

    axs[1,0].text(-0.08, 0.5, labels[1],
                transform=axs[1,0].transAxes,
                rotation=90, va='center', ha='center', fontsize=25)

    cbar = fig.colorbar(
        c1,
        ax=axs,
        orientation='horizontal',
        fraction=0.05,
        pad=0.08
    )
    cbar.set_label('Precipitation difference (mm $d^{-1}$)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.suptitle("", fontsize=20)
    plt.savefig(os.path.join(out_path,out_name))


if __name__ == '__main__':
    #########################################################################################################
    #### This code plots the the multimodel mean of the change of the precipitation for both GWLs and   #####
    #### for the DJF and JJA seasons                                                                    #####
    #########################################################################################################
    out_path = '/path/to/output/'  ### Directory to output the figure
    model_names = ['GFDL-ESM2M', 'NorESM2-LM', 'MIROC-ES2L', 'UKESM1-2','EC-Earth3','IPSL','CNRM']
    name_out =  "Precipitation_changes_multi_model_mean.pdf"  ### Name of output figure

    seasons_dict_only = {'DJF':[12,1,2], 'JJA':[6,7,8]}           ### Seasons to plot

    
    dict_pr_comp_all = {}
    dict_pr_idealized1_all = {}
    dict_pr_idealized2_all = {}

    dict_years_use_comp_all = {}
    dict_years_use_idealized1_all = {}
    dict_years_use_idealized2_all = {}

    lons=360
    lats=180
    
    dict_lonlat = {'lon':lons, 
                'lat':lats}

    delay = 100
    years_range = 50

    for model_name in model_names:
        #### The path to Pre-Industrial Control runs should be in the form of 
        #### /path/to/data/MODEL_NAME/CMIP6Plus/piControl/
        path_pi = '/path/to/data/{}/CMIP6Plus/piControl/'.format(model_name)
        #### The path to GWL stabilzation runs should be in the form of 
        #### /path/to/data/MODEL_NAME/TIPMIP/GWL_RUN/
        path_idealized1 = '/path/to/data/{}/TIPMIP/esm-up2p0-gwl2p0/'.format(model_name)
        path_idealized2 = '/path/to/data/{}/TIPMIP/esm-up2p0-gwl4p0/'.format(model_name)
        
        #### In each directory with data for instance: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/
        #### The next directory will be the variant number as r1, r2, r3 etc
        #### E.g.: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/r1/
        #### And then the variable name:
        #### E.g.: /path/to/data/EC-Earth3/TIPMIP/esm-up2p0-gwl2p0/r1/pr/
        #### This Python code, needs the variables: 'pr' to work
        #### where 'pr' is the raw pr data from the run
        
        
        if path_pi[-1] == '/':
            run = path_pi.split('/')[-2]
            name = path_pi.split('/')[-4]
        else:
            run = path_pi.split('/')[-1]
            name = path_pi.split('/')[-3]

        if path_idealized1[-1] == '/':
            run_idealized1 = path_idealized1.split('/')[-2]
        else:
            run_idealized1 = path_idealized1.split('/')[-1]

        if path_idealized2[-1] == '/':
            run_idealized2 = path_idealized2.split('/')[-2]
        else:
            run_idealized2 = path_idealized2.split('/')[-1]


        #### Check available year ranges
        dict_pi_years = {}
        avail_variants = [f.path for f in os.scandir(path_pi) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
        for i, pi_variant in enumerate(avail_variants):
            try:
                files = get_all_files(os.path.join(path_pi,pi_variant,'pr'))
                files = filter_files_by_var_name(files, 'pr')
                first_year_pi, last_year_pi = get_available_years_range(files)
                dict_pi_years[pi_variant] = [first_year_pi, last_year_pi]
            except:
                continue
        first_year_use_pi, last_year_use_pi = choose_years(dict_pi_years,delay,years_range)
        print('For the piControl the years between {} and {} will be used'.format(first_year_use_pi, last_year_use_pi))


        dict_idealized1_years = {}
        avail_variants = [f.path for f in os.scandir(path_idealized1) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
        for i, idealized_variant in enumerate(avail_variants):
            try:
                files = get_all_files(os.path.join(path_idealized1, idealized_variant,'pr'))
                files = filter_files_by_var_name(files, 'pr')
                first_year_idealized1, last_year_idealized1 = get_available_years_range(files)
                dict_idealized1_years[idealized_variant] = [first_year_idealized1, last_year_idealized1]
            except:
                continue
        first_year_use_idealized1, last_year_use_idealized1 = choose_years(dict_idealized1_years,delay,years_range)
        print('For the {} the years between {} and {} will be used'.format(run_idealized1,first_year_use_idealized1, last_year_use_idealized1))

        dict_idealized2_years = {}
        avail_variants = [f.path for f in os.scandir(path_idealized2) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
        for i, idealized_variant in enumerate(avail_variants):
            try:
                files = get_all_files(os.path.join(path_idealized2, idealized_variant,'pr'))
                files = filter_files_by_var_name(files, 'pr')
                first_year_idealized2, last_year_idealized2 = get_available_years_range(files)
                dict_idealized2_years[idealized_variant] = [first_year_idealized2, last_year_idealized2]
            except:
                continue
        first_year_use_idealized2, last_year_use_idealized2 = choose_years(dict_idealized2_years,delay,years_range)
        print('For the {} the years between {} and {} will be used'.format(run_idealized2,first_year_use_idealized2, last_year_use_idealized2))

            
        ###########################
        ####### Comparison ########
        ###########################

        avail_variants = [f.path for f in os.scandir(path_pi) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
        dict_pr_comp = {}
        dict_years_use_comp = {}
        for i, comp_variant in enumerate(avail_variants):
            files_pr = get_all_files(os.path.join(comp_variant, 'pr'))
            files_pr = filter_files_by_var_name(files_pr, 'pr')

            print('Calculating {} idealized variant'.format(comp_variant))
            ncfile_pr = preparing_pr_data(files_pr,'pr',first_year_use_pi-1, last_year_use_pi+1, dict_lonlat)

            # restrict domain
            P_trop = ncfile_pr.sel(lat=slice(-60, 60))
            dict_pr_comp[comp_variant] = P_trop
            dict_years_use_comp[comp_variant] = [first_year_use_pi, last_year_use_pi]
        
        ###############################
        ########## Idealized 1 ##########
        ###############################
        avail_variants = [f.path for f in os.scandir(path_idealized1) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
        dict_pr_idealized1 = {}
        dict_years_use_idealized1 = {}
        for i, idealized_variant in enumerate(avail_variants):
            files_pr = get_all_files(os.path.join(idealized_variant, 'pr'))
            files_pr = filter_files_by_var_name(files_pr, 'pr')

            #### Check available year ranges
            print('Calculating {} idealized variant'.format(idealized_variant))

            ncfile_pr_idea = preparing_pr_data(files_pr,'pr',first_year_use_idealized1-1, last_year_use_idealized1+1, dict_lonlat)

            # restrict domain
            P_trop_idea = ncfile_pr_idea.sel(lat=slice(-60, 60))
            dict_pr_idealized1[idealized_variant] = P_trop_idea
            dict_years_use_idealized1[idealized_variant] = [first_year_use_idealized1, last_year_use_idealized1]


        ###############################
        ########## Idealized 2 ##########
        ###############################
        avail_variants = [f.path for f in os.scandir(path_idealized2) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
        dict_pr_idealized2 = {}
        dict_years_use_idealized2 = {}
        for i, idealized_variant in enumerate(avail_variants):
            files_pr = get_all_files(os.path.join(idealized_variant, 'pr'))
            files_pr = filter_files_by_var_name(files_pr, 'pr')

            #### Check available year ranges
            print('Calculating {} idealized variant'.format(idealized_variant))

            ncfile_pr_idea = preparing_pr_data(files_pr,'pr',first_year_use_idealized2-1, last_year_use_idealized2+1, dict_lonlat)

            # restrict domain
            P_trop_idea = ncfile_pr_idea.sel(lat=slice(-60, 60))
            dict_pr_idealized2[idealized_variant] = P_trop_idea
            dict_years_use_idealized2[idealized_variant] = [first_year_use_idealized2, last_year_use_idealized2]

        dict_pr_comp_all[model_name] = dict_pr_comp
        dict_pr_idealized1_all[model_name] = dict_pr_idealized1
        dict_pr_idealized2_all[model_name] = dict_pr_idealized2

        dict_years_use_comp_all[model_name] = dict_years_use_comp
        dict_years_use_idealized1_all[model_name] = dict_years_use_idealized1
        dict_years_use_idealized2_all[model_name] = dict_years_use_idealized2

    ##############################
    ####### Start Plotting #######
    ##############################
    labels = [run_idealized1.split('-')[-1], run_idealized2.split('-')[-1]]
    labels_better = []
    for label in labels:
        if label == 'gwl2p0':
            new_label = '$2^oC$ GWL'
        elif label == 'gwl4p0':
            new_label = '$4^oC$ GWL'
        else:
            new_label = 'GWL'
        labels_better.append(new_label)

    plot_changes(dict_pr_comp_all, dict_pr_idealized1_all, dict_pr_idealized2_all, seasons_dict_only, 
                 dict_years_use_comp_all, dict_years_use_idealized1_all, dict_years_use_idealized2_all,labels_better, out_path, name_out)
