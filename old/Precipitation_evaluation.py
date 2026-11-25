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

def add_longitudes_circular(da):
    #print('Extending longitudes for interpolation if needed')
    lons = da['lon'].values
    if 360 not in lons:
        #print('Extending near 360')
        # Extract the data corresponding to the closest at lon=0
        data_at_near_zero_lon = da.sel(lon=0, method='nearest')    
        # Create a new DataArray with lon=360, copying the data from lon=0
        new_data_at_near360 = data_at_near_zero_lon.expand_dims({'lon': [360+np.min(lons)]}, axis=-1)
        da = xr.concat([da , new_data_at_near360], dim='lon')
        da = da.sortby('lon')
    if 0 not in lons:
        #print('Extending near 0')
        # Extract the data corresponding to the closest at lon=360
        data_at_near_360_lon = da.sel(lon=360, method='nearest')
        new_data_at_near0 = data_at_near_360_lon.expand_dims({'lon': [np.max(lons)-360]}, axis=-1)
        # Combine the new data with the original DataArray
        da = xr.concat([new_data_at_near0 ,da], dim='lon')
        
        # Sort the longitude dimension if needed (optional)
        da = da.sortby('lon')
    return da

def add_poles_latitudes(da):
    #print('Adding latitudes of +-90 for interpolation if needed')
    lats = da['lat'].values
    if 90 not in lats:
        #print('Adding 90 latitude')
        # Extract the data corresponding to the closest at lon=0
        data_at_near_north_pole = da.sel(lat=90, method='nearest')    
        # Create a new DataArray with lon=360, copying the data from lon=0
        new_data_at_north_pole = data_at_near_north_pole.expand_dims({'lat': [90]}, axis=-1)
        da = xr.concat([da , new_data_at_north_pole], dim='lat')
        da = da.sortby('lat')
    if -90 not in lats:
        #print('Adding -90 latitude')
        # Extract the data corresponding to the closest at lon=360
        data_at_near_south_pole = da.sel(lat=-90, method='nearest')
        new_data_at_near_south_pole = data_at_near_south_pole.expand_dims({'lat': [-90]}, axis=-1)
        # Combine the new data with the original DataArray
        da = xr.concat([new_data_at_near_south_pole ,da], dim='lat')
        
        # Sort the longitude dimension if needed (optional)
        da = da.sortby('lat')
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

def remapping(array_var,dict_lonlat):
    lons = np.linspace(0, 360, dict_lonlat['lon'])
    lats = np.linspace(-90, 90, dict_lonlat['lat'])
    print('Remapping data from {}x{} to {}x{} resolution'.format(len(array_var.lon),len(array_var.lat),len(lons),len(lats)))
    array_var = array_var.interp(lon=lons,lat=lats,method='linear')
    return array_var

def preparing_pr_data(files,var_name,first_year,last_year, dict_lonlat, do_regrid, min_lon = -180):
    dataaaray = ensemble_time_series(files, var_name, first_year, last_year)
    dataaaray = renaming_dimensions(dataaaray)
    dataaaray = convert_lon_to_360(dataaaray)
    if do_regrid:
        dataaaray = add_longitudes_circular(dataaaray)
        dataaaray = add_poles_latitudes(dataaaray)    
        dataaaray = remapping(dataaaray, dict_lonlat)
    if min_lon == -180:
        dataaaray = convert_lon_to_minus180_180(dataaaray)
    
    dataaaray = dataaaray.transpose('time', 'lat', 'lon')
    if dataaaray.max() < 1e-2:
        #print('Multiplying by 86400')
        dataaaray = dataaaray*86400  ### From flux per second, to daily averaged precipitation
    return dataaaray

def preparing_pr_data_direct(file,var_name,first_year,last_year, dict_lonlat, do_regrid, min_lon = -180):
    dataaaray = xr.open_dataset(file)
    dataaaray = dataaaray[var_name]
    dataaaray = renaming_dimensions(dataaaray)
    dataaaray = dataaaray.where((dataaaray.time.dt.year>=first_year)&(dataaaray.time.dt.year<=last_year),drop=True)
    dataaaray = convert_lon_to_360(dataaaray)
    if not np.all(np.diff(dataaaray['lat'].values) > 0):
        dataaaray = dataaaray.sortby('lat')
    if do_regrid:
        dataaaray = add_longitudes_circular(dataaaray)
        dataaaray = add_poles_latitudes(dataaaray)
        
        dataaaray = remapping(dataaaray, dict_lonlat)
    if min_lon == -180:
        dataaaray = convert_lon_to_minus180_180(dataaaray)
    dataaaray = dataaaray.transpose('time', 'lat', 'lon')
    if dataaaray.max() < 1:
        #print('Multiplying by 86400')
        dataaaray = dataaaray*1000  ### From m/day, to mm/day
    return dataaaray

def compute_seasonal_mean(ds, season, first_year, last_year): 
    ds = ds.isel(time=ds.time.dt.month.isin(season))

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
    ds_season = ds_season.mean(dim='time')

    return ds_season

def calc_variant(dict_pr, season, first_year, last_year):
    variant_list = []
    for comp_variant in dict_pr.keys():
        pr_variant = dict_pr[comp_variant]
        pr_variant_season_mean = compute_seasonal_mean(pr_variant, season, first_year, last_year)
        pr_variant_season_mean = pr_variant_season_mean.assign_coords(variant=comp_variant)
        variant_list.append(pr_variant_season_mean)
    combined_pr = xr.concat(variant_list, dim="variant")
    return combined_pr

def remove_nans(x, y):
    """
    Removes NaN values from both x and y, ensuring only non-NaN values remain.
    
    Parameters:
    x (np.ndarray): First input array.
    y (np.ndarray): Second input array.
    
    Returns:
    np.ndarray, np.ndarray: Filtered arrays with only non-NaN values.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)  # Keep only elements that are non-NaN in both arrays
    return x[mask], y[mask]

def calculate_mspaef_all_variants(comparison_da, model_da):
    '''
    Calculated the metric MSPAEF
    Parameters
    - comparison_da: 2D xr.datarray, the dataarray of comparison dataset.
    - model_da: 3D xr.datarray, the dataarray of model dataset with all variants in extra dimension.
    
    Returns:
    - list_variant_mspaef : list of MSPAEF metric values for all variants
    '''
    list_variant_mspaef = []

    pr_comp = comparison_da.sel(lat=slice(-20, 20))
    x = pr_comp.values.flatten(order='F')
    for variant_n in model_da.variant.values:
        variant_pr = model_da.sel(variant = variant_n, lat=slice(-20, 20))
        y = variant_pr.values.flatten(order='F')
        x, y = remove_nans(x,y)

        std_ratio = np.std(y) / np.std(x)   ### Ratio of standard deviation values of comparison vs model datasets
        corr = np.corrcoef(x, y)[0, 1] #### Pearson Correlation coefficient, for use in SPAEF, MSPAEF and WSPAEF metrics
    
        q75, q25 = np.percentile(x, [75 ,25])
        iqr_x = q75 - q25

        nmse = np.sqrt(np.sum((x-y)**2)/(len(x.flatten())))/iqr_x  #### Normalized (over std of comparison dataset) Root Mean Square Error of original data  

        mspaef = 1 - np.sqrt((nmse)**2 + (np.sqrt((std_ratio**2 - 1)**2 + (1/std_ratio**2 - 1)**2))**2 + ((np.abs(np.mean(x)-np.mean(y))/iqr_x))**2 + (1-corr)**2)/np.sqrt(4)
        list_variant_mspaef.append(mspaef)
    return list_variant_mspaef

def tpai_index(data_model):
    #### Calculates the Tropical Precipitation Assymetry Index (TPAI) #####
    NH_tropical_precipitation = data_model.where((data_model.lat<=20)&(data_model.lat>=0), drop=True)
    SH_tropical_precipitation = data_model.where((data_model.lat<=0)&(data_model.lat>=-20), drop=True)
    Tropical_precipitation = data_model.where((data_model.lat<=20)&(data_model.lat>=-20), drop=True)
    
    weights = np.cos(np.deg2rad(NH_tropical_precipitation.lat))
    weights.name = "weights"
    NH_mean_tropical_precipitation = NH_tropical_precipitation.weighted(weights)

    weights = np.cos(np.deg2rad(SH_tropical_precipitation.lat))
    weights.name = "weights"
    SH_mean_tropical_precipitation = SH_tropical_precipitation.weighted(weights)

    weights = np.cos(np.deg2rad(Tropical_precipitation.lat))
    weights.name = "weights"
    Mean_tropical_precipitation = Tropical_precipitation.weighted(weights)
    
    NH_mean_tropical_precipitation = NH_mean_tropical_precipitation.mean(dim=['lat','lon'])
    SH_mean_tropical_precipitation = SH_mean_tropical_precipitation.mean(dim=['lat','lon'])
    Mean_tropical_precipitation = Mean_tropical_precipitation.mean(dim=['lat','lon'])
    
    tpai = (NH_mean_tropical_precipitation - SH_mean_tropical_precipitation)/Mean_tropical_precipitation
    
    return tpai


def epi_index(data_model):
    #### Calculates the Equatorial Precipitation Index (EPI) #####
    Equatorial_precipitation = data_model.where((data_model.lat<=2)&(data_model.lat>=-2), drop=True)
    Tropical_precipitation = data_model.where((data_model.lat<=20)&(data_model.lat>=-20), drop=True)
    

    weights = np.cos(np.deg2rad(Equatorial_precipitation.lat))
    weights.name = "weights"
    Mean_equatorial_precipitation = Equatorial_precipitation.weighted(weights)

    weights = np.cos(np.deg2rad(Tropical_precipitation.lat))
    weights.name = "weights"
    Mean_tropical_precipitation = Tropical_precipitation.weighted(weights)
    
    Mean_equatorial_precipitation = Mean_equatorial_precipitation.mean(dim=['lat','lon'])
    Mean_tropical_precipitation = Mean_tropical_precipitation.mean(dim=['lat','lon'])
    
    epi = Mean_equatorial_precipitation/Mean_tropical_precipitation
    
    return epi   

def plot_anomalies(dict_pr_comp_all, pr_era5, seasons_dict, first_year, last_year, out_path, out_name, naming_models):
    for season_n in seasons_dict.keys():
        print(season_n)
        season = seasons_dict[season_n]
        fig, axs = plt.subplots(
            3, 3, figsize=(17, 9), constrained_layout=True, 
            subplot_kw=dict(projection=ccrs.Mollweide())
        )
        axs_flatten = axs.flatten()

        mean_pr_era5 = compute_seasonal_mean(pr_era5, season, first_year, last_year)
        levels = np.arange(-5, 5.5, 0.5)
        for j, model_name in enumerate(dict_pr_comp_all.keys()):
            ax = axs_flatten[j]
            dict_pr_model = dict_pr_comp_all[model_name]
            pr_model = calc_variant(dict_pr_model, season, first_year, last_year)
            mean_pr_model = pr_model.mean(dim='variant')
            difference = mean_pr_model - mean_pr_era5
            c1 = ax.contourf(difference.lon.values, difference.lat.values, difference.values, levels=levels, cmap = 'BrBG',
                        transform = ccrs.PlateCarree(), extend='both')
            ax.coastlines()

            ax.set_title(naming_models[model_name], fontsize=25)
            
        for j in range(len(dict_pr_comp_all.keys()), len(axs_flatten)):
            ax = axs_flatten[j]
            ax.axis('off')

        cbar = fig.colorbar(
            c1,
            ax=axs,
            orientation='vertical',
            fraction=0.05,
            pad=0.08
        )
        cbar.set_label('Precipitation difference (mm $d^{-1}$)', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        plt.suptitle("", fontsize=20)
        plt.savefig(os.path.join(out_path,out_name.format(season_n)))

def plot_mspaef_heatmap(dict_pr_comp_all, pr_era5, seasons_dict, first_year, last_year, out_path, out_name, naming_models):
    list_values_all_models = []
    model_name_list = []
    for j, model_name in enumerate(dict_pr_comp_all.keys()):
        print(model_name)
        dict_pr_model = dict_pr_comp_all[model_name]
        model_name_list.append(naming_models[model_name])
        list_values = []
        seasons_name_list = []
        for season_n in seasons_dict.keys():
            print(" ",season_n)
            seasons_name_list.append(season_n)
            season = seasons_dict[season_n]
            mean_pr_era5 = compute_seasonal_mean(pr_era5, season, first_year, last_year)
            pr_model = calc_variant(dict_pr_model, season, first_year, last_year)
            list_mspaef_model = calculate_mspaef_all_variants(mean_pr_era5, pr_model)
            list_values.append(np.mean(list_mspaef_model))
        list_values_all_models.append(list_values)
    array_mspaef_values = np.array(list_values_all_models)

    fig, ax = plt.subplots(figsize=(9,8))
    im = ax.imshow(array_mspaef_values, vmin=0.62, vmax=0.8, cmap = 'YlGnBu')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(seasons_name_list)), labels=seasons_name_list,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(model_name_list)), labels=model_name_list)

    # Loop over data dimensions and create text annotations.
    for i in range(len(model_name_list)):
        for j in range(len(seasons_name_list)):
            ax.text(j, i, np.round(array_mspaef_values[i, j], 2),
                        ha="center", va="center", color="w")
    fig.colorbar(
        im,
        ax=ax,
        orientation='horizontal',
        fraction=0.05,
        pad=0.08
    )
    ax.set_title("MSPAEF values")
    fig.tight_layout()
    plt.savefig(os.path.join(out_path,out_name))

def plot_double_itcz_indices(dict_pr_comp_all, dict_obs, first_year, last_year, out_path, out_name, naming_models):
    annual = [1,2,3,4,5,6,7,8,9,10,11,12]
    region = [120, 260]
    list_values_tpai_models = []
    list_values_epi_models = []
    list_values_tpai_obs = []
    list_values_epi_obs = []
    model_name_list = []
    obs_name_list = []

    for i, model_name in enumerate(dict_pr_comp_all.keys()):
        print(model_name)
        dict_pr_model = dict_pr_comp_all[model_name]
        model_name_list.append(naming_models[model_name])
        epi_vals = []
        tpai_vals = []
        for j, variant_name in enumerate(dict_pr_model.keys()):
            da_pr_variant = dict_pr_model[variant_name]
            mean_pr_variant = compute_seasonal_mean(da_pr_variant, annual, first_year, last_year)
            mean_pr_variant = mean_pr_variant.sel(lon=slice(region[0], region[1]))
            tpai_vals.append(tpai_index(mean_pr_variant))
            epi_vals.append(epi_index(mean_pr_variant))
        
        list_values_epi_models.append(np.mean(epi_vals))
        list_values_tpai_models.append(np.mean(tpai_vals))
    
    for i, obs_name in enumerate(dict_obs.keys()):
        print(obs_name)
        pr_obs = dict_obs[obs_name]
        obs_name_list.append(obs_name)
        mean_pr_obs = compute_seasonal_mean(pr_obs, annual, first_year, last_year)
        mean_pr_obs = mean_pr_obs.sel(lon=slice(region[0], region[1]))

        list_values_epi_obs.append(epi_index(mean_pr_obs))
        list_values_tpai_obs.append(tpai_index(mean_pr_obs))
    
    plt.figure(figsize=(8, 6))

    # --- Plot models (circles) ---
    for i in range(len(list_values_tpai_models)):
        plt.scatter(
            list_values_tpai_models[i],
            list_values_epi_models[i],
            color=f"C{i}",          # automatic distinct colors
            marker="o",             # circles
            s=80,
            label=model_name_list[i]
        )

    # --- Plot observations (rectangles/squares) ---
    for i in range(len(list_values_tpai_obs)):
        plt.scatter(
            list_values_tpai_obs[i],
            list_values_epi_obs[i],
            color=f"C{i+len(list_values_tpai_models)}",
            marker="s",             # square (rectangle-like)
            s=100,
            edgecolor="black",
            label=obs_name_list[i]
        )

    # Labels and title
    plt.xlabel("TPAI", fontsize=15)
    plt.ylabel("EPI", fontsize=15)
    #plt.title("TPAI vs EPI (Models vs Observations)")

    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # moves legend outside

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path,out_name))
    
    


if __name__ == '__main__':
    ################################################################################################################
    #### This code can make 3 different plots:                                                                 #####
    ####     1) plot_anomalies : Plot the precipitation anomalies of the multi-variant mean of the historical  #####
    ####        run for each model compared to ERA5 for the 1981-2005 period for each of the specified seasons #####
    ####        (a single plot for each season)                                                                #####
    ####     2) plot_mspaef_heatmap : Plot the MSPAEF metric valuesof precipitation for the multi-variant mean #####
    ####        of the historical un for each model compared to ERA5 for the 1981-2005 period for each of the  #####
    ####        specified seasons ( a single plot for all seasons)                                             #####
    ####     3) plot_double_itcz_indices : Plot the multi-variant mean of the TPAI and EPI indices for the     #####
    ####        historical run of each model, and for ERA5, NCEP Reanalysis 2 and GPCP for comparison          #####
    ####        (a single plot calculated only for the annual mean of precipitation)                           #####
    ################################################################################################################
    out_path = '/path/to/output/'  ### Directory to output the figure
    
    model_names = ['UKESM1-2','EC-Earth3','IPSL','CNRM','GFDL-ESM2M', 'MIROC-ES2L', 'NorESM2-LM']
    
    ##### Names to use for each model in the figure #######
    naming_models = {'UKESM1-2' : 'UKESM1-2-LL', 'EC-Earth3' : 'EC-Earth3-ESM-1', 'IPSL' : 'IPSL-CM6-ESMCO2',
              'CNRM' : 'CNRM-ESM2-2', 'GFDL-ESM2M': 'GFDL-ESM2M', 'MIROC-ES2L' : 'MIROC-ES2L', 'NorESM2-LM': 'NorESM2-LM'}
    
    name_out =  "Precipitation_evaluation_OptimESM_{}.png"           #### Output figure name for the precipitation anomalies plot
    name_out_mspaef = "MSPAEF_Precipitation_evaluation_OptimESM.pdf" #### Output figure name for the MSPAEF plot
    name_out_ditcz = "Double_ITCZ_evaluation.pdf"                      #### Output figure name for the Double ITCZ indices plot
    seasons_dict = {'DJF':[12,1,2], 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11], 'Annual': [1,2,3,4,5,6,7,8,9,10,11,12]}
    seasons_dict_only = {'DJF':[12,1,2], 'JJA':[6,7,8], 'Annual': [1,2,3,4,5,6,7,8,9,10,11,12]}

    do_regrid = True
    min_lon = 0     ####### Use 0 for Double ITCZ index, and -180 for plotting
    
    path_era5 = '/path/to/data/pr_ERA5_1940-2023.nc'
    model_name_era5 = 'ERA5'

    path_obs = '/path/to/data/pr_ncep_rea2_1979-2024.nc'
    model_name_obs = 'NCEP Reanalysis 2'

    path_obs2 = '/path/to/data/pr_gpcp_1979_2023.nc'
    model_name_obs2 = 'GPCP'

    lons=360
    lats=180
    
    first_year_use = 1981
    last_year_use = 2005
    
    dict_lonlat = {'lon':lons, 
                'lat':lats}
    
    ncfile_pr_era5= preparing_pr_data_direct(path_era5,'tp',first_year_use,last_year_use, dict_lonlat, do_regrid, min_lon = min_lon)
    ncfile_pr_obs = preparing_pr_data_direct(path_obs,'pr',first_year_use - 1, last_year_use + 1, dict_lonlat, do_regrid, min_lon = min_lon)
    ncfile_pr_obs2 = preparing_pr_data_direct(path_obs2,'pr',first_year_use - 1, last_year_use + 1, dict_lonlat, do_regrid, min_lon = min_lon)
    dict_rea_obs = {model_name_era5: ncfile_pr_era5, model_name_obs: ncfile_pr_obs, model_name_obs2: ncfile_pr_obs2}
    
    dict_pr_hist_all = {}

    for model_name in model_names:
        #### The path to historical runs should be in the form of 
        #### /path/to/data/MODEL_NAME/CMIP6Plus/hist/
        path_hist = '/path/to/data/{}/CMIP6Plus/hist/'.format(model_name)
        #### In each directory with data for instance: /path/to/data/EC-Earth3/CMIP6Plus/hist/
        #### The next directory will be the variant number as r1, r2, r3 etc
        #### E.g.: /path/to/data/EC-Earth3/CMIP6Plus/hist/r1/
        #### And then the variable name:
        #### E.g.: /path/to/data/EC-Earth3/CMIP6Plus/hist/r1/pr/
        #### This Python code, needs the variables: 'pr' to work
        #### where 'pr' is the raw pr data from the run 

        avail_variants = [f.path for f in os.scandir(path_hist) if f.is_dir() and f.path.split('/')[-1].startswith('r')]
        dict_pr_hist = {}
        for i, comp_variant in enumerate(avail_variants):
            files_pr = get_all_files(os.path.join(comp_variant, 'pr'))
            files_pr = filter_files_by_var_name(files_pr, 'pr')

            print('Calculating {} idealized variant'.format(comp_variant))
            ncfile_pr = preparing_pr_data(files_pr,'pr',first_year_use, last_year_use, dict_lonlat, do_regrid, min_lon = min_lon)

            # restrict domain
            dict_pr_hist[comp_variant] = ncfile_pr
        dict_pr_hist_all[model_name] = dict_pr_hist


    ##############################
    ####### Start Plotting #######
    ##############################
    ### Plot the precipitation anomalies
    plot_anomalies(dict_pr_hist_all, ncfile_pr_era5, seasons_dict_only, first_year_use, last_year_use,  out_path, name_out, naming_models)
    
    ### Plot the MSPAEF values
    plot_mspaef_heatmap(dict_pr_hist_all, ncfile_pr_era5, seasons_dict, first_year_use, last_year_use,  out_path, name_out_mspaef, naming_models)

    ### Plot the Double ITCZ indices
    plot_double_itcz_indices(dict_pr_hist_all, dict_rea_obs, first_year_use, last_year_use, out_path, name_out_ditcz, naming_models)
    
    
