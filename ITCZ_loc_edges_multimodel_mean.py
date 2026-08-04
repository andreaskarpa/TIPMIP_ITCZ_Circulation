import numpy as np
import xarray as xr
import os
import copy
import pandas as pd
import matplotlib.pyplot as plt
import warnings
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

def preparing_pr_data(files,var_name,first_year,last_year):
    dataaaray = ensemble_time_series(files, var_name, first_year, last_year)
    dataaaray = renaming_dimensions(dataaaray)
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

def itcz_centroid_robust_time(
    P_da,
    lons,
    lat_range=(-40, 40),
    alpha=0.5):
    """
    Time-aware robust ITCZ centroid.

    Returns
    -------
    phi_itcz : xarray.DataArray (time)
    diagnostics : dict of xarray.DataArray (time)
    """

    # --- common spatial preprocessing ---
    P = longitude_cut(P_da,lons)

    # Zonal mean 
    P_zm = P.mean(dim="lon")
    lats_to_interpolate = np.arange(lat_range[0],lat_range[1], 0.1)
    P_zm = P_zm.interp(lat=lats_to_interpolate)
    P_zm = P_zm.sel(lat=slice(lat_range[0], lat_range[1]))

    lat = P_zm.lat
    coslat = np.cos(np.deg2rad(lat))

    phi_list = []
    width_list = []
    strength_list = []
    latmin_list = []
    latmax_list = []

    # --- loop over time ---
    for t in tqdm(P_zm.time):
        P_t = P_zm.sel(time=t)
        #print(P_t)

        # threshold
        P_thr = alpha * P_t.max(dim="lat")
        #P_thr = 4
        mask = P_t >= P_thr

        # contiguous regions
        mask_np = mask.values
        regions = []
        start = None

        for i in range(len(lat)):
            if mask_np[i] and start is None:
                start = i
            elif not mask_np[i] and start is not None:
                regions.append((start, i - 1))
                start = None

        if start is not None:
            regions.append((start, len(lat) - 1))

        if len(regions) == 0:
            phi_list.append(np.nan)
            width_list.append(np.nan)
            strength_list.append(np.nan)
            latmin_list.append(np.nan)
            latmax_list.append(np.nan)
            continue

        pr_peaks = []
        widths = []
        lat_peaks = []
        for i0, i1 in regions:
            Sp = (P_t.isel(lat=slice(i0, i1 + 1))* coslat.isel(lat=slice(i0, i1 + 1)))     
            pr_peaks.append(Sp.max())
            lat_peaks.append(Sp.idxmax(dim='lat'))
            widths.append(i1 - i0)


        pr_peaks = xr.concat(pr_peaks, dim="region")
        lat_peaks = xr.concat(lat_peaks, dim="region")
        widths = xr.DataArray(widths, dims="region")


        max_peak = pr_peaks.max()

        # peaks within 80% of max
        mask_peaks = pr_peaks >= 0.8 * max_peak

        valid_lats = lat_peaks.where(mask_peaks)

        # count valid peaks
        n_valid = mask_peaks.sum()

        # check for north and south peaks
        has_north = (valid_lats > 0).any()
        has_south = (valid_lats < 0).any()

        if (n_valid >= 2) and has_north and has_south:
            # keep only northern peaks
            north_lats = valid_lats.where(valid_lats > 0)

            # choose closest to equator → smallest absolute latitude
            k_star = np.abs(north_lats).argmin(dim="region")
        elif n_valid >= 2:
            k_star = np.abs(valid_lats).argmin(dim="region")
        else:
            k_star = pr_peaks.argmax(dim="region")


        i0, i1 = regions[int(k_star)]
        S = (
            (P_t.isel(lat=slice(i0, i1 + 1))
                * coslat.isel(lat=slice(i0, i1 + 1)))).sum(dim="lat")*np.mean(np.diff(P_t.lat))
                

        band_P = P_t.isel(lat=slice(i0, i1 + 1))
        band_lat = lat.isel(lat=slice(i0, i1 + 1))

        # centroid
        phi = (
            (band_P * band_lat * np.cos(np.deg2rad(band_lat))).sum(dim="lat")
            / (band_P * np.cos(np.deg2rad(band_lat))).sum(dim="lat")
        )

        phi_list.append(phi.values)
        latmin_list.append(band_lat.min().values)
        latmax_list.append(band_lat.max().values)
        width_list.append((band_lat.max() - band_lat.min()).values)
        strength_list.append(S)

    # --- assemble output ---
    phi_itcz = xr.DataArray(
        phi_list,
        coords={"time": P_zm.time},
        dims="time",
        name="itcz_latitude"
    )

    diagnostics = {
        "lat_min": xr.DataArray(latmin_list, coords={"time": P_zm.time}, dims="time"),
        "lat_max": xr.DataArray(latmax_list, coords={"time": P_zm.time}, dims="time"),
        "width": xr.DataArray(width_list, coords={"time": P_zm.time}, dims="time"),
        "strength": xr.DataArray(strength_list, coords={"time": P_zm.time}, dims="time")
    }
    return phi_itcz, diagnostics

def median_of_dict_values(dict_in, variable=None):
    list_out = []
    if variable == None:
        for variant in dict_in.keys():
            values_var = dict_in[variant]
            list_out.append(np.nanmedian(values_var.values))
    else:
        for variant in dict_in.keys():
            values_var = dict_in[variant][variable]
            list_out.append(np.nanmedian(values_var.values))
    return list_out

def calculate_model_mean_diff(dict_pr_comp, dict_pr_idealized,season, 
                              years_use_comp, years_use_idealized,lons_region):
    phi_itcz_dict, diagnostics_dict = {}, {}
    phi_itcz_idea_dict, diagnostics_idea_dict = {}, {}

    ### Calculate the ITCZ characteristics for each year of each variant and keep everything
    for comp_variant in dict_pr_comp.keys():
        P_trop = dict_pr_comp[comp_variant]
        first_year, last_year = years_use_comp[comp_variant]
        p_trop_season = compute_seasonal_mean(P_trop,season, first_year, last_year)

        phi_itcz, diagnostics = itcz_centroid_robust_time(p_trop_season, lons_region)
        phi_itcz_dict[comp_variant] = phi_itcz
        diagnostics_dict[comp_variant] = diagnostics
    
    for idea_variant in dict_pr_idealized.keys():
        P_trop_idea = dict_pr_idealized[idea_variant]
        first_year, last_year = years_use_idealized[idea_variant]
        p_trop_idea_season = compute_seasonal_mean(P_trop_idea,season, first_year, last_year)
        
        phi_itcz_idea, diagnostics_idea = itcz_centroid_robust_time(p_trop_idea_season, lons_region)
        phi_itcz_idea_dict[idea_variant] = phi_itcz_idea
        diagnostics_idea_dict[idea_variant] = diagnostics_idea
    ### Calculate the change of the median
    itcz_loc_change = np.mean( median_of_dict_values(phi_itcz_idea_dict) ) - np.mean(median_of_dict_values(phi_itcz_dict))
    northern_edge_loc_change = np.mean( median_of_dict_values(diagnostics_idea_dict, variable='lat_max') ) - np.mean( median_of_dict_values(diagnostics_dict, variable='lat_max') )                                    
    southern_edge_loc_change = np.mean( median_of_dict_values(diagnostics_idea_dict, variable='lat_min') ) - np.mean( median_of_dict_values(diagnostics_dict, variable='lat_min') )
    strength_change = np.mean( median_of_dict_values(diagnostics_idea_dict, variable='strength') ) - np.mean( median_of_dict_values(diagnostics_dict, variable='strength') )
    
    snr_loc = np.abs(np.mean(array_from_dict(phi_itcz_idea_dict)) - np.mean(array_from_dict(phi_itcz_dict))) / np.std(array_from_dict(phi_itcz_dict))
    snr_northern = np.abs(np.mean(array_from_dict(diagnostics_idea_dict, variable='lat_max')) - np.mean(array_from_dict(diagnostics_dict, variable='lat_max'))) / np.std(array_from_dict(diagnostics_dict, variable='lat_max'))
    snr_southern = np.abs(np.mean(array_from_dict(diagnostics_idea_dict, variable='lat_min')) - np.mean(array_from_dict(diagnostics_dict, variable='lat_min'))) / np.std(array_from_dict(diagnostics_dict, variable='lat_min'))
    snr_strength = np.abs(np.mean(array_from_dict(diagnostics_idea_dict, variable='strength')) - np.mean(array_from_dict(diagnostics_dict, variable='strength'))) / np.std(array_from_dict(diagnostics_dict, variable='strength'))
    
    return itcz_loc_change, northern_edge_loc_change, southern_edge_loc_change, strength_change, snr_loc, snr_northern, snr_southern, snr_strength

def array_from_dict(dict_in, variable=None):
    array_out = np.array([])
    if variable == None:
        for variant in dict_in.keys():
            values_var = dict_in[variant]
            array_out = np.append(array_out, values_var.values)
    else:
        for variant in dict_in.keys():
            values_var = dict_in[variant][variable]
            array_out = np.append(array_out, values_var.values)
    number_per_variant = len(values_var)
    if len(array_out) > number_per_variant:
        random_sample = np.random.choice(array_out, number_per_variant)
        return random_sample
    else:
        return array_out


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

def multi_model(dict_pr_comp_all, dict_pr_idealized_all, season, 
                years_use_comp_all, years_use_idealized_all, lons_region):
    loc_list, latmax_list, latmin_list, strength_list = [], [], [], []
    snr_loc_list, snr_latmax_list, snr_latmin_list, snr_strength_list = [], [], [], []
    for model_name in dict_pr_comp_all.keys():
        dict_pr_comp = dict_pr_comp_all[model_name]
        dict_pr_idealized = dict_pr_idealized_all[model_name]
        years_use_comp = years_use_comp_all[model_name]
        years_use_idealized = years_use_idealized_all[model_name]
        loc, latmax, latmin, strength, snr_loc, snr_northern, snr_southern, snr_strength = calculate_model_mean_diff(dict_pr_comp, dict_pr_idealized,season, 
                                                                  years_use_comp, years_use_idealized,lons_region)
        loc_list.append(loc)
        latmax_list.append(latmax)
        latmin_list.append(latmin)
        strength_list.append(strength)

        snr_loc_list.append(snr_loc)
        snr_latmax_list.append(snr_northern)
        snr_latmin_list.append(snr_southern)
        snr_strength_list.append(snr_strength)
    return loc_list, latmax_list, latmin_list, strength_list, snr_loc_list, snr_latmax_list, snr_latmin_list, snr_strength_list


def plot_changes(dict_pr_comp_all, dict_pr_idealized1_all, dict_pr_idealized2_all, regions, seasons, years_use_comp_all, 
                 years_use_idealized1_all, years_use_idealized2_all, labels, out_path, out_name):
    fig, axs = plt.subplots(
        4, len(regions.keys()), figsize=(18, 15),
        constrained_layout=True, gridspec_kw={'height_ratios': [1, 1, 1 ,1]}
    )
    offset = 0.1 
    box_width = 0.1 
    for j, region in enumerate(regions.keys()):
        lons_region = regions[region]
        for i, season_n in enumerate(seasons.keys()):
            if season_n == 'JFM' or season_n == 'DJF':
                c= 'blue'
                # Shift the base positions [1, 2] to the left
                pos = [1 - offset, 2 - offset]
            else:
                c = 'red'
                # Shift the base positions [1, 2] to the right
                pos = [1 + offset, 2 + offset]
            season = seasons[season_n]

            loc_list1, latmax_list1, latmin_list1, strength_list1, snr_loc_list1, snr_latmax_list1, snr_latmin_list1, snr_strength_list1 = multi_model(dict_pr_comp_all, dict_pr_idealized1_all, season, years_use_comp_all, years_use_idealized1_all, lons_region)
            loc_list2, latmax_list2, latmin_list2, strength_list2, snr_loc_list2, snr_latmax_list2, snr_latmin_list2, snr_strength_list2 = multi_model(dict_pr_comp_all, dict_pr_idealized2_all, season, years_use_comp_all, years_use_idealized2_all, lons_region)

            box_width = 0.1
            axs[0, j].boxplot(latmax_list1, positions=[pos[0]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c), label=season_n)
            axs[0, j].boxplot(latmax_list2, positions=[pos[1]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c))

            axs[1, j].boxplot(loc_list1, positions=[pos[0]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c), label=season_n)
            axs[1, j].boxplot(loc_list2, positions=[pos[1]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c))

            axs[2, j].boxplot(latmin_list1, positions=[pos[0]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c), label=season_n)
            axs[2, j].boxplot(latmin_list2, positions=[pos[1]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c))

            axs[3, j].boxplot(strength_list1, positions=[pos[0]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c), label=season_n)
            axs[3, j].boxplot(strength_list2, positions=[pos[1]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c))

        if j==len(regions)-1:
            axs[0,j].legend()
        axs[0,j].axhline(0, color='black')
        axs[1,j].axhline(0, color='black')
        axs[2,j].axhline(0, color='black')
        axs[3,j].axhline(0, color='black')

        axs[0, j].set_xticks([1,2],labels, fontsize=13)
        axs[0, j].set_xticklabels([])
        axs[0, j].set_xlim(0.7,2.3)
        axs[0, j].grid(visible=True, which='major',axis='y', color='grey', alpha = 0.5, linestyle='--')
        axs[0, j].xaxis.set_tick_params(which='both', labelbottom=True)
        axs[0, j].yaxis.set_tick_params(labelsize=13)

        axs[1, j].set_xticks([1,2],labels, fontsize=13)
        axs[1, j].set_xticklabels([])
        axs[1, j].set_xlim(0.7,2.3)
        axs[1, j].grid(visible=True, which='major',axis='y', color='grey', alpha = 0.5, linestyle='--')
        axs[1, j].xaxis.set_tick_params(which='both', labelbottom=True)
        axs[1, j].yaxis.set_tick_params(labelsize=13)

        axs[2, j].set_xticks([1,2],labels, fontsize=13)
        axs[2, j].set_xticklabels([])
        axs[2, j].set_xlim(0.7,2.3)
        axs[2, j].grid(visible=True, which='major',axis='y', color='grey', alpha = 0.5, linestyle='--')
        axs[2, j].xaxis.set_tick_params(which='both', labelbottom=True)
        axs[2, j].yaxis.set_tick_params(labelsize=13)

        axs[3, j].set_xticks([1,2],labels, fontsize=18)
        axs[3, j].set_xlim(0.7,2.3)
        axs[3, j].grid(visible=True, which='major',axis='y', color='grey', alpha = 0.5, linestyle='--')
        axs[3, j].xaxis.set_tick_params(which='both', labelbottom=True)
        axs[3, j].yaxis.set_tick_params(labelsize=13)

        axs[3, j].set_xlabel(region, fontsize=20)
    axs[0,0].set_ylabel('Northern edge \nlatitude difference (${}^o$)', fontsize=18)
    axs[1,0].set_ylabel('ITCZ location \nlatitude difference (${}^o$)', fontsize=18)
    axs[2,0].set_ylabel('Southern edge \nlatitude difference (${}^o$)', fontsize=18)
    axs[3,0].set_ylabel('Strength difference \n(mm $d^{-1}$ ${}^o$)', fontsize=18)
    plt.suptitle("", fontsize=20)
    plt.savefig(os.path.join(out_path,out_name))


if __name__ == '__main__':
    #########################################################################################################
    #### This code plots the the multimodel mean of the change of the ITCZ characteristics and strength #####
    #### for the different longitudinal regions for both GWLs                                           #####
    #########################################################################################################
    out_path = '/path/to/output/'  ### Directory to output the figure
    model_names = ['GFDL-ESM2M', 'NorESM2-LM', 'MIROC-ES2L', 'UKESM1-2','EC-Earth3','IPSL','CNRM']
    
    name_out =  "ITCZ_loc_edges_multi_model_mean_boxplots.pdf"  ### Name of output figure
    seasons_dict_only = {'DJF':[12,1,2], 'JJA':[6,7,8]}             ### Seasons to plot


    regions = {'America':[-90,-40], 'Atlantic':[-40,-10],
                'Africa':[-10,40],'Indian Ocean':[40,120],
                'Western Pacific':[120,170], 'Central Pacific':[170,230], 
                'Eastern Pacific':[230,270], 'World':[0,360]}               ##### Individual Regions 
    
    dict_pr_comp_all = {}
    dict_pr_idealized1_all = {}
    dict_pr_idealized2_all = {}

    dict_years_use_comp_all = {}
    dict_years_use_idealized1_all = {}
    dict_years_use_idealized2_all = {}

    delay = 100   # Years to skip at the beginning of each run
    years_range = 50 # How many years the analysis will span for each run

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
            ncfile_pr = preparing_pr_data(files_pr,'pr',first_year_use_pi-1, last_year_use_pi+1)

            # restrict domain
            P_trop = ncfile_pr.sel(lat=slice(-40, 40))
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

            ncfile_pr_idea = preparing_pr_data(files_pr,'pr',first_year_use_idealized1-1, last_year_use_idealized1+1)

            # restrict domain
            P_trop_idea = ncfile_pr_idea.sel(lat=slice(-40, 40))
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

            ncfile_pr_idea = preparing_pr_data(files_pr,'pr',first_year_use_idealized2-1, last_year_use_idealized2+1)

            # restrict domain
            P_trop_idea = ncfile_pr_idea.sel(lat=slice(-40, 40))
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
    labels = ['$2^o$C', '$4^o$C']
    plot_changes(dict_pr_comp_all, dict_pr_idealized1_all, dict_pr_idealized2_all, regions, seasons_dict_only, 
                 dict_years_use_comp_all, dict_years_use_idealized1_all, dict_years_use_idealized2_all, labels, out_path, name_out)