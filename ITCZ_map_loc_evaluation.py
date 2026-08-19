import numpy as np
import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import cartopy.crs as ccrs
from tqdm import tqdm
import cmocean as cmo
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


def preparing_pr_data_direct(file,var_name,first_year,last_year):
    dataaaray = xr.open_dataset(file)
    #print(dataaaray)
    dataaaray = dataaaray[var_name]
    dataaaray = renaming_dimensions(dataaaray)
    if not np.all(np.diff(dataaaray['lat'].values) > 0):
        dataaaray = dataaaray.sortby('lat')
    dataaaray = dataaaray.where((dataaaray.time.dt.year>=first_year)&(dataaaray.time.dt.year<=last_year),drop=True)
    dataaaray = dataaaray.transpose('time', 'lat', 'lon')
    if dataaaray.max() < 1:
        #print('Multiplying by 86400')
        dataaaray = dataaaray*1000  ### From m/day, to mm/day
    return dataaaray

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

if __name__ == '__main__':
    #############################################################################################################
    #### This code plots the the multimodel mean of the ITCZ characteristics for the ERA5, NCEP Reanalysis 2#####
    #### and GPCP                                                                                           #####
    #############################################################################################################
    path_era5 = '/path/to/data/pr_ERA5_1940-2023.nc'
    model_name_era5 = 'ERA5'

    path_obs = '/path/to/data/pr_ncep_rea2_1979-2024.nc'
    model_name_obs = 'NCEP Reanalysis 2'

    path_obs2 = '/path/to/data/pr_gpcp_1979_2023.nc'
    model_name_obs2 = 'GPCP'
    
    out_path = '/path/to/output/'  ### Directory to output the figure

    years_range = 30  # How many years the analysis will span

    first_year_use = 1980
    last_year_use = 2010

    regions = {'America':[-90,-40], 'Atlantic':[-40,-10],
                'Africa':[-10,40],'Indian Ocean':[40,120],
                'Western Pacific':[120,170], 'Central Pacific':[170,230], 
                'Eastern Pacific':[230,270]}
    seasons_dict = {'DJF':[12,1,2], 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]}
    seasons_dict_only = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    seasons_dict_shifted = {'JFM':[1,2,3], 'AMJ':[4,5,6], 'JAS':[7,8,9], 'OND':[10,11,12]}
    seasons_dict_shifted_only = {'JFM':[1,2,3], 'JAS':[7,8,9]}
    
    lon_lines = regions.values()
    lon_lines = [val for sublist in lon_lines for val in sublist]
    lon_lines = set(lon_lines)
    lon_lines_beyond_180 = [f-360 for f in lon_lines if f>180]
    lon_lines_before_180 = [f for f in lon_lines if f<180]
    lon_lines = lon_lines_beyond_180 + lon_lines_before_180

    print(lon_lines)

    name_out = "ITCZ_map_{}_{}_{}.pdf".format(model_name_era5, model_name_obs, model_name_obs2)

    ncfile_pr_era5 = preparing_pr_data_direct(path_era5,'tp',first_year_use - 1, last_year_use + 1)
    ncfile_pr_obs = preparing_pr_data_direct(path_obs,'pr',first_year_use - 1, last_year_use + 1)
    ncfile_pr_obs2 = preparing_pr_data_direct(path_obs2,'pr',first_year_use - 1, last_year_use + 1)

    
    # restrict domain
    lat_limit = 40
    P_trop_era5 = ncfile_pr_era5.sel(lat=slice(-lat_limit, lat_limit))
    P_trop_obs = ncfile_pr_obs.sel(lat=slice(-lat_limit, lat_limit))
    P_trop_obs2 = ncfile_pr_obs2.sel(lat=slice(-lat_limit, lat_limit))

    dict_rea_obs = {model_name_era5: P_trop_era5, model_name_obs: P_trop_obs, model_name_obs2: P_trop_obs2}

    fig, axs = plt.subplots(
        len(dict_rea_obs), 2, figsize=(20, 11),constrained_layout=True, 
        subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    for i, season_n in enumerate(seasons_dict_only.keys()):
        if season_n == 'JFM' or season_n == 'DJF':
            c= 'blue'
        else:
            c = 'red'
        season = seasons_dict_only[season_n]
        axs[0, i].set_title(season_n, fontsize=25)
        for k, name in enumerate(dict_rea_obs.keys()):
            P_trop = dict_rea_obs[name]
            p_trop_season = compute_seasonal_mean(P_trop,season, first_year_use, last_year_use)

            p_trop_plot = p_trop_season.mean(dim='time')
            p_trop_plot = convert_lon_to_minus180_180(p_trop_plot)

            levels = np.arange(0,18,3)
            c1 = axs[k,i].contourf(p_trop_plot.lon.values,p_trop_plot.lat.values, p_trop_plot,vmin=0,vmax=15,levels=levels,cmap = cmo.cm.rain,transform = ccrs.PlateCarree(),extend='max')
            axs[k,i].coastlines()
            
            
            gl = axs[k,i].gridlines(draw_labels=True,color = 'gray', alpha=0.4)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 20, 'color': 'gray'}
            gl.ylabel_style = {'size': 20, 'color': 'gray'}
            
            for j, region in enumerate(regions.keys()):
                lons_region = regions[region]

                pos = np.mean(lons_region)
                if pos> 180:
                    pos = pos - 360
                phi_itcz, diagnostics = itcz_centroid_robust_time(p_trop_season, lons_region)

                phi_itcz_season = np.nanmedian(phi_itcz.values)
                edgemin = np.nanmedian(diagnostics['lat_min'].values)
                edgemax = np.nanmedian(diagnostics['lat_max'].values)

                phi_itcz_season_min = np.nanmin(phi_itcz.values)
                edgemin_min = np.quantile(diagnostics['lat_min'].values, 0.1)
                edgemax_min = np.quantile(diagnostics['lat_max'].values, 0.1)

                phi_itcz_season_max = np.nanmax(phi_itcz.values)
                edgemin_max = np.quantile(diagnostics['lat_min'].values, 0.9)
                edgemax_max = np.quantile(diagnostics['lat_max'].values, 0.9)

                center = np.array([phi_itcz_season])
                lat_bounds = np.array([[phi_itcz_season - edgemin],
                                    [edgemax - phi_itcz_season]])
                
                center_min = np.array([phi_itcz_season_min])
                lat_bounds_min = np.array([[phi_itcz_season_min - edgemin_min],
                                    [edgemax_min - phi_itcz_season_min]])

                center_max = np.array([phi_itcz_season_max])
                lat_bounds_max = np.array([[phi_itcz_season_max - edgemin_max],
                                    [edgemax_max - phi_itcz_season_max]])

                axs[k,i].hlines(
                                [edgemin_min, edgemax_max],
                                xmin=pos - 10,
                                xmax=pos + 10,
                                color='black',
                                linewidth=3,
                                transform=ccrs.PlateCarree()
                            )
                bbplots = axs[k,i].errorbar(pos,center, yerr=lat_bounds, color = c, elinewidth=2,
                                        capsize=7, capthick =2, marker = 's', markersize=8, linestyle='',
                                        transform = ccrs.PlateCarree())

                axs[k,i].set_ylim(-lat_limit,lat_limit)
            axs[k, 0].text(
                -0.1, 0.5, name,
                transform=axs[k,0].transAxes,
                fontsize=20,
                rotation=90,
                va='center',
                ha='center'
            )
            for lon in lon_lines:
                axs[k,i].vlines(x=lon, ymin=-40, ymax=40, color='red', linestyle='--', transform = ccrs.PlateCarree())  
    cbar = plt.colorbar(c1, ax=axs, orientation="horizontal", pad=0.05, label="mm $d^{-1}$", shrink=0.5)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('mm $d^{-1}$',fontsize=20)              

    plt.savefig(os.path.join(out_path,name_out))
