import numpy as np
import xarray as xr
import os
import datetime as dt
import copy
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
import warnings
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from tqdm import tqdm
import seaborn as sns
import pathlib
import cmocean as cmo
warnings.filterwarnings("ignore")

def get_file(files,index):
    prefixed = [entry for entry in files if entry.startswith(index)]
    return prefixed

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
    lons = np.linspace(-180, 180, dict_lonlat['lon'])
    lats = np.linspace(-90, 90, dict_lonlat['lat'])
    print('Remapping data from {}x{} to {}x{} resolution'.format(len(array_var.lon),len(array_var.lat),len(lons),len(lats)))
    array_var = array_var.interp(lon=lons,lat=lats,method='linear')
    return array_var


def open_ready_nc_file(path, filename, dict_lonlat, model_name):
    ncfile = xr.open_dataset(os.path.join(path, filename))
    ncfile = convert_lon_to_minus180_180(ncfile)
    ncfile = remapping(ncfile, dict_lonlat)
    ncfile = ncfile.expand_dims("model")
    ncfile = ncfile.assign_coords(model=[model_name])
    return ncfile
    

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


def get_snr(delta, pi_periods):
    list_snrs = []
    for model_name in delta.model.values:
        delta_model = delta.sel(model=model_name)
        means = pi_periods.sel(model=model_name)
        diffs = means.diff('per')

        std = diffs.std(dim='per')
        std = std.interp_like(delta_model)
        #print(std)
        #print(delta_model)
        snr_val = np.abs(delta_model/std)
        snr_bool = xr.where(snr_val>1,1,0)

        snr_bool = snr_bool.assign_coords(model=model_name)
        list_snrs.append(snr_bool)
    combined_total_snr = xr.concat(list_snrs, dim="model")
    return combined_total_snr

def plot_changes(combined_pr_diffs_gwl2, combined_pr_diffs_gwl4, pi_control_prs, seasons, labels, out_path):
    name_out =  "Precipitation_changes_multi_model_mean_delta_pr.png"
    fig, axs = plt.subplots(
        2, len(seasons.keys()), figsize=(17, 9), constrained_layout=True, 
        subplot_kw=dict(projection=ccrs.Mollweide())
    )
    for j, season_n in enumerate(seasons.keys()):
        change_multimodel1 = combined_pr_diffs_gwl2['delta_pr_adj'].sel(season=season_n)
        change_multimodel2 = combined_pr_diffs_gwl4['delta_pr_adj'].sel(season=season_n)

        pi_periods_season = pi_control_prs['pr'].sel(season=season_n)
        snr1 = get_snr(change_multimodel1, pi_periods_season)
        snr2 = get_snr(change_multimodel2, pi_periods_season)

        ### Hatching where at least 5/7 models agree on the sign of change and at least 4/7 are robust (SNR>1) ###
        hatch_model_agreement_sign_1, mean_change_1 = get_hatching_mask(change_multimodel1, 7, agreement=True)
        hatch_model_agreement_robust_1, _ = get_hatching_mask(snr1, 5, agreement=True)
        hatch_1 = hatch_model_agreement_sign_1 * hatch_model_agreement_robust_1

        levels = np.arange(-2,2.2,0.2)
        c1 = axs[0,j].contourf(mean_change_1.lon.values,mean_change_1.lat.values, mean_change_1,
                                 vmin=-2,vmax=2,levels=levels,cmap = 'BrBG', alpha = 0.85,
                                 transform = ccrs.PlateCarree(),extend='both')
        axs[0,j].coastlines()
        density=2
        cs2_r = axs[0,j].contourf(hatch_1.lon,hatch_1.lat,hatch_1,levels=[0.5,1.5],colors='none', alpha=0,hatches=[2*density*'/',4*density*'/'], transform = ccrs.PlateCarree())
        axs[0,j].set_title(season_n, fontsize=25)

        hatch_model_agreement_sign_2, mean_change_2 = get_hatching_mask(change_multimodel2, 7, agreement=True)
        hatch_model_agreement_robust_2, _ = get_hatching_mask(snr2, 5, agreement=True)
        hatch_2 = hatch_model_agreement_sign_2 * hatch_model_agreement_robust_2

        c1 = axs[1,j].contourf(mean_change_2.lon.values,mean_change_2.lat.values, mean_change_2,
                                 vmin=-2,vmax=2,levels=levels,cmap = 'BrBG', alpha = 0.85,
                                 transform = ccrs.PlateCarree(),extend='both')
        axs[1,j].coastlines()
        cs2_r = axs[1,j].contourf(hatch_2.lon,hatch_2.lat,hatch_2,levels=[0.5,1.5],colors='none', alpha=0,hatches=[2*density*'/',4*density*'/'], transform = ccrs.PlateCarree())
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
    plt.savefig(os.path.join(out_path,'Precipitation_changes_multi_model_mean_adj.png'))

if __name__ == '__main__':
    data_path = '/path/to/data/'
    out_path = '/path/to/out/'
    model_names = ['GFDL-ESM2M', 'NorESM2-LM', 'MIROC-ES2L', 'UKESM1-2','EC-Earth3','IPSL','CNRM']
    name_out =  "Precipitation_changes_multi_model_mean.png"

    seasons_dict = {'DJF':[12,1,2], 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]}
    seasons_dict_only = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    seasons_dict_shifted = {'JFM':[1,2,3], 'AMJ':[4,5,6], 'JAS':[7,8,9], 'OND':[10,11,12]}
    seasons_dict_shifted_only = {'JFM':[1,2,3], 'JAS':[7,8,9]}
    scratch_out_path = '/nvme/h/akarpasitis/scratch/ITCZ_calc/'
    
    pr_idealized2_all = []
    pr_idealized4_all = []

    lons=360
    lats=180
    
    dict_lonlat = {'lon':lons, 
                'lat':lats}

    delay = 100
    years_range = 50

    pi_control_prs = xr.open_dataset(os.path.join(data_path,'precipitation_picontrol_periods.nc'))


    combined_pr_diffs_gwl2 = xr.open_dataset(os.path.join(data_path,'Pr_deltas_gwl2_vs_picontrol.nc'))
    combined_pr_diffs_gwl4 = xr.open_dataset(os.path.join(data_path,'Pr_deltas_gwl4_vs_picontrol.nc'))


    ##############################
    ####### Start Plotting #######
    ##############################
    labels = ['gwl2p0', 'gwl4p0']
    labels_better = []
    for label in labels:
        if label == 'gwl2p0':
            new_label = '$2^oC$ GWL'
        elif label == 'gwl4p0':
            new_label = '$4^oC$ GWL'
        else:
            new_label = 'GWL'
        labels_better.append(new_label)
    plot_changes(combined_pr_diffs_gwl2, combined_pr_diffs_gwl4, pi_control_prs, seasons_dict_only, 
                 labels_better, out_path)
