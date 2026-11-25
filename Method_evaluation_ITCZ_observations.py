import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import pandas as pd
import cmocean as cmo
import os
import warnings
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")

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

data_path = '/path/to/data/'
out_path = '/path/to/out/'


##############################################################
###### Plot 8 (ITCZ methodology evaluation)             ######
##############################################################
file_csv = os.path.join(data_path,'observations_itcz_characteristics.csv')
file = os.path.join(data_path,'Obervations_reanalysis_precipitation_fig.nc')

da = xr.open_dataset(file)
da = da['pr']

df = pd.read_csv(file_csv)

lat_limit = 40
seasons_dict_only = {'DJF':[12,1,2], 'JJA':[6,7,8]}
regions = {'America':[-90,-40], 'Atlantic':[-40,-10],
            'Africa':[-10,40],'Indian Ocean':[40,120],
            'Western Pacific':[120,170], 'Central Pacific':[170,230], 
            'Eastern Pacific':[230,270]}

model_name_era5 = 'ERA5'
model_name_obs = 'NCEP Reanalysis 2'
model_name_obs2 = 'GPCP'

name_out = "ITCZ_map_{}_{}_{}.pdf".format(model_name_era5, model_name_obs, model_name_obs2)

lon_lines = regions.values()
lon_lines = [val for sublist in lon_lines for val in sublist]
lon_lines = set(lon_lines)
lon_lines_beyond_180 = [f-360 for f in lon_lines if f>180]
lon_lines_before_180 = [f for f in lon_lines if f<180]
lon_lines = lon_lines_beyond_180 + lon_lines_before_180

dict_rea_obs = [model_name_era5, model_name_obs, model_name_obs2]

fig, axs = plt.subplots(
    len(dict_rea_obs), 2, figsize=(20, 11),constrained_layout=True, 
    subplot_kw=dict(projection=ccrs.PlateCarree())
)

for i, season_n in enumerate(seasons_dict_only.keys()):
    if season_n == 'JFM' or season_n == 'DJF':
        c= 'blue'
    else:
        c = 'red'
    axs[0, i].set_title(season_n, fontsize=25)

    for k, name in enumerate(dict_rea_obs):
        
        p_trop_plot = da.sel(model = name, season = season_n)
        print(p_trop_plot.values)
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
            
            phi_itcz_season = df[(df['region']==region)&(df['season']==season_n)&(df['model']==name)]['phi_itcz_season'].values[0]
            phi_itcz_season_min = df[(df['region']==region)&(df['season']==season_n)&(df['model']==name)]['phi_itcz_season_min'].values[0]
            phi_itcz_season_max = df[(df['region']==region)&(df['season']==season_n)&(df['model']==name)]['phi_itcz_season_max'].values[0]
            
            edgemin = df[(df['region']==region)&(df['season']==season_n)&(df['model']==name)]['edgemin'].values[0]
            edgemin_min = df[(df['region']==region)&(df['season']==season_n)&(df['model']==name)]['edgemin_min'].values[0]
            edgemin_max = df[(df['region']==region)&(df['season']==season_n)&(df['model']==name)]['edgemin_max'].values[0]
            
            edgemax = df[(df['region']==region)&(df['season']==season_n)&(df['model']==name)]['edgemax'].values[0]
            edgemax_min = df[(df['region']==region)&(df['season']==season_n)&(df['model']==name)]['edgemax_min'].values[0]
            edgemax_max = df[(df['region']==region)&(df['season']==season_n)&(df['model']==name)]['edgemax_max'].values[0]
            
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
