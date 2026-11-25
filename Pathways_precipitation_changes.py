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

out_name = "Precipitation_changes_contributions_pathway_50y.png"
    

file_pr_pi_data_periods = os.path.join(data_path,'precipitation_picontrol_periods.nc')
da_pr_pi_periods = xr.open_dataset(file_pr_pi_data_periods)
da_pr_pi_periods = da_pr_pi_periods['pr']


file_early = os.path.join(data_path,'pr_changes_pathways_early_adjustment_50y.nc')
file_late = os.path.join(data_path,'pr_changes_pathways_late_adjustment_50y.nc')

da_early = xr.open_dataset(file_early)
da_late = xr.open_dataset(file_late)

da_early = da_early['pr']
da_late = da_late['pr']



seasons_series = ['DJF', 'DJF', 'JJA', 'JJA']
gwl_series = [2,4,2,4]

change_list_1 = []
change_list_2 = []

fig, axs = plt.subplots(
    4, 2, figsize=(19, 14), constrained_layout=True, 
    subplot_kw=dict(projection=ccrs.PlateCarree())
)
for j, (season_n, gwl) in enumerate(zip(seasons_series,gwl_series)):
    print(gwl, season_n)

    change = da_early.sel(season=season_n, gwl=gwl)
    pi_periods_season = da_pr_pi_periods.sel(season=season_n)
    snr = get_snr(change, pi_periods_season)  


    levels = np.arange(4, 16, 4)
    levels_diff = np.arange(-0.5, 0.6, 0.1) #[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]


    density=2
    c1 = axs[j,0].contourf(change.lon.values,change.lat.values, change.mean(dim='model'),
                            levels=levels_diff,cmap = 'BrBG', alpha = 0.85,
                            transform = ccrs.PlateCarree(),extend='both')
    '''c2 = axs[j,0].contour(mean.lon.values, mean.lat.values, mean.mean(dim='model'),
                            levels=levels,cmap = 'Blues', alpha = 0.95, linewidths=3,
                            transform = ccrs.PlateCarree())'''
    hatch_model_agreement_sign, _ = get_hatching_mask(change, 5, agreement=True)
    hatch_model_agreement_robust, _ = get_hatching_mask(snr, 4, agreement=True)
    hatch = hatch_model_agreement_sign * hatch_model_agreement_robust
    cs2_r = axs[j,0].contourf(hatch.lon,hatch.lat,hatch,levels=[0.5,1.5],colors='none', alpha=0,hatches=[2*density*'/',4*density*'/'], transform = ccrs.PlateCarree())
    
    axs[j,0].set_ylim(-40, 40)
    axs[j,0].coastlines()

    change = da_late.sel(season=season_n, gwl=gwl)
    snr = get_snr(change, pi_periods_season)      

    
    c1 = axs[j,1].contourf(change.lon.values,change.lat.values, change.mean(dim='model'),
                            levels=levels_diff,cmap = 'BrBG', alpha = 0.85,
                            transform = ccrs.PlateCarree(),extend='both')
    '''c2 = axs[j,1].contour(mean.lon.values, mean.lat.values, mean.mean(dim='model'),
                            levels=levels,cmap = 'Blues', alpha = 0.95, linewidths=3,
                            transform = ccrs.PlateCarree())'''
    hatch_model_agreement_sign, _ = get_hatching_mask(change, 5, agreement=True)
    hatch_model_agreement_robust, _ = get_hatching_mask(snr, 4, agreement=True)
    hatch = hatch_model_agreement_sign * hatch_model_agreement_robust
    cs2_r = axs[j,1].contourf(hatch.lon,hatch.lat,hatch,levels=[0.5,1.5],colors='none', alpha=0,hatches=[2*density*'/',4*density*'/'], transform = ccrs.PlateCarree())
        
    axs[j,1].set_ylim(-40, 40)
    axs[j,1].coastlines()
    
    sm, tick_locs, tick_labels = colorbar_contour(levels, 'Blues', 4, 12, 1)

axs[0,0].text(-0.08, 0.5, '$2^oC$ GWL\nDJF',
        transform=axs[0,0].transAxes,
        rotation=90, va='center', ha='center', fontsize=20)

axs[1,0].text(-0.08, 0.5, '$4^oC$ GWL\nDJF',
        transform=axs[1,0].transAxes,
        rotation=90, va='center', ha='center', fontsize=20)
axs[2,0].text(-0.08, 0.5, '$2^oC$ GWL\nJJA',
        transform=axs[2,0].transAxes,
        rotation=90, va='center', ha='center', fontsize=20)

axs[3,0].text(-0.08, 0.5, '$4^oC$ GWL\nJJA',
        transform=axs[3,0].transAxes,
        rotation=90, va='center', ha='center', fontsize=20)
cbar2 = fig.colorbar(
    c1,
    ax=axs,
    orientation='horizontal',
    fraction=0.05,
    pad=0.08,
    aspect=50
)
cbar2.set_label('Precipitation difference (mm $d^{-1}$)',fontsize=20)
cbar2.ax.tick_params(labelsize=17)

axs[0,0].set_title('Early adjustment', fontsize=25)
axs[0,1].set_title('Late adjustment', fontsize=25)

plt.suptitle("", fontsize=20)
#plt.tight_layout()
plt.savefig(os.path.join(out_path,out_name))








##### 100 years #####
out_name = "Precipitation_changes_contributions_pathway_100y.png"

file_early = os.path.join(data_path,'pr_changes_pathways_early_adjustment_100y.nc')
file_late = os.path.join(data_path,'pr_changes_pathways_late_adjustment_100y.nc')

da_early = xr.open_dataset(file_early)
da_late = xr.open_dataset(file_late)

da_early = da_early['pr']
da_late = da_late['pr']



seasons_series = ['DJF', 'DJF', 'JJA', 'JJA']
gwl_series = [2,4,2,4]

change_list_1 = []
change_list_2 = []

fig, axs = plt.subplots(
    4, 2, figsize=(19, 14), constrained_layout=True, 
    subplot_kw=dict(projection=ccrs.PlateCarree())
)
for j, (season_n, gwl) in enumerate(zip(seasons_series,gwl_series)):
    print(gwl, season_n)

    change = da_early.sel(season=season_n, gwl=gwl)
    pi_periods_season = da_pr_pi_periods.sel(season=season_n)
    snr = get_snr(change, pi_periods_season)  


    levels = np.arange(4, 16, 4)
    levels_diff = np.arange(-0.5, 0.6, 0.1) #[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]


    density=2
    c1 = axs[j,0].contourf(change.lon.values,change.lat.values, change.mean(dim='model'),
                            levels=levels_diff,cmap = 'BrBG', alpha = 0.85,
                            transform = ccrs.PlateCarree(),extend='both')
    '''c2 = axs[j,0].contour(mean.lon.values, mean.lat.values, mean.mean(dim='model'),
                            levels=levels,cmap = 'Blues', alpha = 0.95, linewidths=3,
                            transform = ccrs.PlateCarree())'''
    hatch_model_agreement_sign, _ = get_hatching_mask(change, 5, agreement=True)
    hatch_model_agreement_robust, _ = get_hatching_mask(snr, 4, agreement=True)
    hatch = hatch_model_agreement_sign * hatch_model_agreement_robust
    cs2_r = axs[j,0].contourf(hatch.lon,hatch.lat,hatch,levels=[0.5,1.5],colors='none', alpha=0,hatches=[2*density*'/',4*density*'/'], transform = ccrs.PlateCarree())
    
    axs[j,0].set_ylim(-40, 40)
    axs[j,0].coastlines()

    change = da_late.sel(season=season_n, gwl=gwl)
    snr = get_snr(change, pi_periods_season)      

    
    c1 = axs[j,1].contourf(change.lon.values,change.lat.values, change.mean(dim='model'),
                            levels=levels_diff,cmap = 'BrBG', alpha = 0.85,
                            transform = ccrs.PlateCarree(),extend='both')
    '''c2 = axs[j,1].contour(mean.lon.values, mean.lat.values, mean.mean(dim='model'),
                            levels=levels,cmap = 'Blues', alpha = 0.95, linewidths=3,
                            transform = ccrs.PlateCarree())'''
    hatch_model_agreement_sign, _ = get_hatching_mask(change, 5, agreement=True)
    hatch_model_agreement_robust, _ = get_hatching_mask(snr, 4, agreement=True)
    hatch = hatch_model_agreement_sign * hatch_model_agreement_robust
    cs2_r = axs[j,1].contourf(hatch.lon,hatch.lat,hatch,levels=[0.5,1.5],colors='none', alpha=0,hatches=[2*density*'/',4*density*'/'], transform = ccrs.PlateCarree())
        
    axs[j,1].set_ylim(-40, 40)
    axs[j,1].coastlines()
    
    sm, tick_locs, tick_labels = colorbar_contour(levels, 'Blues', 4, 12, 1)

axs[0,0].text(-0.08, 0.5, '$2^oC$ GWL\nDJF',
        transform=axs[0,0].transAxes,
        rotation=90, va='center', ha='center', fontsize=20)

axs[1,0].text(-0.08, 0.5, '$4^oC$ GWL\nDJF',
        transform=axs[1,0].transAxes,
        rotation=90, va='center', ha='center', fontsize=20)
axs[2,0].text(-0.08, 0.5, '$2^oC$ GWL\nJJA',
        transform=axs[2,0].transAxes,
        rotation=90, va='center', ha='center', fontsize=20)

axs[3,0].text(-0.08, 0.5, '$4^oC$ GWL\nJJA',
        transform=axs[3,0].transAxes,
        rotation=90, va='center', ha='center', fontsize=20)
cbar2 = fig.colorbar(
    c1,
    ax=axs,
    orientation='horizontal',
    fraction=0.05,
    pad=0.08,
    aspect=50
)
cbar2.set_label('Precipitation difference (mm $d^{-1}$)',fontsize=20)
cbar2.ax.tick_params(labelsize=17)

axs[0,0].set_title('Early adjustment', fontsize=25)
axs[0,1].set_title('Late adjustment', fontsize=25)

plt.suptitle("", fontsize=20)
#plt.tight_layout()
plt.savefig(os.path.join(out_path,out_name))
