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

file_pi_data_periods = os.path.join(data_path,'moisture_fluxes_picontrol_periods.nc')

da_pi_periods = xr.open_dataset(file_pi_data_periods)
da_pi_periods = da_pi_periods['waphus']


file = os.path.join(data_path,'moisture_fluxes_changes_fig.nc')
name_of_plot = 'Moisture_mass_flux_change_multimodel_picontrol_gwl2p0_gwl4p0'

da = xr.open_dataset(file)
da = da['waphus']

seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
labels = ['$2\ {}^oC$ GWL', '$4\ {}^oC$ GWL']
g = 9.81  # gravity in m/s^2
bottom_level = 1000
top_level = 100

fig, axes = plt.subplots(2,3,figsize=(25,15),
                                gridspec_kw={'width_ratios': [15,15, 2]},  # Ensure equal heights
                                )
#fig.subplots_adjust(hspace=0.2)
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
    print(season_n)
    ####### Comparison piControl model
    print('waphus')

    change_waphus_multimodel1 = da.sel(GWL='gwl2_change', season = season_n)
    change_waphus_multimodel2 = da.sel(GWL='gwl4_change', season = season_n)
    multimodel_waphus_comp = da.sel(GWL='pi', season = season_n)

    da_pi_periods_season = da_pi_periods.sel(season=season_n)
    snr_waphus1 = get_snr(change_waphus_multimodel1, da_pi_periods_season)
    snr_waphus2 = get_snr(change_waphus_multimodel2, da_pi_periods_season)

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

    hatch_model_agreement_sign, mean_change = get_hatching_mask(change_waphus_multimodel1, 7, agreement=True)
    hatch_model_agreement_robust, _ = get_hatching_mask(snr_waphus1, 5, agreement=True)
    hatch = hatch_model_agreement_sign * hatch_model_agreement_robust
    #hatch = reverse_binary_array(hatch)

    density=1
    ax1.contourf(hatch.lat,hatch.level,hatch,levels=[0.5,1.5],
                    colors='none', alpha=0,hatches=[density*'/',2*density*'/'], 
                    label='Agreement')
    
    if season_n == 'JFM' or season_n == 'DJF':
        ax1.set_ylabel('{}\nPressure (hPa)'.format(labels[0]), fontsize=25)

    #ax1.set_xlabel('Latitude', fontsize=15)
    ax1.tick_params(axis='x', which='major', labelsize=20)
    ax1.tick_params(axis='y', which='major', labelsize=20)

    ax1.set_title('{}'.format(season_n), fontsize=25)

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

    #ax2.set_xlabel('Latitude', fontsize=15)
    ax2.tick_params(axis='x', which='major', labelsize=20)
    ax2.tick_params(axis='y', which='major', labelsize=20)

    #ax2.set_title('Vertical Moisture Flux', fontsize=25)
ax13.set_axis_off()
ax23.set_axis_off()

fig.subplots_adjust(right=0.9)
ax_cb11 = fig.add_axes([0.92, 0.55, 0.015, 0.4])
ax_cb12 = fig.add_axes([0.92, 0.05, 0.015, 0.4])

cbar11 = plt.colorbar(c11,ax=ax_cb11,fraction=0.8)
cbar11.ax.tick_params(labelsize=15)
cbar11.set_label('Moisture Transport Difference ($10^{-5}$ kg $m^{-2}s^{-1}$)',fontsize=20)
cbar11.ax.yaxis.get_offset_text().set_visible(False)

sm_ps, tick_locs_ps, tick_labels_ps = colorbar_contour(levels_wh, 'bwr_r', -6e-5, 6e-5, 1/1e-5)

cbar22 = plt.colorbar(sm_ps, ticks=tick_locs_ps, ax=ax_cb12, fraction=0.8)
cbar22.ax.set_yticklabels(tick_labels_ps)
cbar22.ax.tick_params(labelsize=15)
cbar22.set_label('Moisture Transport ($10^{-5}$ kg $m^{-2}s^{-1}$)',fontsize=20)
cbar22.ax.yaxis.get_offset_text().set_visible(False)


ax11.annotate(
    '(a)',
    xy=(0, 1.1), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize=25, verticalalignment='top', fontfamily='serif',
    bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
ax12.annotate(
    '(b)',
    xy=(0, 1.1), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize=25, verticalalignment='top', fontfamily='serif',
    bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
ax21.annotate(
    '(c)',
    xy=(0, 1.1), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize=25, verticalalignment='top', fontfamily='serif',
    bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
ax22.annotate(
    '(d)',
    xy=(0, 1.1), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize=25, verticalalignment='top', fontfamily='serif',
    bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))


for ax_cb in [ax_cb11, ax_cb12]:
    ax_cb.spines[['top', 'bottom', 'left', 'right']].set_visible(False)  # Remove axes borders
    ax_cb.set_xticks([])  # Remove x ticks
    ax_cb.set_yticks([])  # Remove y ticks
    ax_cb.tick_params(which='both', bottom=False, top=False, left=False, right=False)  # Remove tick marks
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(out_path,name_of_plot+'_World.pdf'), dpi=300)
