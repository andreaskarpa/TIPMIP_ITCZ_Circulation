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

file2 = os.path.join(data_path,'wap_area_changes_fig_esm-up2p0-gwl2p0.nc')
file4 = os.path.join(data_path,'wap_area_changes_fig_esm-up2p0-gwl4p0.nc')
file_pi_data_periods = os.path.join(data_path,'wap_area_picontrol_periods.nc')

da2 = xr.open_dataset(file2)
da4 = xr.open_dataset(file4)
pi_data_periods = xr.open_dataset(file_pi_data_periods)

da2 = da2['wap']
da4 = da4['wap']
pi_data_periods = pi_data_periods['wap']

regions = {'America':[-90,-40], 'Atlantic':[-40,-10],
            'Africa':[-10,40],'Indian Ocean':[40,120],
            'Western Pacific':[120,170], 'Central Pacific':[170,230], 
            'Eastern Pacific':[230,270]}
seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}


bottom_level = 1000
top_level = 100

for da, run_gwl in zip([da2,da4], ['gwl2p0', 'gwl4p0']):
    name_of_plot = 'MMA_omega_change_regions_pi_{}'.format(run_gwl)

    fig, axes = plt.subplots(7,3, figsize=(30,30),
                                    gridspec_kw={'width_ratios': [15,15,2]},  # Ensure equal heights
                                    )
    for k, area in enumerate(list(regions.keys())): 
        ax11, ax12, ax_empty = axes[k]  


        for season_n in list(seasons.keys()):
            if season_n == 'DJF':
                ax1 = ax11
            else:
                ax1 = ax12
            print(area, season_n)
            ### wap
            ####### Comparison piControl model

            change_wap_multimodel = da.sel(GWL='change', season=season_n, area=area)
            multimodel_wap_comp = da.sel(GWL='pi', season=season_n, area=area)
            pi_periods_season = pi_data_periods.sel(season=season_n, area=area)
            
            snr_wap = get_snr(change_wap_multimodel, pi_periods_season)

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

            hatch_model_agreement_sign, mean_change = get_hatching_mask(change_wap_multimodel, 7, agreement=True)
            hatch_model_agreement_robust, _ = get_hatching_mask(snr_wap, 5, agreement=True)
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
    ax_cb11 = fig.add_axes([0.92, 0.55, 0.015, 0.4])
    ax_cb12 = fig.add_axes([0.92, 0.05, 0.015, 0.4])
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
    #plt.show()
    plt.savefig(os.path.join(out_path,name_of_plot+'.pdf'), dpi=300)
