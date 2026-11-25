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


############################################
######     (Precipitation changes)    ######
############################################
file = os.path.join(data_path,'supplementary_precipitation_changes_fig.nc')
pi_control_prs = xr.open_dataset(os.path.join(data_path,'precipitation_picontrol_periods.nc'))
out_name =  "Supplementary_Precipitation_changes_multi_model_mean.pdf"

da = xr.open_dataset(file)
da = da['pr']
#print(da)
seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
labels = ['$2^oC$ GWL', '$4^oC$ GWL']
fig, axs = plt.subplots(
    2, len(seasons.keys()), figsize=(17, 9), constrained_layout=True, 
    subplot_kw=dict(projection=ccrs.Mollweide())
)
for j, season_n in enumerate(seasons.keys()):
    change_multimodel1 = da.sel(GWL='gwl2_change', season=season_n)
    change_multimodel2 = da.sel(GWL='gwl4_change', season=season_n)
    
    pi_periods_season = pi_control_prs['pr'].sel(season=season_n)
    snr1 = get_snr(change_multimodel1, pi_periods_season)
    snr2 = get_snr(change_multimodel2, pi_periods_season)


    ### Hatching where all 7/7 models agree on the sign of change and at least 5/7 are robust (SNR>1) ###
    hatch_model_agreement_sign_1, mean_change_1 = get_hatching_mask(change_multimodel1, 7, agreement=True)
    hatch_model_agreement_robust_1, _ = get_hatching_mask(snr1, 5, agreement=True)
    hatch_1 = hatch_model_agreement_sign_1 * hatch_model_agreement_robust_1

    levels = np.arange(-1.6,1.8,0.2)
    c1 = axs[0,j].contourf(mean_change_1.lon.values,mean_change_1.lat.values, mean_change_1,
                                vmin=-1.6,vmax=1.6,levels=levels,cmap = 'BrBG', alpha = 0.85,
                                transform = ccrs.PlateCarree(),extend='both')
    axs[0,j].coastlines()
    density=2
    cs2_r = axs[0,j].contourf(hatch_1.lon,hatch_1.lat,hatch_1,levels=[0.5,1.5],colors='none', alpha=0,hatches=[2*density*'/',4*density*'/'], transform = ccrs.PlateCarree())
    axs[0,j].set_title(season_n, fontsize=25)

    hatch_model_agreement_sign_2, mean_change_2 = get_hatching_mask(change_multimodel2, 7, agreement=True)
    hatch_model_agreement_robust_2, _ = get_hatching_mask(snr2, 5, agreement=True)
    hatch_2 = hatch_model_agreement_sign_2 * hatch_model_agreement_robust_2

    c1 = axs[1,j].contourf(mean_change_2.lon.values,mean_change_2.lat.values, mean_change_2,
                                vmin=-1.6,vmax=1.6,levels=levels,cmap = 'BrBG', alpha = 0.85,
                                transform = ccrs.PlateCarree(),extend='both')
    axs[1,j].coastlines()
    #cs2 = axs[1,j].contourf(hatch3_2.lon,hatch3_2.lat,hatch3_2,levels=[0.5,1.5],colors='none', alpha=0,hatches=[density*'/',2*density*'/'], label='3')
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






##############################################################
###### Plot S2 (Local Pressure velocity changes)       ######
##############################################################
file4 = os.path.join(data_path,'supplementary_wap_area_changes_fig_esm-up2p0-gwl4p0.nc')
file_pi_data_periods = os.path.join(data_path,'wap_area_picontrol_periods.nc')
da4 = xr.open_dataset(file4)
pi_data_periods = xr.open_dataset(file_pi_data_periods)

da4 = da4['wap']
pi_data_periods = pi_data_periods['wap']

regions = {'America':[-90,-40], 'Atlantic':[-40,-10],
            'Africa':[-10,40],'Indian Ocean':[40,120],
            'Western Pacific':[120,170], 'Central Pacific':[170,230], 
            'Eastern Pacific':[230,270]}
seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}


bottom_level = 1000
top_level = 100

for da, run_gwl in zip([da4], ['gwl4p0']):
    name_of_plot = 'Supplemetary_MMA_omega_change_regions_pi_{}'.format(run_gwl)

    fig, axes = plt.subplots(7,3, figsize=(25,30),
                                    gridspec_kw={'width_ratios': [15,15,3]},  # Ensure equal heights
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
    #plt.show()
    plt.savefig(os.path.join(out_path,name_of_plot+'.pdf'), dpi=300)





##############################################################
###### Plot S3 (Zonal Streamfunction changes)          ######
##############################################################
file_psi_pi_data_periods = os.path.join(data_path,'global_zonal_psi_picontrol_periods.nc')
file_wap_pi_data_periods = os.path.join(data_path,'global_zonal_wap_picontrol_periods.nc')

da_psi_pi_periods = xr.open_dataset(file_psi_pi_data_periods)
da_wap_pi_periods = xr.open_dataset(file_wap_pi_data_periods)

da_psi_pi_periods = da_psi_pi_periods['psi']
da_wap_pi_periods = da_wap_pi_periods['wap']


file4 = os.path.join(data_path,'supplementary_zonal_psi_changes_fig_esm-up2p0-gwl4p0.nc')
da4_psi = xr.open_dataset(file4)
da4_psi = da4_psi['psi']


file4 = os.path.join(data_path,'supplementary_zonal_pr_changes_fig_esm-up2p0-gwl4p0.nc')
da4_pr = xr.open_dataset(file4)
da4_pr = da4_pr['pr']


file4 = os.path.join(data_path,'supplementary_zonal_wap_changes_fig_esm-up2p0-gwl4p0.nc')
da4_wap = xr.open_dataset(file4)
da4_wap = da4_wap['wap']

seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
bottom_level = 1000
top_level = 100

    
for da_psi, da_wap, da_pr, run_gwl in zip([da4_psi], [da4_wap], [da4_pr], ['gwl4p0']):
    name_of_plot = 'Supplementary_Streamfunction_change_omega_{}'.format('multimodel_{}_{}'.format('pictontrol', run_gwl))
    
    fig, axes = plt.subplots(3,4,figsize=(30,18),
                                    gridspec_kw={'width_ratios': [15,15, 1, 1]},  # Ensure equal heights
                                    )
    #fig.subplots_adjust(hspace=0.2)
    # Unpack the axes
    ax11, ax12, ax_cb11, ax_cb12 = axes[0]
    ax21, ax22, ax_cb21, ax_cb22 = axes[1]
    ax31, ax32, ax_cb31, ax_cb32 = axes[2]  # No colorbar for the third plot, so leave it empty
    dfs_to_combine = []

    for season_n in list(seasons.keys()):
        if season_n == 'JFM' or season_n == 'DJF':
            ax1 = ax11
            ax2 = ax21
            ax3 = ax31
        else:
            ax1 = ax12
            ax2 = ax22
            ax3 = ax32

        ####### Comparison piControl model
        print('psi')

        change_psi_multimodel = da_psi.sel(GWL = 'change', season = season_n)
        multimodel_psi_comp = da_psi.sel(GWL = 'pi', season = season_n)
        multimodel_psi_idea = da_psi.sel(GWL = 'gwl', season = season_n)

        pi_psi_periods_season = da_psi_pi_periods.sel(season=season_n)
        snr_psi = get_snr(change_psi_multimodel, pi_psi_periods_season)
        
        change_wap_multimodel = da_wap.sel(GWL = 'change', season = season_n)
        multimodel_wap_comp = da_wap.sel(GWL = 'pi', season = season_n)
        multimodel_wap_idea = da_wap.sel(GWL = 'gwl', season = season_n)

        pi_wap_periods_season = da_wap_pi_periods.sel(season=season_n)
        snr_wap = get_snr(change_wap_multimodel, pi_wap_periods_season)
        
        change_pr_multimodel = da_pr.sel(GWL = 'change', season = season_n)


        plot_in_contours_psi = multimodel_psi_comp
        plot_in_contours_wap = multimodel_wap_comp

        levels_ps=np.arange(-1.2e11, 1.6e11, 4e10)
        levels2_ps=np.arange(-3e10, 3.5e10, 0.5e10)
        
        c11 = ax1.contourf(change_psi_multimodel.lat, change_psi_multimodel.level, change_psi_multimodel.mean(dim='model').values, 
                        levels=levels2_ps, cmap='BrBG_r', extend='both', alpha=0.7)
        

        ax1.set_ylim(bottom_level,top_level)

        ax1.contour(plot_in_contours_psi.lat, plot_in_contours_psi.level, plot_in_contours_psi.mean(dim='model').values, 
                    levels=levels_ps, vmin=-1.4e11, vmax= 1.4e11, colors='black', linewidths=4, linestyles='-')
        
        ax1.contour(plot_in_contours_psi.lat, plot_in_contours_psi.level, plot_in_contours_psi.mean(dim='model').values, 
                    levels=levels_ps, vmin=-1.4e11, vmax= 1.4e11, cmap='bwr', linewidths=3)
        print(np.max(np.abs(plot_in_contours_psi.mean(dim='model'))))
        hatch_model_agreement_sign, mean_change = get_hatching_mask(change_psi_multimodel, 7, agreement=True)
        hatch_model_agreement_robust, _ = get_hatching_mask(snr_psi, 5, agreement=True)
        hatch = hatch_model_agreement_sign * hatch_model_agreement_robust
    
        density=1
        ax1.contourf(hatch.lat,hatch.level,hatch,levels=[0.5,1.5],
                        colors='none', alpha=0,hatches=[density*'/',2*density*'/'], 
                        label='Agreement')
        

        ax1.set_ylabel('Pressure (hPa)', fontsize=25)
        #ax1.set_xlabel('Latitude', fontsize=15)
        ax1.tick_params(axis='x', which='major', labelsize=20)
        ax1.tick_params(axis='y', which='major', labelsize=20)

        ax1.set_title('{}\n Mass Streamfunction'.format(season_n), fontsize=25)
        ax2.set_title('Pressure Velocity', fontsize=25)
        ax3.set_title('Precipitation Change', fontsize=25)

        #####################
        #####################
        #####################

        levels2_wap = np.arange(-0.008,0.010,0.002)
        levels_wap = np.arange(-0.05,0.06,0.01)

        c22 = ax2.contourf(change_wap_multimodel.lat, change_wap_multimodel.level, change_wap_multimodel.mean(dim='model').values,
                        vmin=-0.008,vmax=0.008,cmap='BrBG_r',levels=levels2_wap,extend='both', alpha=0.7)
        ax2.set_ylim(bottom_level,top_level)

        ax2.contour(plot_in_contours_wap.lat, plot_in_contours_wap.level, plot_in_contours_wap.mean(dim='model').values,
                    vmin=-0.05,vmax=0.05,colors='black',levels=levels_wap, linewidths=4, linestyles='-')
        ax2.contour(plot_in_contours_wap.lat, plot_in_contours_wap.level, plot_in_contours_wap.mean(dim='model').values,
                    vmin=-0.06,vmax=0.06,cmap='bwr_r',levels=levels_wap, linewidths=3, linestyles='-')
        
        hatch_model_agreement_sign, mean_change = get_hatching_mask(change_wap_multimodel, 7, agreement=True)
        hatch_model_agreement_robust, _ = get_hatching_mask(snr_wap, 5, agreement=True)
        hatch = hatch_model_agreement_sign * hatch_model_agreement_robust

        density=1
        ax2.contourf(hatch.lat,hatch.level,hatch,levels=[0.5,1.5],colors='none', alpha=0,hatches=[density*'/',2*density*'/'], label='Strengthening')
        
        ax2.set_ylabel('Pressure (hPa)', fontsize=25)
        
        ax2.tick_params(axis='x', which='major', labelsize=20)
        ax2.tick_params(axis='y', which='major', labelsize=20)
        

        ax3.plot(change_pr_multimodel.lat.values, change_pr_multimodel.mean(dim='model').values,color='black')
        ax3.set_ylim(-2,2)
        ax3.set_xlim(np.min(change_pr_multimodel.lat.values),np.max(change_pr_multimodel.lat.values))
        ax3.axhline(y=0, color='grey', linestyle='-',alpha=0.7)
        # Fill the area above the line with blue
        ax3.fill_between(change_pr_multimodel.lat.values, change_pr_multimodel.mean(dim='model').values, 0, where=change_pr_multimodel.mean(dim='model').values >= 0, color='blue', alpha=0.5)
        
        # Fill the area below the line with red
        ax3.fill_between(change_pr_multimodel.lat.values, change_pr_multimodel.mean(dim='model').values, 0, where=change_pr_multimodel.mean(dim='model').values < 0, color='red', alpha=0.5)
        ax3.tick_params(axis='x', which='major', labelsize=20)
        ax3.tick_params(axis='y', which='major', labelsize=20)
        ax3.set_ylabel('Precipitation difference (mm)', fontsize=25)
        ax3.set_xlabel('Latitude', fontsize=25)
    

    cbar11 = plt.colorbar(c11,ax=ax_cb11,fraction=0.95)
    cbar11.ax.tick_params(labelsize=15)
    cbar11.set_label('Mass Streamfunction Difference ($10^{10}$ kg $s^{-1}$)',fontsize=20)
    cbar11.ax.yaxis.get_offset_text().set_visible(False)
    
    sm_ps, tick_locs_ps, tick_labels_ps = colorbar_contour(levels_ps, 'bwr', -1.4e11, 1.4e11, 1/1e11)

    cbar12 = plt.colorbar(sm_ps, ticks=tick_locs_ps, ax=ax_cb12, fraction=0.95)
    cbar12.ax.set_yticklabels(tick_labels_ps)
    cbar12.ax.tick_params(labelsize=15)
    cbar12.set_label('Mass Streamfunction ($10^{11}$ kg $s^{-1}$)',fontsize=20)
    cbar12.ax.yaxis.get_offset_text().set_visible(False)

    cbar21 = plt.colorbar(c22,ax=ax_cb21,fraction=0.95)
    cbar21.ax.tick_params(labelsize=15)
    cbar21.set_label('Pressure Velocity Difference (Pa $s^{-1}$)',fontsize=20)
    cbar21.ax.yaxis.get_offset_text().set_visible(False)
    
    sm_wap, tick_locs_wap, tick_labels_wap = colorbar_contour(levels_wap, 'bwr_r', -0.06, 0.06, 1)

    cbar22 = plt.colorbar(sm_wap, ticks=tick_locs_wap, ax=ax_cb22, fraction=0.95)
    cbar22.ax.set_yticklabels(tick_labels_wap)
    cbar22.ax.tick_params(labelsize=15)
    cbar22.set_label('Pressure Velocity (Pa $s^{-1}$)',fontsize=20)
    cbar22.ax.yaxis.get_offset_text().set_visible(False)
    
    ax11.annotate(
        '(a)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax12.annotate(
        '(b)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax21.annotate(
        '(c)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax22.annotate(
        '(d)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax31.annotate(
        '(e)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax32.annotate(
        '(f)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    
    for ax_cb in [ax_cb11, ax_cb21, ax_cb31, ax_cb12, ax_cb22, ax_cb32]:
        ax_cb.spines[['top', 'bottom', 'left', 'right']].set_visible(False)  # Remove axes borders
        ax_cb.set_xticks([])  # Remove x ticks
        ax_cb.set_yticks([])  # Remove y ticks
        ax_cb.tick_params(which='both', bottom=False, top=False, left=False, right=False)  # Remove tick marks
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(out_path,name_of_plot+'_World.pdf'), dpi=300)
