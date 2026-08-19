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


out_path = '/path/to/output/directory/'   ####### Path where the output figures and csv files will be deposited
input_path = '/path/to/input/data/'       ####### Path where the input data (reduced datasets) are

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


def plot_s1():
    ############################################
    ###### Plot S1 (Precipitation changes) ######
    ############################################
    file = os.path.join(input_path,'suuplementary_precipitation_changes_fig.nc')
    out_name =  "Precipitation_changes_multi_model_mean.pdf"
    
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
        
        snr1 = da.sel(GWL='snr2', season=season_n)
        snr2 = da.sel(GWL='snr4', season=season_n)
        
    
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



def plot_s2():
    ##############################################################
    ###### Plot S2 (Local Pressure velocity changes)       ######
    ##############################################################
    file4 = os.path.join(input_path,'supplementary_wap_area_changes_fig_esm-up2p0-gwl4p0.nc')

    da4 = xr.open_dataset(file4)
    
    da4 = da4['wap']
    
    regions = {'America':[-90,-40], 'Atlantic':[-40,-10],
               'Africa':[-10,40],'Indian Ocean':[40,120],
               'Western Pacific':[120,170], 'Central Pacific':[170,230], 
               'Eastern Pacific':[230,270]}
    seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    

    bottom_level = 1000
    top_level = 100
    
    for da, run_gwl in zip([da4], ['gwl4p0']):
        name_of_plot = 'MMA_omega_change_regions_pi_{}'.format(run_gwl)
    
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
                snr_wap = da.sel(GWL='snr', season=season_n, area=area)
    
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
    
                hatch_model_agreement_sign, mean_change = get_hatching_mask(change_wap_multimodel, 5, agreement=True)
                hatch_model_agreement_robust, _ = get_hatching_mask(snr_wap, 4, agreement=True)
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

def plot_s3():
    ##############################################################
    ###### Plot S3 (Zonal Streamfunction changes)          ######
    ##############################################################
    file4 = os.path.join(input_path,'zonal_psi_changes_fig_esm-up2p0-gwl4p0.nc')
    da4_psi = xr.open_dataset(file4)
    da4_psi = da4_psi['psi']


    file4 = os.path.join(input_path,'zonal_pr_changes_fig_esm-up2p0-gwl4p0.nc')
    da4_pr = xr.open_dataset(file4)
    da4_pr = da4_pr['pr']
    

    file4 = os.path.join(input_path,'zonal_wap_changes_fig_esm-up2p0-gwl4p0.nc')
    da4_wap = xr.open_dataset(file4)
    da4_wap = da4_wap['wap']
    
    seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    bottom_level = 1000
    top_level = 100
    
     
    for da_psi, da_wap, da_pr, run_gwl in zip([da4_psi], [da4_wap], [da4_pr], ['gwl4p0']):
        name_of_plot = 'Streamfunction_change_omega_{}'.format('multimodel_{}_{}'.format('pictontrol', run_gwl))
       
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
            snr_psi = da_psi.sel(GWL = 'snr', season = season_n)
            print(snr_psi.values)
            
            change_wap_multimodel = da_wap.sel(GWL = 'change', season = season_n)
            multimodel_wap_comp = da_wap.sel(GWL = 'pi', season = season_n)
            multimodel_wap_idea = da_wap.sel(GWL = 'gwl', season = season_n)
            snr_wap = da_wap.sel(GWL = 'snr', season = season_n)
            
            change_pr_multimodel = da_pr.sel(GWL = 'change', season = season_n)
            #multimodel_pr_comp = da_pr.sel(GWL = 'pi', season = season_n)
            #multimodel_pr_idea = da_pr.sel(GWL = 'gwl', season = season_n)
            
    
    
            df_season = simple_metrics_models(multimodel_psi_comp, multimodel_psi_idea, multimodel_wap_comp, multimodel_wap_idea, season_n)
            dfs_to_combine.append(df_season)
    
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
            hatch_model_agreement_sign, mean_change = get_hatching_mask(change_psi_multimodel, 5, agreement=True)
            hatch_model_agreement_robust, _ = get_hatching_mask(snr_psi, 4, agreement=True)
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
            
            hatch_model_agreement_sign, mean_change = get_hatching_mask(change_wap_multimodel, 5, agreement=True)
            hatch_model_agreement_robust, _ = get_hatching_mask(snr_wap, 4, agreement=True)
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
        df_final = pd.concat(dfs_to_combine, ignore_index=True)
        df_final.to_csv(os.path.join(out_path, 'Metrics_criculation_models_{}.csv'.format('multimodel_{}_{}'.format('pictontrol', run_gwl))))
    

    
plot_s1()
plot_s2()
plot_s3()

