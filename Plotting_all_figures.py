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






def plot1():
    ############################################
    ###### Plot 1 (Precipitation changes) ######
    ############################################
    file = os.path.join(input_path,'precipitation_changes_fig.nc')
    out_name =  "Precipitation_changes_multi_model_mean.pdf"
    
    da = xr.open_dataset(file)
    da = da['pr']
    #print(da)
    seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    labels = ['$2\ {}^oC$ GWL', '$4\ {}^oC$ GWL']
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
    
        levels = np.arange(-2,2.2,0.2)
        c1 = axs[0,j].contourf(mean_change_1.lon.values,mean_change_1.lat.values, mean_change_1,
                                 vmin=-2,vmax=2,levels=levels,cmap = 'BrBG', alpha = 0.85,
                                 transform = ccrs.PlateCarree(),extend='both')
        axs[0,j].coastlines()
        density=2
        #cs2 = axs[0,j].contourf(hatch3_1.lon,hatch3_1.lat,hatch3_1,levels=[0.5,1.5],colors='none', alpha=0,hatches=[density*'/',2*density*'/'], label='3')
        axs[0,j].contourf(hatch_1.lon,hatch_1.lat,hatch_1,levels=[0.5,1.5],colors='none', alpha=0,hatches=[2*density*'/',4*density*'/'], transform = ccrs.PlateCarree())
        axs[0,j].set_title(season_n, fontsize=25)
    
        hatch_model_agreement_sign_2, mean_change_2 = get_hatching_mask(change_multimodel2, 5, agreement=True)
        hatch_model_agreement_robust_2, _ = get_hatching_mask(snr2, 4, agreement=True)
        hatch_2 = hatch_model_agreement_sign_2 * hatch_model_agreement_robust_2
    
        c1 = axs[1,j].contourf(mean_change_2.lon.values,mean_change_2.lat.values, mean_change_2,
                                 vmin=-2,vmax=2,levels=levels,cmap = 'BrBG', alpha = 0.85,
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


def plot2():
    ############################################
    ###### Plot 2 (ITCZ changes)          ######
    ############################################
    file = os.path.join(input_path,'itcz_characteristics_changes.csv')
    df = pd.read_csv(file)
    regions = {'America':[-90,-40], 'Atlantic':[-40,-10],
                    'Africa':[-10,40],'Indian Ocean':[40,120],
                    'Western Pacific':[120,170], 'Central Pacific':[170,230], 
                    'Eastern Pacific':[230,270], 'World':[0,360]}
    seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    out_name =  "ITCZ_loc_edges_multi_model_mean_boxplots.pdf"
    labels = ['$2\ {}^oC$', '$4\ {}^oC$']
    
    fig, axs = plt.subplots(
        5, len(regions.keys()), figsize=(18, 19),
        constrained_layout=True, gridspec_kw={'height_ratios': [1, 1, 1 ,1, 1]}
    )
    offset = 0.1 
    box_width = 0.1 
    # List to collect rows across all iterations
    
    for j, region in enumerate(regions.keys()):
        for i, season_n in enumerate(seasons.keys()):
            if season_n == 'JFM' or season_n == 'DJF':
                c= 'blue'
                # Shift the base positions [1, 2] to the left
                pos = [1 - offset, 2 - offset]
            else:
                c = 'red'
                # Shift the base positions [1, 2] to the right
                pos = [1 + offset, 2 + offset] 
            
            loc_list1 = list(df[(df['region']==region)&(df['season']==season_n)]['loc_idealized1'])
            loc_list2 = list(df[(df['region']==region)&(df['season']==season_n)]['loc_idealized2'])
            
            latmax_list1 = list(df[(df['region']==region)&(df['season']==season_n)]['latmax_idealized1'])
            latmax_list2 = list(df[(df['region']==region)&(df['season']==season_n)]['latmax_idealized2'])
            
            latmin_list1 = list(df[(df['region']==region)&(df['season']==season_n)]['latmin_idealized1'])
            latmin_list2 = list(df[(df['region']==region)&(df['season']==season_n)]['latmin_idealized2'])
            
            width_list1 = list(df[(df['region']==region)&(df['season']==season_n)]['width_idealized1'])
            width_list2 = list(df[(df['region']==region)&(df['season']==season_n)]['width_idealized2'])
            
            strength_list1 = list(df[(df['region']==region)&(df['season']==season_n)]['strength_idealized1'])
            strength_list2 = list(df[(df['region']==region)&(df['season']==season_n)]['strength_idealized2'])
            
    
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
    
            axs[3, j].boxplot(width_list1, positions=[pos[0]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c), label=season_n)
            axs[3, j].boxplot(width_list2, positions=[pos[1]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c))

            axs[4, j].boxplot(strength_list1, positions=[pos[0]], widths=box_width, showfliers=False,
                                    boxprops=dict(color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    medianprops=dict(color=c), label=season_n)
            axs[4, j].boxplot(strength_list2, positions=[pos[1]], widths=box_width, showfliers=False,
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
        axs[3, j].set_xticklabels([])
        axs[3, j].set_xlim(0.7,2.3)
        axs[3, j].grid(visible=True, which='major',axis='y', color='grey', alpha = 0.5, linestyle='--')
        axs[3, j].xaxis.set_tick_params(which='both', labelbottom=True)
        axs[3, j].yaxis.set_tick_params(labelsize=13)
    
        axs[4, j].set_xticks([1,2],labels, fontsize=18)
        axs[4, j].set_xlim(0.7,2.3)
        axs[4, j].grid(visible=True, which='major',axis='y', color='grey', alpha = 0.5, linestyle='--')
        axs[4, j].xaxis.set_tick_params(which='both', labelbottom=True)
        axs[4, j].yaxis.set_tick_params(labelsize=13)

        axs[4, j].set_xlabel(region, fontsize=20)
    # Create final pandas DataFrame
    
    axs[0,0].set_ylabel('Northern edge \nlatitude difference (${}^o$)', fontsize=18)
    axs[1,0].set_ylabel('ITCZ location \nlatitude difference (${}^o$)', fontsize=18)
    axs[2,0].set_ylabel('Southern edge \nlatitude difference (${}^o$)', fontsize=18)
    axs[3,0].set_ylabel('Width \nlatitude difference (${}^o$)', fontsize=18)
    axs[4,0].set_ylabel('Strength difference \n(mm $d^{-1}$ ${}^o$)', fontsize=18)
    plt.suptitle("", fontsize=20)
    plt.savefig(os.path.join(out_path,out_name))


def plot34():
    ##############################################################
    ###### Plot 3-4 (Local Pressure velocity changes)       ######
    ##############################################################
    file2 = os.path.join(input_path,'wap_area_changes_fig_esm-up2p0-gwl2p0.nc')
    file4 = os.path.join(input_path,'wap_area_changes_fig_esm-up2p0-gwl4p0.nc')
    
    da2 = xr.open_dataset(file2)
    da4 = xr.open_dataset(file4)
    
    da2 = da2['wap']
    da4 = da4['wap']
    
    regions = {'America':[-90,-40], 'Atlantic':[-40,-10],
               'Africa':[-10,40],'Indian Ocean':[40,120],
               'Western Pacific':[120,170], 'Central Pacific':[170,230], 
               'Eastern Pacific':[230,270]}
    seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    

    bottom_level = 1000
    top_level = 100
    
    for da, run_gwl in zip([da2,da4], ['gwl2p0', 'gwl4p0']):
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

                levels2 = np.arange(-0.015,0.018,0.003)
                levels_simple = [-0.03,-0.01,0,0.01,0.03]
    
                
                #### The normal
                plot_in_contours = multimodel_wap_comp
                c12 = ax1.contourf(change_wap_multimodel.lat, change_wap_multimodel.level, change_wap_multimodel.mean(dim='model').values,
                                vmin=-0.015,vmax=0.015,cmap='BrBG_r',levels=levels2,extend='both', alpha=0.7)
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

def plot56():
    ##############################################################
    ###### Plot 5-6 (Zonal Streamfunction changes)          ######
    ##############################################################
    file2 = os.path.join(input_path,'zonal_psi_changes_fig_esm-up2p0-gwl2p0.nc')
    file4 = os.path.join(input_path,'zonal_psi_changes_fig_esm-up2p0-gwl4p0.nc')
    
    da2_psi = xr.open_dataset(file2)
    da4_psi = xr.open_dataset(file4)
    
    da2_psi = da2_psi['psi']
    da4_psi = da4_psi['psi']
    
    
    file2 = os.path.join(input_path,'zonal_pr_changes_fig_esm-up2p0-gwl2p0.nc')
    file4 = os.path.join(input_path,'zonal_pr_changes_fig_esm-up2p0-gwl4p0.nc')
    
    da2_pr = xr.open_dataset(file2)
    da4_pr = xr.open_dataset(file4)
    
    da2_pr = da2_pr['pr']
    da4_pr = da4_pr['pr']
    
    
    file2 = os.path.join(input_path,'zonal_wap_changes_fig_esm-up2p0-gwl2p0.nc')
    file4 = os.path.join(input_path,'zonal_wap_changes_fig_esm-up2p0-gwl4p0.nc')
    
    da2_wap = xr.open_dataset(file2)
    da4_wap = xr.open_dataset(file4)
    
    da2_wap = da2_wap['wap']
    da4_wap = da4_wap['wap']
    
    seasons = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    bottom_level = 1000
    top_level = 100
    
     
    for da_psi, da_wap, da_pr, run_gwl in zip([da2_psi, da4_psi], [da2_wap, da4_wap], [da2_pr, da4_pr], ['gwl2p0', 'gwl4p0']):
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
    

def plot7():
    ##############################################################
    ###### Plot 7 (Vertical Moisture Transport changes)     ######
    ##############################################################
    file = os.path.join(input_path,'moisture_fluxes_changes_fig.nc')
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
        snr_waphus1 = da.sel(GWL='snr2', season = season_n)
        snr_waphus2 = da.sel(GWL='snr4', season = season_n)

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

        hatch_model_agreement_sign, mean_change = get_hatching_mask(change_waphus_multimodel1, 5, agreement=True)
        hatch_model_agreement_robust, _ = get_hatching_mask(snr_waphus1, 4, agreement=True)
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
 
def plot8():
    ##############################################################
    ###### Plot 8 (ITCZ methodology evaluation)             ######
    ##############################################################
    file_csv = os.path.join(input_path,'observations_itcz_characteristics.csv')
    file = os.path.join(input_path,'Obervations_reanalysis_precipitation_fig.nc')
    
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
    
def plot9():
    ##############################################################
    ###### Plot 9-10-11 (TIPMIP Models evaluation)          ######
    ##############################################################
    file_double_itcz = os.path.join(input_path,'Double_itcz_indices_fig.csv')
    file_mspaef = os.path.join(input_path,'mspaef_values_fig.csv')
    file = os.path.join(input_path,'historical_precipitation_differences_fig.nc')
    
    df_double = pd.read_csv(file_double_itcz)
    df_mspaef = pd.read_csv(file_mspaef)
    da = xr.open_dataset(file)
    da = da['pr']
    
    naming_models = {'UKESM1-2' : 'UKESM1-2-LL', 'EC-Earth3' : 'EC-Earth3-ESM-1', 'IPSL' : 'IPSL-CM6-ESMCO2',
              'CNRM' : 'CNRM-ESM2-2', 'GFDL-ESM2M': 'GFDL-ESM2M', 'MIROC-ES2L' : 'MIROC-ES2L', 'NorESM2-LM': 'NorESM2-LM'}
    name_out =  "Precipitation_evaluation_OptimESM_{}.png"
    name_out_mspaef = "MSPAEF_Precipitation_evaluation_OptimESM.pdf"
    name_out_ditcz = "Double_ITCZ_evaluation.pdf"
    


    # Combine model and obs data into unified lists
    model_name_list = list(df_double[df_double['data_type']=='model']['dataset'])
    obs_name_list  = list(df_double[df_double['data_type']=='observational']['dataset'])
    
    list_values_epi_models = list(df_double[df_double['data_type']=='model']['EPI'])
    list_values_tpai_models = list(df_double[df_double['data_type']=='model']['TPAI'])
    
    list_values_epi_obs = list(df_double[df_double['data_type']=='observational']['EPI'])
    list_values_tpai_obs = list(df_double[df_double['data_type']=='observational']['TPAI'])

    
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
    plt.savefig(os.path.join(out_path,name_out_ditcz))
    
    
    ##### Plot MSPAEF heatmap #####

    array_mspaef_values = df_mspaef[['DJF','MAM',"JJA",'SON', 'Annual']].values
    seasons_name_list = list(df_mspaef.columns.values)[1:]
    model_name_list = list(df_mspaef['Unnamed: 0'])

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
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation='horizontal',
        fraction=0.05,
        pad=0.08
    )
    ax.set_title("MSPAEF values")
    fig.tight_layout()
    plt.savefig(os.path.join(out_path,name_out_mspaef))
    
    
    #### Precipitation differences evaluation #####
    seasons_dict = {'DJF':[12,1,2], 'JJA':[6,7,8], 'Annual': [1,2,3,4,5,6,7,8,9,10,11,12]}
    for season_n in seasons_dict.keys():
        print(season_n)
        fig, axs = plt.subplots(
            3, 3, figsize=(17, 9), constrained_layout=True, 
            subplot_kw=dict(projection=ccrs.Mollweide())
        )
        axs_flatten = axs.flatten()

        levels = np.arange(-5, 5.5, 0.5)
        model_name_list = list(da.model.values)
        for j, model_name in enumerate(model_name_list):
            ax = axs_flatten[j]
            
            difference = da.sel(model=model_name, season = season_n)
            c1 = ax.contourf(difference.lon.values, difference.lat.values, difference.values, levels=levels, cmap = 'BrBG',
                        transform = ccrs.PlateCarree(), extend='both')
            ax.coastlines()

            ax.set_title(naming_models[model_name], fontsize=25)


        for j in range(len(model_name_list), len(axs_flatten)):
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
        plt.savefig(os.path.join(out_path,name_out.format(season_n)))
    
def plot10():
    ##############################################################
    ###### Plot of TIPMIP Models evaluation (historical and rampup          ######
    ##############################################################
    file_rampup = os.path.join(input_path,'historical_rampup_precipitation_differences_fig.nc')
    file = os.path.join(input_path,'historical_precipitation_differences_fig.nc')
    
    da = xr.open_dataset(file)
    da = da['pr']
    
    da_rampup = xr.open_dataset(file_rampup)
    da_rampup = da_rampup['pr']
    
    naming_models = {'UKESM1-2' : 'UKESM1-2-LL', 'EC-Earth3' : 'EC-Earth3-ESM-1', 'IPSL' : 'IPSL-CM6-ESMCO2',
              'CNRM' : 'CNRM-ESM2-2', 'GFDL-ESM2M': 'GFDL-ESM2M', 'MIROC-ES2L' : 'MIROC-ES2L', 'NorESM2-LM': 'NorESM2-LM'}
    name_out =  "Precipitation_historical_rampup_evaluation_OptimESM.png"

    
    #### Precipitation differences evaluation #####
    seasons_dict = {'DJF':[12,1,2], 'JJA':[6,7,8]}
    fig, axs = plt.subplots(
        7, 4, figsize=(19, 17), constrained_layout=True, 
        subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    for i, season_n in enumerate(seasons_dict.keys()):
        print(season_n)
        levels = np.arange(-5, 5.5, 0.5)
        levels2 = np.arange(-3, 3.3, 0.3)
        
        axs[0,0 +i].set_title(season_n ,fontsize=20)
        axs[0,2 + i].set_title(season_n ,fontsize=20)
 
        model_name_list = list(da.model.values)
        for j, model_name in enumerate(model_name_list):
            ax = axs[j, 0 + i]
            ax1 = axs[j, 2 + i]
            difference = da.sel(model=model_name, season = season_n)
            difference_idea = da_rampup.sel(model=model_name, season = season_n)
            
            c = ax.contourf(difference.lon.values, difference.lat.values, difference.values, levels=levels, cmap = 'BrBG',
                        transform = ccrs.PlateCarree(), extend='both')
            ax.coastlines()

            c1 = ax1.contourf(difference_idea.lon.values, difference_idea.lat.values, difference_idea.values, levels=levels, cmap = 'RdBu',
                                    transform = ccrs.PlateCarree(), extend='both')
            ax1.coastlines()  


    for j, model_name in enumerate(model_name_list):
        axs[j,0].text(-0.07, 0.5, naming_models[model_name],
                        transform=axs[j,0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=14)

    axs[0,0].annotate(
        '(a)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, 0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    axs[0,1].annotate(
        '(b)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, 0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    axs[0,2].annotate(
        '(c)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, 0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    axs[0,3].annotate(
        '(d)',
        xy=(0, 1.1), xycoords='axes fraction',
        xytext=(+0.5, 0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

    
    #axs[0,2:4].set_title('Ramp up' ,fontsize=15)
    cbar = fig.colorbar(
        c,
        ax=axs[:,0:2],
        orientation='horizontal',
        fraction=0.03,
        pad=0.08
    )
    cbar.set_label('Precipitation anomalies (historical - reanalysis) (mm $d^{-1}$)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    cbar1 = fig.colorbar(
        c1,
        ax=axs[:,2:4],
        orientation='horizontal',
        fraction=0.03,
        pad=0.08
    )
    cbar1.set_label('Precipitation differences (ramp up - historical) (mm $d^{-1}$)', fontsize=18)
    cbar1.ax.tick_params(labelsize=14)
    plt.suptitle('Historical - Reanalysis                                                    Ramp up -  Historical' ,fontsize=25)
    plt.savefig(os.path.join(out_path,name_out))
    
plot1()
plot2()
plot34()
plot56()
plot7()
plot8()
plot9()
plot10()
