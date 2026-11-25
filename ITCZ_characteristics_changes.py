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

data_path = '/path/to/data/'
out_path = '/path/to/out/'


############################################
###### Plot 2 (ITCZ changes)          ######
############################################
file = os.path.join(data_path,'itcz_characteristics_changes.csv')
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
    axs[4,j].axhline(0, color='black')

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
