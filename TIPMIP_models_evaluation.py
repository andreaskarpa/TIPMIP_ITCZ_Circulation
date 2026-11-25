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


##############################################################
###### Plot of TIPMIP Models evaluation (historical and rampup          ######
##############################################################
file_rampup = os.path.join(data_path,'historical_rampup_precipitation_differences_fig.nc')
file = os.path.join(data_path,'historical_precipitation_differences_fig.nc')

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





##############################################################
###### Plot MSPAEF and Double ITCZ , models evaluation) ######
##############################################################
file_double_itcz = os.path.join(data_path,'Double_itcz_indices_fig.csv')
file_mspaef = os.path.join(data_path,'mspaef_values_fig.csv')
file = os.path.join(data_path,'historical_precipitation_differences_fig.nc')

df_double = pd.read_csv(file_double_itcz)
df_mspaef = pd.read_csv(file_mspaef)
da = xr.open_dataset(file)
da = da['pr']

naming_models = {'UKESM1-2' : 'UKESM1-2-LL', 'EC-Earth3' : 'EC-Earth3-ESM-1', 'IPSL' : 'IPSL-CM6-ESMCO2',
            'CNRM' : 'CNRM-ESM2-2', 'GFDL-ESM2M': 'GFDL-ESM2M', 'MIROC-ES2L' : 'MIROC-ES2L', 'NorESM2-LM': 'NorESM2-LM'}
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
