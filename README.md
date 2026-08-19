# TIPMIP_ITCZ_Circulation
Python files for calculation of ITCZ location and characteristics and general circulation changes in TIPMIP runs
Python scripts for analyzing Intertropical Convergence Zone (ITCZ) location, structure, and large-scale circulation changes in TIPMIP simulations.

Contents:
1) ITCZ_loc_edges_multimodel_mean.py – Plots the multi-model mean of the difference in ITCZ position edges and strength between the GWL stabilization run and the pre-Industrial control runs
2) ITCZ_map_loc_evaluation.py – ITCZ methodology verification using ERA5, NCEP Reanalysis 2 and GPCP datasets
3) Precipitation_changes_multimodel_mean.py – Plots the multimodel mean precipitation changes between the GWL stabilization run and the pre-Industrial control runs
4) Moisture_flux_vertical_multimodel_mean.py – Plots vertical moisture flux changes between the GWL stabilization run and the pre-Industrial control runs
5) MMA_pressure_velocity_changes_pi_gwl_all_regions.py – Plots pressure and velocity changes across the different longitudinal regions between the GWL stabilization run and the pre-Industrial control runs
6) Circulation_changes_pi_gwl_single_plot_multimodel.py – Calculates and plots the multi-model mean of the difference of the general circulation between the GWL stabilization run and the pre-Industrial control (global zonal mean of mass streamfunction, pressure velocity, and precipitation)
7) Precipitation_evaluation.py – Evaluates precipitation in the historical runs in the models participating in TIPMIP
8) Plotting_all_figures.py - Plots all figures using the reduced datasets found in https://doi.org/10.5281/zenodo.21818786

9) Supplementary_Precipitation_changes_multimodel_mean.py - Plot supplementary figure (4C - 2C GWL) of the precipitation changes
10) Supplementary_Circulation_changes_pi_gwl_single_plot_multimodel.py - Plots supplementary figure (4C - 2C) of the global zonal mean of mass streamfunction and pressure velocity
11) Supplementary_MMA_pressure_velocity_changes_pi_gwl_all_regions.py - Plots supplementary figure (4C - 2C GWL) of the the local pressure velocity changes
12) Supplementary_Plotting_figures.py - Plots all supplementary figures using the reduced datasets found in https://doi.org/10.5281/zenodo.21818786

Requirements:
1. Python 3.x
2. Common scientific libraries (numpy, xarray, matplotlib, scipy, pandas, cartopy)
