Python scripts for analyzing Intertropical Convergence Zone (ITCZ) location, structure, and large-scale circulation changes in TIPMIP simulations.
Contents:
  1. Global_zonal_circulation_changes.py - Calculates and plots the multi-model mean of the difference of the general circulation between the GWL stabilization run and the pre-Industrial control (global zonal mean of mass streamfunction, pressure velocity, and precipitation). Plots the GWL2-piControl (Supplementary) and GWL4-piControl (main manuscript). 
  2. ITCZ_characteristics_changes.py – Plots the multi-model mean of the difference in ITCZ position edges and strength between the GWL stabilization run and the pre-Industrial control runs.
  3. Method_evaluation_ITCZ_observations.py - ITCZ methodology verification using ERA5, NCEP Reanalysis 2 and GPCP datasets.
  4. Moisture_transport_changes.py - Plots vertical moisture flux changes between the GWL stabilization run and the pre-Industrial control runs.
  5. Pathways_ITCZ_characteristics_changes.py - Plots the changes in the ITCZ characteristics between the late stabilization period and the transient ramp-up for both GWL2 and GWL4.
  6. Pathways_precipitation_changes.py - Plots the changes in precipitation between the early stabilization and transient ramp-up, as well as for late and early stabilization. Produces two plots, one using 50-year periods (main manuscript) and one using 100-year periods (Supplementary).
  7. Pathways_wap_changes_contributions.py -  Plots the changes in pressure velocity at 500hPa between the early stabilization and transient ramp-up, as well as for late and early stabilization. Produces two plots, one for the contribution of the pressure velocity to the meridional and one for the zonal overturning circulations.
  8. Precipitation_changes_multimodel_mean_adj.py - Plots the multimodel mean adjusted precipitation changes between the GWL stabilization run and the pre-Industrial control runs.
  9. Regional_pressure_velocity_changes.py - Plots pressure and velocity changes across the different longitudinal regions between the GWL stabilization run and the pre-Industrial control runs. Plots the GWL2-piControl (Supplementary) and GWL4-piControl (main manuscript).
  10. Supplementary_figures.py - It produces the rest of the figures in the Supplementary (GWL4-GWL2) for precipitation changes, regional pressure velocity changes and global zonal circulation changes.
  11. TIPMIP_models_evaluation.py - - Plots the precipitation anomalies for the historical runs of the models (1981-2005) against ERA5, and the differences of the first 10-50 years of the ramp-up against the same historical period, for the DJF and JJA seasons.

Requirements:
  Python 3.x
  Common scientific libraries (numpy, xarray, matplotlib, scipy, pandas, cartopy)
