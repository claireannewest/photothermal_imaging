### Material Parameters ###
# Dipole spacing
DS: 1
# Dielectric data of room temperature metal
rt_dir: /gscratch/chem/clairew4/dielectrics/au_Conor_0K.txt
# Dielectric data of heated metal
heat_dir: /gscratch/chem/clairew4/dielectrics/
# Room temperature background refractive index 
n_R: 1.473
# Number of different absorbering materials in system
num_k: 1
# Thermal conductivity of background
k_back: 0.3
# Thermal conductivity of target
k_in: 314
# Thermal conductivity of substrate
k_sub: 0.6

### Pump Scattering Calculation ###
# Pump laser wavelength [um]
pump_um: 0.532
# Pump laser power [W]
P_pu: 0.0002
# Pump Polarization
P_pu_pol: y
# Pump Focal offset [DS]
P_pu_offset: 0

### Probe Scattering Calculation ###
# Probe laser wavelength [um]
probe_um: 0.785
# Probe laser power [W]
P_pr: 0.0002
# Probe Polarization
P_pr_pol: y
# Probe Focal offset [DS]
P_pr_offset: 0

### Field Integration and Raster Parameters ###
# number of phi values, from -180 to 180 deg
nplanes: 100
# theta_min, theta_max, and increment in degree
# (0 to 90)/(90 to 180)/(0 to 180) for forward/backward/all scattering 
theta_info: 0 35 5
NA: 1.25
n_ambient_heated: 0.00 
