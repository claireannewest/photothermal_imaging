### Material Parameters ###
# Dipole spacing
DS: 1
# Radius of sphere
radius: 5
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
P_pu_offset: 1

### Probe Scattering Calculation ###
# Probe laser wavelength [um]
probe_um: 0.785
# Probe laser power [W]
P_pr: 0.0002
# Probe Polarization
P_pr_pol: y
# Probe Focal offset [DS]
P_pr_offset: 0

# Field parameters
# phi_min, phi_max, step size
phi_info: -180 180 1
# theta_min, theta_max, step size
theta_info: 0 35 1
# Numerical apperature of lens
num_app: 1.25
# IDK
n_ambient_heated: 0.00
# Half size of image box
image_width: 50
# Step size of raster
ss: 10
