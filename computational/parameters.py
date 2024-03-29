### Material Parameters ###
# Dipole spacing
lat_space: 1
# Shell or no shell
shell: True
# shape
shapefile: shape.dat_little
diel_paths_RT: '/home/caw97/rds/hpc-work/diels/Mg_Palik.txt','/home/caw97/rds/hpc-work/diels/MgO.txt'
wave_pump: 0.532
wave_probe: 0.785

### Thermal Details  ###
k_out: 0.6
k_in: 314
k_sub: 0.6
n_back: 1.473
k_shell: 0.5
I0: 1E9
r_thermal: 30


# n_R: 1.473
# # Number of different absorbering materials in system
# num_k: 1
# # Thermal conductivity of background
# k_back: 0.3
# # Thermal conductivity of target
# k_in: 314
# # Thermal conductivity of substrate
# k_sub: 0.6


### Image Parameters ###
# Raster length (make sure devisible by stepsize)
raster_length: 20
# Stepsize 
stepsize: 20

####################

NA: 1.25
# phi_min, phi_max, step size
phi_info: -180 180 1
# theta_min, theta_max, step size
theta_info: 0 35 1

###########



























# # Radius of sphere
# radius: 5
# # Dielectric data of room temperature metal
# rt_dir: /home/caw97/rds/hpc-work/diels/au_Conor_0K.txt
# # Dielectric data of heated metal
# heat_dir: /home/caw97/rds/hpc-work/diels/
# # Room temperature background refractive index 
# n_R: 1.473
# # Number of different absorbering materials in system
# num_k: 1
# # Thermal conductivity of background
# k_back: 0.3
# # Thermal conductivity of target
# k_in: 314
# # Thermal conductivity of substrate
# k_sub: 0.6

# ### Pump Scattering Calculation ###
# # Pump laser wavelength [um]
# pump_um: 0.532
# # Pump laser power [W]
# P_pu: 0.0002
# # Pump Polarization
# P_pu_pol: y
# # Pump Focal offset [DS]
# P_pu_offset: 1

# ### Probe Scattering Calculation ###
# # Probe laser wavelength [um]
# probe_um: 0.785
# # Probe laser power [W]
P_pr: 0.0002
# # Probe Polarization
P_pr_pol: y
# # Probe Focal offset [DS]
P_pr_offset: 0

# # Field parameters
# # Numerical apperature of lens
# num_app: 1.25
# # IDK
# n_ambient_heated: 0.00
# # Half size of image box
# image_width: 40
# # Step size of raster
# ss: 20
