# Confocal Photothermal Imaging Pipeline

This is set up to create a photothermal image of a single spherical nanoparticle. 

## Instructions to Make Photothermal Image 
1. Update `parameters.py` to adjust the shape, materials involved (metal, background, substrate refractive indices and thermal conductivities), laser parameters, and imaging conditions.  

2. The current submission scripts are written as slurm files. If running on the cluster, simply change Line 20 in `launch_full.slurm`with the jobname you desire. 

3. Read through `batch.slurm` and make sure you have access to the source codes called. (E.g., lines 45, 65, 89, 115.) You will need [g-dda](http://github.com/MasielloGroup/g-dda) and [t-dda](http://github.com/MasielloGroup/t-dda) if you are not running these scripts on UW's Mox Hyak. 

4. The full image can be run by running `launch_full.slurm` (type `sbatch launch_full.slurm` if submitting to a cluster).

## Be Aware
* When setting step size of the detection area in `parameters.py` (lines 43 and 45), make sure the step size goes in evenly with the range you've selected. `g-dda` always ends with the maximum value, regardless of wheter or not the last step size equals the step size you've set.


## Extensions
* To run different shapes, you will need to edit `spheremaker.py` to make the shape you desire, and edit `parameters.py` to incorportate any parameters you wish to have set there. 

* To run a single raster point (e.g. calculate the photothermal signal at one beam position), change `image_width` in `parameters.py` to 0, and run `single.slurm`.

* If you'd like to not integrate the fields across the detector area, and instead only calculate the photothermal signal on the optical axis, move the two files outside of `no_integration` into the main folder. Adjust the theta and phi range to be smaller than step size in `parameters.py`. (E.g. `phi_info: 0 0.1 0.5`). Then, run `noint_single.slurm`. Note this is only set up to run a single raster point, and not a full photothermal image. 


