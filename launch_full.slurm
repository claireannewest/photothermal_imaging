#!/bin/bash
## Job Name
#SBATCH --job-name=pt_image
## Allocation Definition
#SBATCH --account=chem-ckpt
#SBATCH --partition=ckpt
## Resources
## Nodes
#SBATCH --nodes=1
## Tasks per node (28 is Slurm default)
#SBATCH --ntasks-per-node=28
## Walltime (days-HH:MM:SS)
#SBATCH --time=15:00:00
## Memory per node
#SBATCH --mem=80G
##Output file
#SBTACH --output cfs.out

### Shape Parameters ###
lat_space=1
radius=5
jobname=sph10

### Photothermal Signal Parameters ###
lam_probe=0.785 #um

### Raster Parameters ###
width=10 # half the size of the box, e.g. width=100 means box size of -100 to 100 
ss=10

#########################################################################################
#########################################################################################
### Batch Launch the Jobs ###
zrange=$((2*width))
zstart=-width
zend=$zstart+$zrange
 
mkdir raster_data
mkdir gend_launch_files
mkdir standard_output 

for ((z=$zstart;z<=$zend;z+=$ss));do
    launch_file=gend_launch_files/lb_y-${width}_0_z${z}.slurm
    cp template_files/launch_batch.slurm ${launch_file}
    sed -i "s/name=NAMEVAL/name=pt_${jobname}/g" ${launch_file}
    sed -i "s/--name NAMEVAL/--name pt_${jobname}/g" ${launch_file}
    sed -i "s/lat_space=LATVAL/lat_space=${lat_space}/g" ${launch_file}
    sed -i "s/radius=RADVAL/radius=${radius}/g" ${launch_file}
    sed -i "s/yrange=YRVAL/yrange=${width}/g" ${launch_file}
    sed -i "s/ystart=YSVAL/ystart=-${width}/g" ${launch_file}
    sed -i "s/ss=SSVAL/ss=${ss}/g" ${launch_file}
    sed -i "s/z=ZVAL/z=${z}/g" ${launch_file}
    sbatch --output=standard_output/y-${width}_0_z${z}.out ${launch_file}
done 

# for ((z=$zstart;z<=$zend;z+=$ss));do
#     launch_file=gend_launch_files/lb_y${ss}_${width}_z${z}.slurm
#     cp template_files/launch_batch.slurm ${launch_file}
#     sed -i "s/name=NAMEVAL/name=pt_${jobname}/g" ${launch_file}
#     sed -i "s/--name NAMEVAL/--name pt_${jobname}/g" ${launch_file}
#     sed -i "s/lat_space=LATVAL/lat_space=${lat_space}/g" ${launch_file}
#     sed -i "s/radius=RADVAL/radius=${radius}/g" ${launch_file}
#     sed -i "s/yrange=YRVAL/yrange=$(( $width - $ss))/g" ${launch_file}
#     sed -i "s/ystart=YSVAL/ystart=${ss}/g" ${launch_file}
#     sed -i "s/ss=SSVAL/ss=${ss}/g" ${launch_file}
#     sed -i "s/z=ZVAL/z=${z}/g" ${launch_file}
#     sbatch --output=standard_output/y${ss}_${width}_z${z}.out ${launch_file}
# done
# wait

# ### Convert using Mueller Matrices to get a PT Image ###

# cp template_files/collect_temp_fml.py raster_data/collect_fml.py
# sed -i "s/begin=BVAL/begin=-${width}/g" raster_data/collect_fml.py
# sed -i "s/end=EVAL/end=$(( $width + 1))/g" raster_data/collect_fml.py
# sed -i "s/stepsize=SSVAL/stepsize=${ss}/g" raster_data/collect_fml.py
# sed -i "s/lam_probe=LVAL/lam_probe=${lam_probe}/g" raster_data/collect_fml.py
# wait

# sbatch --dependency=singleton --job-name=pt_${jobname} template_files/launch_collect.slurm
