#!/bin/bash                                                                                                             
#! Name of the job:                                                                                                     
#SBATCH -J pt_image                                                                                                        
#SBATCH -A RINGE-SL3-CPU                                                                                                
#SBATCH -p icelake                                                                                                      
#! How many whole nodes should be allocated?                                                                            
#SBATCH --nodes=1                                                                                                       
#! How many (MPI) tasks will there be in total? (<= nodes*76)                                                           
#! The Ice Lake (icelake) nodes have 76 CPUs (cores) each and                                                           
#! 3380 MiB of memory per CPU.                                                                                          
#SBATCH --ntasks=1                                                                                                      
#! How many many cores will be allocated per task? (for single core jobs always leave this at 1)                        
#SBATCH --cpus-per-task=1                                                                                               
#! Estimated runtime: hh:mm:ss (job is force-stopped after if exceeded):                                                
#SBATCH --time=10:00:00                                                                                                 
#! Estimated maximum memory needed (job is force-stopped if exceeded):                                                  
#! RAM is allocated in ~5980mb blocks, you are charged per block used,                                                  
#! and unused fractions of blocks will not be usable by others.                                                         
#SBATCH --mem=5980mb                                                                                                    
    

jobname=sph10

params=($(python -c "from my_inputgen import Photothermal_Image as pt; pt().return_params()"))
width=${params[0]}
ss=${params[1]}
lat_space=${params[2]}
radius=${params[3]}

### Batch Launch the Jobs ###
zrange=$((2*width))
zstart=-width
zend=$zstart+$zrange
 
mkdir raster_data
mkdir gend_launch_files
mkdir standard_output 

for ((z=$zstart;z<=$zend;z+=$ss));do
    echo $z
    launch_file=gend_launch_files/lb_y-${width}_0_z${z}.slurm
    cp batch.slurm ${launch_file}
    sed -i "s/name=NAMEVAL/name=pt_${jobname}/g" ${launch_file}
    sed -i "s/--name NAMEVAL/--name pt_${jobname}/g" ${launch_file}
    sed -i "s/lat_space=LATVAL/lat_space=${lat_space}/g" ${launch_file}
    sed -i "s/radius=RADVAL/radius=${radius}/g" ${launch_file}
    sed -i "s/yrange=YRVAL/yrange=${width}/g" ${launch_file}
    sed -i "s/ystart=YSVAL/ystart=-${width}/g" ${launch_file}
    sed -i "s/ss=SSVAL/ss=${ss}/g" ${launch_file}
    sed -i "s/z=ZVAL/z=${z}/g" ${launch_file}
    sbatch --output=standard_output/x0_y-${width}_0_z${z}.out ${launch_file}
done 

for ((z=$zstart;z<=$zend;z+=$ss));do
    launch_file=gend_launch_files/lb_y${ss}_${width}_z${z}.slurm
    cp batch.slurm ${launch_file}
    sed -i "s/name=NAMEVAL/name=pt_${jobname}/g" ${launch_file}
    sed -i "s/--name NAMEVAL/--name pt_${jobname}/g" ${launch_file}
    sed -i "s/lat_space=LATVAL/lat_space=${lat_space}/g" ${launch_file}
    sed -i "s/radius=RADVAL/radius=${radius}/g" ${launch_file}
    sed -i "s/yrange=YRVAL/yrange=$(( $width - $ss))/g" ${launch_file}
    sed -i "s/ystart=YSVAL/ystart=${ss}/g" ${launch_file}
    sed -i "s/ss=SSVAL/ss=${ss}/g" ${launch_file}
    sed -i "s/z=ZVAL/z=${z}/g" ${launch_file}
    sbatch --output=standard_output/x0_y${ss}_${width}_z${z}.out ${launch_file}
done
wait

### Convert using Mueller Matrices to get a PT Image ###
sbatch --dependency=singleton --job-name=pt_${jobname} collect.slurm
