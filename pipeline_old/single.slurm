#!/bin/bash                                                                                                             
#! Name of the job:                                                                                                     
#SBATCH -J job1                                                                                                        
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

module load anaconda

jobname=sph10

params=($(python -c "from my_inputgen import Photothermal_Image as pt; pt().return_params()"))
width=${params[0]}
ss=${params[1]}
lat_space=${params[2]}
radius=${params[3]}

### Batch Launch the Jobs ###                                                                                                     

mkdir raster_data
mkdir gend_launch_files
mkdir standard_output

# Raster Parameters
y=0
z=0

# Initial pump laser calculation
cd raster_data
mkdir x0_y${y}_z${z} 
cd x0_y${y}_z${z}
cp ../../spheremaker.py ./
python -c "from spheremaker import Generate_Sphere as gs;\
               gs(lat_space=$lat_space, radius_nm=$radius, yraster=${y}, zraster=${z}).write_shape()"
cp ../../makemetal_temp.py makemetal.py
cp ../../my_inputgen.py ./ 
cp ../../parameters.py ./
python -c "from my_inputgen import Photothermal_Image as pt;\
               pt().make_ddscatpar(shapefile=str('shape.dat'),step=str('pump'))"
/home/caw97/codes/g-dda/source_code/ddscat &> x0_y${y}_z${z}_DDApump.out &
cd ../../
wait

# Rename files
cd raster_data/x0_y${y}_z${z}
mv ddscat.par ddscat.par_pump; mv shape.dat shape.dat_pump
mv qtable qtable_pump; rm qtable2; rm mtable
rm Einc_w000_ddscat.par; rm EBsca_w000_ddscat.par; 
mv tdda_input_w000_ddscat.par tdda_input
rm Integration_f11f11
cd ../../


# tDDA calculation
cd raster_data/x0_y${y}_z${z}
python -c "from my_inputgen import Photothermal_Image as pt;\
               pt().make_varpar(shapefile=str('shape.dat_pump'))"
/home/caw97/codes/t-dda/source_code/Lattice_Diffusion /home/caw97/codes/t-dda/lattice_greenfunction/Green_grid300.txt var.par tdda_input temp.out &> x0_y${y}_z${z}_tDDA.out &
cd ../../
wait

# Check temperatures
cd raster_data
RESULT=`python ../check_temp.py x0_y${y}_z${z}`
if [ "$RESULT" = "True" ]; then
	echo "Job x0_y${y}_z${z} canceled due to negative temperatures."
	scancel --name NAMEVAL
fi
cd ../

# Hot Probe calculations
cd raster_data/x0_y${y}_z${z}
python -c "from my_inputgen import Photothermal_Image as pt;\
               pt().make_makemetal(shapefile=str('shape.dat_pump'))"
python makemetal.py
python -c "from my_inputgen import Photothermal_Image as pt;\
               pt().make_ddscatpar(shapefile=str('shape.dat'),step=str('probe_hot'))"
/home/caw97/codes/g-dda/source_code/ddscat &> x0_y${y}_z${z}_DDAprobe.out &
cd ../../
wait 

# Rename after jobs are finished
cd raster_data/x0_y${y}_z${z}
mv ddscat.par ddscat.par_probehot;
mv qtable qtable_probehot; rm qtable2; rm mtable
rm Einc_w000_ddscat.par; rm ddscat_filler; rm temp-shift.txt
mv Integration_f11f11 Integration_f11f11_hot
mv w000r000k000.fml fml_x0y${y}z${z}_H
mv shape.dat shape.dat_probehot
cd ../../
wait

###################################################################################
 ### Room temperature calculation ###
###################################################################################

#Initial set up & scattering calculation
cd raster_data/x0_y${y}_z${z}
mv shape.dat_pump shape.dat
python -c "from my_inputgen import Photothermal_Image as pt;\
               pt().make_ddscatpar(shapefile=str('shape.dat'),step=str('probe_room'))"
/home/caw97/codes/g-dda/source_code/ddscat &> x0_y${y}_z${z}_DDAroom.out &
cd ../../
wait 

cd raster_data/x0_y${y}_z${z}
mv ddscat.par ddscat.par_proberoom;
mv qtable qtable_proberoom; rm qtable2; rm mtable
rm Einc_w000_ddscat.par;
mv Integration_f11f11 Integration_f11f11_room
mv w000r000k000.fml fml_x0y${y}z${z}_R
rm -r __pycache__
cd ../../


wait
python -c "from my_inputgen import Photothermal_Image as pt; pt().collect_fml()"
