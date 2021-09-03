#!/bin/bash
## Job Name
#SBATCH --job-name=NAMEVAL
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

module load anaconda3_4.3.1

# Shape specifications
lat_space=LATVAL
radius=RADVAL

# Raster Parameters
yrange=YRVAL
ystart=YSVAL
yend=$ystart+$yrange
ss=SSVAL
z=ZVAL

# Initial pump laser calculation
for ((y=$ystart;y<=$yend;y+=$ss));do
    cd raster_data
    mkdir x0_y${y}_z${z} 
    cd x0_y${y}_z${z}
    cp ../../template_files/spheremaker.py ./
    python -c "from spheremaker import Generate_Sphere as gs;\
               gs(lat_space=$lat_space, radius_nm=$radius, yraster=${y}, zraster=${z}).write_shape()"
    cp ../../template_files/makemetal_temp.py makemetal.py
    cp ../../template_files/my_inputgen.py ./ 
    cp ../../parameters.py ./
    python -c "from my_inputgen import Photothermal_Image as pt;\
               pt().make_ddscatpar(shapefile=str('shape.dat'),step=str('pump'))"
   /gscratch/chem/masiello_group/g-dda/source_code/ddscat &> x0_y${y}_z${z}_DDApump.out &
    cd ../../
done; wait

# Rename files
for ((y=$ystart;y<=$yend;y+=$ss));do
    cd raster_data/x0_y${y}_z${z}
    mv ddscat.par ddscat.par_pump; mv shape.dat shape.dat_pump
    mv qtable qtable_pump; rm qtable2; rm mtable
    rm Einc_w000_ddscat.par; rm EBsca_w000_ddscat.par; 
    mv tdda_input_w000_ddscat.par tdda_input
    cd ../../
done; echo 'Finished all pump scattering calculations'

# tDDA calculation
for ((y=$ystart;y<=$yend;y+=$ss));do
    cd raster_data/x0_y${y}_z${z}
    python -c "from my_inputgen import Photothermal_Image as pt;\
               pt().make_varpar(shapefile=str('shape.dat_pump'))"
    /gscratch/chem/masiello_group/t-dda/source_code/Lattice_Diffusion /gscratch/chem/masiello_group/myGreen.num_300 var.par tdda_input temp.out &> x0_y${y}_z${z}_tDDA.out &
    cd ../../
done; wait; echo 'Temperature calculations finished'

# Check temperatures
cd raster_data
for ((y=$ystart;y<=$yend;y+=$ss));do
    RESULT=`python ../template_files/check_temp.py x0_y${y}_z${z}`
    echo "Made it here"
    if [ "$RESULT" = "True" ]; then
	echo "Job x0_y${y}_z${z} canceled due to negative temperatures."
	scancel --name NAMEVAL
    fi
done
cd ../

# Hot Probe calculations
for ((y=$ystart;y<=$yend;y+=$ss));do
    cd raster_data/x0_y${y}_z${z}
    python -c "from my_inputgen import Photothermal_Image as pt;\
               pt().make_makemetal(shapefile=str('shape.dat_pump'))"
    python makemetal.py
    python -c "from my_inputgen import Photothermal_Image as pt;\
               pt().make_ddscat.par(shapefile=str('shape.dat'),step=str('probe_hot'))"
    /gscratch/chem/masiello_group/g-dda/source_code/ddscat &> x0_y${y}_z${z}_DDAprobe.out &
    cd ../../
done; wait

# # Rename after jobs are finished
# for ((y=$ystart;y<=$yend;y+=$ss));do
#     cd raster_data/x0_y${y}_z${z}
#     mv ddscat.par ddscat.par_hot;
#     mv qtable qtable_hot; rm qtable2; rm mtable
#     rm Einc_w000_ddscat.par; rm makemetal_temp.py; rm ddscat_filler; rm temp-shift.txt
#     mv Integration_f11f11 Integration_f11f11_hot
#     mv w000r000k000.fml fml_x0y${y}z${z}_H
#     rm tdda_input
#     rm shape.dat*
#     cd ../../
# done; wait; echo 'Probe scattering calculation finished' 

# ###################################################################################
#  ### Room temperature calculation ###
# ###################################################################################

# #Initial set up & scattering calculation
# for ((y=$ystart;y<=$yend;y+=$ss));do
#     cd raster_data/x0_y${y}_z${z}
#     cp ../../template_files/spheremaker.py .
#     python -c "from spheremaker import Generate_Sphere;\
#                Generate_Sphere(lat_space=$lat_space, radius_nm=$radius, yraster=${y}, zraster=${z}).write_shape()"
#     cp ../../template_files/input_generator.py ./ 
#     cp ../../parameters.input ./
#     python input_generator.py -d parameters.input shape.dat
#     python input_generator.py -dR parameters.input shape.dat
#     /gscratch/chem/masiello_group/g-dda/source_code/ddscat &> x0_y${y}_z${z}_DDAroom.out &
#     cd ../../
# done; wait

# for ((y=$ystart;y<=$yend;y+=$ss));do
#     cd raster_data/x0_y${y}_z${z}
#     mv ddscat.par ddscat.par_room;
#     mv qtable qtable_room; rm qtable2; rm mtable
#     rm Einc_w000_ddscat.par;
#     mv Integration_f11f11 Integration_f11f11_room
#     mv w000r000k000.fml fml_x0y${y}z${z}_R
#     cd ../../
# done

