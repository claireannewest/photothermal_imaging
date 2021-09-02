#!/bin/bash
module load anaconda3_4.3.1

y=10
z=10


cd ../raster_data
RESULT=`python ../template_files/check_temp.py x0_y${y}_z${z}`
if [ "$RESULT" = "True" ]; then
    echo "it false"
    scancel --name NAMEVAL
fi
cd ../


# # Probe calculations
# for ((y=$ystart;y<=$yend;y+=$ss));do
#     cd raster_data/x0_y${y}_z${z}
#     mv shape.dat shape.dat_pump
#     python input_generator.py -p parameters.input shape.dat_pump
#     python makemetal.py
#     python input_generator.py -dN parameters.input shape.dat_pump shape.dat ddscat_filler n_T_of_temp_max
#     /gscratch/chem/masiello_group/g-dda/source_code/ddscat &> x00_y${y}_z${z}_DDAprobe.out &
#     cd ../../
# done; wait

# # Rename after jobs are finished
# for ((y=$ystart;y<=$yend;y+=$ss));do
#     cd x00_y${y}_z${z}
#     mv ddscat.par ddscat.par_hot;
#     mv qtable qtable_hot; rm qtable2; rm mtable
#     rm Einc_w000_ddscat.par; rm makemetal_temp.py; rm ddscat_filler; rm temp-shift.txt
#     mv Integration_f11f11 Integration_f11f11_hot
#     mv w000r000k000.fml fml_x00y${y}z${z}_H
#     rm tdda_input
#     rm x00_y${y}_z${z}_tDDA.out
#     rm x00_y${y}_z${z}_DDAprobe.out
#     rm shape.dat*
#     cd ..
# done; wait; echo 'Probe scattering calculation finished' 

# ###################################################################################
#  ### Room temperature calculation ###
# ###################################################################################

# #Initial set up & scattering calculation
# for ((y=$ystart;y<=$yend;y+=$ss));do
#     cd x00_y${y}_z${z}
#     cp ../spheremaker.py .
#     python -c "from spheremaker import Generate_Sphere; Generate_Sphere(lat_space=$lat_space, radius_nm=$radius, yraster=${y}, zraster=${z}).write_shape()"
#     cp ../input_generator.py .; cp ../parameters.input .
#     python input_generator.py -d parameters.input shape.dat
#     python input_generator.py -dR parameters.input shape.dat
#     /gscratch/chem/masiello_group/g-dda/source_code/ddscat &> x00_y${y}_z${z}_DDAroom.out &
#     cd ..
# done; wait

# for ((y=$ystart;y<=$yend;y+=$ss));do
#     cd x00_y${y}_z${z}
#     mv ddscat.par ddscat.par_room;
#     mv qtable qtable_room; rm qtable2; rm mtable
#     rm Einc_w000_ddscat.par;
#     mv Integration_f11f11 Integration_f11f11_room
#     mv w000r000k000.fml fml_x00y${y}z${z}_R
#     rm x00_y${y}_z${z}_DDAroom.out
#     cd ..
# done

