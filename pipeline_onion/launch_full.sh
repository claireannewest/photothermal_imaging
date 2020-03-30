#!/bin/bash

zrange=400
zstart=-200
zend=$zstart+$zrange
ss=10

for ((z=$zstart;z<=$zend;z+=$ss));do
    cp launch_temp1 launch_temp_${z}
    sed -i "s/name=b_0/name=a_${z}/g" launch_temp_${z}
    sed -i "s/z=0/z=${z}/g" launch_temp_${z}
    sbatch launch_temp_${z}
done 

for ((z=$zstart;z<=$zend;z+=$ss));do
    cp launch_temp2 launch_temp_${z}
    sed -i "s/name=b_0/name=a_${z}/g" launch_temp_${z}
    sed -i "s/z=0/z=${z}/g" launch_temp_${z}
    sbatch launch_temp_${z}
done

