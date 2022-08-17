#!/bin/bash
#! Name of the job:
#SBATCH -J 4-RoomOptical
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
#SBATCH --time=12:00:00
#! Estimated maximum memory needed (job is force-stopped if exceeded):
#! RAM is allocated in ~5980mb blocks, you are charged per block used,
#! and unused fractions of blocks will not be usable by others.
#SBATCH --mem=5980mb
#! Submit a job array with index values between 0 and 31
#! NOTE: This must be a range, not a single number (i.e. specifying '32' here would only run one job, with index 32)
#SBATCH --array=1-3


##### Array of Calculations #####
#################################




################################# 

cp -r raster_data_H raster_data_R
file=raster_data_R/y${yarray[$SLURM_ARRAY_TASK_ID]}_z${zarray[$SLURM_ARRAY_TASK_ID]}
cd $file
echo $file
/home/caw97/codes/g-dda/ddscat
cd ../../
