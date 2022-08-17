#!/bin/bash
#! Name of the job:
#SBATCH -J all
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

module load anaconda
module load gcc/6

FIRST=$(sbatch  --parsable launch1.slurm)

SECOND=$(sbatch --parsable --dependency=afterany:${FIRST} launch2.slurm)

THIRD=$(sbatch --parsable --dependency=afterany:${SECOND} launch3.slurm)

FOURTH=$(sbatch --parsable --dependency=afterany:${THIRD} launch4.slurm)

sbatch --dependency=afterany:${FOURTH} launch5.slurm
