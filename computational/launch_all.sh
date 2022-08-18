#!/bin/bash

FIRST=$(sbatch  --parsable launch1.slurm)

SECOND=$(sbatch --parsable --dependency=afterany:${FIRST} launch2.slurm)

THIRD=$(sbatch --parsable --dependency=afterany:${SECOND} launch3.slurm)

FOURTH=$(sbatch --parsable --dependency=afterany:${THIRD} launch4.slurm)

FIFTH=$(sbatch --parsable --dependency=afterany:${FOURTH} launch5.slurm)

sbatch --dependency=afterany:${FIFTH} launch6.slurm
