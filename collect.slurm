#!/bin/bash
## Job Name
#SBATCH --job-name=collect
## Allocation Definition
#SBATCH --account=chem-ckpt
#SBATCH --partition=ckpt
## Resources
## Nodes
#SBATCH --nodes=1
## Tasks per node (28 is Slurm default)
#SBATCH --ntasks-per-node=28
## Walltime (days-HH:MM:SS)
#SBATCH --time=1:00:00
## Memory per node
#SBATCH --mem=80G
##Output file
#SBTACH --output cfs.out

module load anaconda3_4.3.1
python -c "from my_inputgen import Photothermal_Image as pt; pt().collect_fml()"


