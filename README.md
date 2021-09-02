# Confocal Photothermal Imaging Pipeline

This is set up to run a single sphere. To run different shapes, edit the `spheremaker.py` script in `template_files`. To make a rastered photothermal image: 
1. Update `parameters.input` and `launch_full.slurm`
2. Run `sbatch launch_full.slurm`.