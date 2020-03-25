# confocalpsf_pipeline

This is set up to run on Mox. Here are the steps to take to create a confocal PSF:

1. Simply adjust "shape.f90" to contain the shape that you would like to run. Make sure you do not change the variables "rastery" and "rasterz" and that those variables are added onto your y and z directions. 

2. Update the values in "parameters.input" to model the systemm of interest. At a minimum, make sure you adjust:
    a. line 5 to the wavlength (in microns) of your desired pump / heating beam.
    b. line 17 to the dipole spacing you've decided upon according to your shape.f90 file

3. Once all is updated, you can run a single test point if you'd like by running "launch_temp"
    a. This will launch a single calculation (i.e. a single raster position.)
    b. Adjust "ystart" and "z" in the file "launch_temp" to be the position (in lattice units) where you'd like your shape to be. (Recall we raster the shape, not the Gaussian beam. The Gaussian beamm is always at (0,0) and it is the shape that moves.)
    c. To launch the calculation, simply type "sbatch launch_temp" in the command line.
    d. Once the calculation has completed, type "module load anaconda3_4.3.1" then enter, then type "python collect_temps.py". (Double check that the header, lines 4-11 will grab your point)
    e. The files "scatter_hot" and "scatter_room" should be created. Check that the last column of each file is reasonable.

4. Before launching the full 2D calculation, identify the window where you wish to calculate your psf. Change files "launch_temp1", "launch_temp2", and launch_ful.sh" accordingly.
    a. The variables "yrange", "ystart" in "launch_temp1" and "launch_temp2" should be updated to cover the y ranges you wish to span. 
    b. The variables "zrange", "zstart" in launch_full.sh should be updated to cover the z ranges.
    c. The variables "ss" in all three files should be identical. This is the step size and can be adjusted to take more / less points in the psf.

5. To launch, type "bash launch_full.sh" into command line. Once complete, do steps 3d. and 3e. to collect all of the points. You can then make a 2D map to view your psf.
    
   
