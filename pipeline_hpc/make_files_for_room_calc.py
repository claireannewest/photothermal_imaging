import numpy as np
from make_files import Mg_Shapes
import os
import shutil
shapefile=str('shape.dat_little')
mgshape = Mg_Shapes(lat_space=1,
                shell=True)

boxsize = 80 # nm, rastering 
stepsize = 20

yrange = np.arange(-boxsize/2, boxsize/2+1, stepsize)
zrange = np.arange(-boxsize/2, boxsize/2+1, stepsize)
ygrid, zgrid = np.meshgrid(yrange, zrange)
ypoints = np.ravel(ygrid); zpoints = np.ravel(zgrid)




for i in range(0, len(ypoints)):
    directory = str('raster_data_R/')+str("y")+str(int(ypoints[i]))+str("_z")+str(int(zpoints[i]))
    shutil.copyfile(str(directory)+str('/shape.dat_pump'), str(directory)+str('/shape.dat'))
    diel_paths = ['/home/caw97/rds/hpc-work/diels/Mg_Palik.txt', '/home/caw97/rds/hpc-work/diels/MgO.txt']
    mgshape.write_ddscatpar(shapefile=shapefile,
                            wavelength=0.785, # um
                            diel_paths=diel_paths,
                            nrfld=0,
                            iwrksc=1,
                            nplanes=101,
                            gauss_center=np.array([ypoints[i], zpoints[i]])
                           )
    os.rename('ddscat.par', str(directory)+str('/ddscat.par'))
