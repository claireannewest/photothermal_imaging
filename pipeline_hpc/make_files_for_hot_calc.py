import numpy as np
from make_files import Mg_Shapes
import os
import shutil


boxsize = 80 # nm, rastering 
stepsize = 20
shapefile=str('shape.dat_little')
shape=np.loadtxt(str(shapefile), skiprows=7)

# Create ddscat.par files
mgshape = Mg_Shapes(lat_space=1, 
                shell=True)

yrange = np.arange(-boxsize/2, boxsize/2+1, stepsize)
zrange = np.arange(-boxsize/2, boxsize/2+1, stepsize)
ygrid, zgrid = np.meshgrid(yrange, zrange)
ypoints = np.ravel(ygrid); zpoints = np.ravel(zgrid)




shape = np.loadtxt(str('raster_data_H/y0_z0/shape.dat_pump'),skiprows=7)
shape_points = shape[:,1:4]
r_thermal = 20  # nm
x = np.arange(-30, 31)
y = np.arange(-30, 31)
z = np.arange(-30, 31)

newx= []; newy=[]; newz=[]

for xval in x:
    for yval in y:
        for zval in z:
            if xval**2 + yval**2 + zval**2 <= r_thermal**2:
                newx = np.append(newx, xval)
                newy = np.append(newy, yval)
                newz = np.append(newz, zval)

spec_grid = np.column_stack((newx, newy, newz))


newshape = shape[:,1:5]

for line in range(0, len(spec_grid)):
    idx = np.where( (shape_points == spec_grid[line,:]).all(axis=1) )
    if len(idx[0]) == 0:
        newline = np.append(spec_grid[line,:], 3)
        newshape = np.append(newshape, np.reshape(newline,(1,4)),axis=0)
        
        
file = open('shape.dat_extended','w')
file.write(str('Extended Shape') + '\n')
file.write('\t' + str(len(newshape)) + ' = number of dipoles in target' + '\n')
file.write(' 1.000000 0.000000 0.000000 = A_1 vector' + '\n')
file.write(' 0.000000 1.000000 0.000000 = A_2 vector' + '\n')
file.write(' 1.000000 1.000000 1.000000 = (d_x,d_y,d_z)/d' + '\n')
file.write(' 0.000000 0.000000 0.000000 = (x,y,z)/d' + '\n')
file.write(' JA  IX  IY  IZ ICOMP(x,y,z)'+ '\n')

for i in range(0, len(newshape)):
    file.write('\t' + str(i) + '\t' + str(int(newshape[i,0])) + '\t' + str(int(newshape[i,1])) + '\t' + str(int(newshape[i,2])) + '\t' + str(int(newshape[i,3])) + '\t' + str(int(newshape[i,3])) + '\t' + str(int(newshape[i,3])) + '\n')
file.close()

        
for i in range(0, len(ypoints)):
    directory = str('raster_data_H/')+str("y")+str(int(ypoints[i]))+str("_z")+str(int(zpoints[i]))
    temps = np.loadtxt(str(directory)+str('/temp.out'))
    maxT = int(np.round(np.max(temps[:,4])))
    if maxT == 0: 
        shutil.copyfile(str(directory)+str('/shape.dat_pump'), str(directory)+str('/shape.dat'))
        diel_paths = ['/home/caw97/rds/hpc-work/diels/Mg_Palik.txt', '/home/caw97/rds/hpc-work/diels/MgO.txt']
    else:
        shutil.copyfile('shape.dat_extended', str(directory)+str('/shape.dat'))
        diel_paths = ['/home/caw97/rds/hpc-work/diels/Mg_Palik.txt', 
                      '/home/caw97/rds/hpc-work/diels/MgO.txt',
                       str('/home/caw97/rds/hpc-work/diels/temp_dep_gold/glyc_')+str(maxT)+str('K.txt')]

        
    mgshape.write_ddscatpar(shapefile=shapefile,
                            wavelength=0.785, # um
                            diel_paths=diel_paths,
                            nrfld=0,
                            iwrksc=1,
                            nplanes=101,
                            gauss_center=np.array([ypoints[i], zpoints[i]])
                           )
    os.rename('ddscat.par', str(directory)+str('/ddscat.par'))
    
    
    
    
    
