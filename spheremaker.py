import numpy as np

class Generate_Sphere:
    def __init__(self, lat_space, radius_nm, yraster, zraster):
        '''Defines system parameters 

        Keyword arguments:
        lat_space -- lattice spacing in units of nm (equivalent to dipole spacing)
        radius_nm -- radius of sphere in units of nm
        '''
        self.lat_space = lat_space
        self.radius_nm = radius_nm
        self.yraster = yraster
        self.zraster = zraster

    def make_sphere(self):
        ''' This function is where you input the parameters for a single sphere.
        '''
        radius_ls = int(self.radius_nm/self.lat_space) # radius_ls of sphere in units of nm per lat. space
        # Define grid in which the sphere will be carved out of 
        x_range = np.arange(-radius_ls, radius_ls+1) # x points
        y_range = np.arange(-radius_ls, radius_ls+1) # y points
        z_range = np.arange(-radius_ls, radius_ls+1) # z points
        xgrid, ygrid, zgrid = np.meshgrid(x_range, y_range, z_range) # turns the 1D arrays into 3D grids
        all_points = np.column_stack((np.ravel(xgrid), np.ravel(ygrid), np.ravel(zgrid))) # restacks the 3 3D grids into 3 1D arrays
        all_points = all_points[all_points[:,0].argsort()] # sort to appease makemetal
        Xval = []; Yval = []; Zval = []
        for row in range(0, len(all_points[:,0])): #loops through each x, y, z coordinate
            x = all_points[row, 0]
            y = all_points[row, 1]
            z = all_points[row, 2]
            if x**2 + y**2 + z**2  <= radius_ls**2 : # checks if the x, y, z point should be a sphere point
                Xval = np.append(Xval, x) # adds point to shapefile array
                Yval = np.append(Yval, y) 
                Zval = np.append(Zval, z) 
        Xshifted = Xval - max(Xval) - 1 # the shape must start beneath the substrate-background interface 
        Yshifted = Yval + self.yraster # the shape is rastered across the laser focus, the y shift
        Zshifted = Zval + self.zraster # the shape is rastered across the laser focus, the z shift
        return Xshifted.astype(int), Yshifted.astype(int), Zshifted.astype(int)

    def write_shape(self):
        ''' This function writes the shape to a file compatabile with DDSCAT.
        '''
        x, y, z = self.make_sphere()
        num_points = len(x)
        file = open(str('shape.dat'),'w')
        file.write(str(' Sphere of radius ') + str(self.radius_nm) + str(' nm and dipole spacing ') + str(self.lat_space) + str(' nm')+' \n')
        file.write('\t' + str(num_points) + str(' = number of dipoles in target') + '\n')
        file.write(str(' 1.000000 0.000000 0.000000 = A_1 vector') + '\n')
        file.write(str(' 0.000000 1.000000 0.000000 = A_2 vector') + '\n')
        file.write(str(' 1.000000 1.000000 1.000000 = (d_x,d_y,d_z)/d') + '\n')
        file.write(str(' 0.000000 0.000000 0.000000 = (x,y,z)/d') + '\n')
        file.write(str(' JA  IX  IY  IZ ICOMP(x,y,z)') + '\n')
        for j in range(0, num_points):
            file.write('\t' + str(j+1) + '\t' + str(int(x[j])) + '\t' + str(int(y[j])) + '\t' + str(int(z[j])) + 
               '\t' + str(1) + '\t' + str(1) + '\t' + str(1) + '\n')
        file.close()
