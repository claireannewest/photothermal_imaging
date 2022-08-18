import numpy as np
import sys
import os
import shutil

class Photothermal_Files:
	def __init__(self):

		# Obtain all the parameters
		par = []
		val = []
		with open(str("parameters.py")) as file:
			for line in file:
				if "#" != line[0] and len(line.split()) != 0:
					par.append(line.split()[0][:-1])
					val.append(line.split()[1])

		self.lat_space = int(val[par.index("lat_space")])
		self.shell = bool(val[par.index("shell")])
		self.raster_length = int(val[par.index("raster_length")])
		self.stepsize = int(val[par.index("stepsize")])
		self.shapefile = val[par.index("shapefile")]
		self.k_out = float(val[par.index("k_out")])
		self.k_in = float(val[par.index("k_in")])
		self.k_shell = float(val[par.index("k_shell")])
		self.k_sub = float(val[par.index("k_sub")])
		self.n_back = float(val[par.index("n_back")])
		self.I0 = float(val[par.index("I0")])
		self.diel_paths_RT = (str(val[par.index("diel_paths_RT")])).split(',')
		self.r_thermal = int(val[par.index("r_thermal")])
		self.wave_pump = float(val[par.index("wave_pump")])
		self.wave_probe = float(val[par.index("wave_probe")])
		


		yrange = np.arange(-self.raster_length/2, self.raster_length/2+1, self.stepsize)
		zrange = np.arange(-self.raster_length/2, self.raster_length/2+1, self.stepsize)
		ygrid, zgrid = np.meshgrid(yrange, zrange)
		self.ypoints = np.ravel(ygrid); 
		self.zpoints = np.ravel(zgrid)


	def write_ddscatpar(self, shapefile, wavelength, diel_paths, nrfld, iwrksc, nplanes, gauss_center):
		data = np.loadtxt(shapefile,skiprows=7)
		xlen = max(data[:,1]) + np.abs(min(data[:,1]))
		ylen = max(data[:,2]) + np.abs(min(data[:,2]))
		zlen = max(data[:,3]) + np.abs(min(data[:,3]))
		N = len(data[:,1])
		eff_r = np.round((3*N/(4*np.pi))**(1/3)*self.lat_space*1E-3, 5)
		file = open(str('ddscat.par'),'w')
		file.write(str(" ' ========== Parameter file for v7.3 ==================='") + '\n')
		file.write(str(" '**** Preliminaries ****'") + '\n')
		file.write(str(" 'NOTORQ' = CMTORQ*6 (DOTORQ, NOTORQ) -- either do or skip torque calculations") + '\n')
		file.write(str(" 'PBCGS2' = CMDSOL*6 (PBCGS2, PBCGST, GPBICG, QMRCCG, PETRKP) -- CCG method)") + '\n')
		file.write(str(" 'GPFAFT' = CMETHD*6 (GPFAFT, FFTMKL) -- FFT method") + '\n')
		file.write(str(" 'GKDLDR' = CALPHA*6 (GKDLDR, LATTDR, FLTRCD) -- DDA method") + '\n')
		file.write(str(" 'NOTBIN' = CBINFLAG (NOTBIN, ORIBIN, ALLBIN)") + '\n')
		file.write(str(" '**** Initial Memory Allocation ****'") + '\n')
		file.write(str(" ") + str(int(xlen+50)) + str(" ") + str(int(ylen+50)) + str(" ") + \
				   str(int(zlen+50)) + str(" = dimensioning allowance for target generation") + '\n')
		file.write(str(" '**** Target Geometry and Composition ****'") + '\n')
		file.write(str(" 'FROM_FILE' = CSHAPE*9 shape directive") + '\n')
		file.write(str(" no SHPAR parameters needed") + '\n')
		file.write(str(" ")+str(len(diel_paths))+str("         = NCOMP = number of dielectric materials") + '\n')
		for idx, val in enumerate(diel_paths):
			file.write(str(" ")+str(val)+str(" = file with refractive index ") + str(idx+1) + '\n')
		file.write(str(" '**** Additional Nearfield calculation? ****'") + '\n')
		file.write(str(" ")+str(nrfld)+str(" = NRFLD (=0 to skip nearfield calc., =1 to calculate nearfield E, =2 to calculate nearfield E and B)") + '\n')
		file.write(str(" 0.0 0.0 0.0 0.0 0.0 0.0 (fract. extens. of calc. vol. in -x,+x,-y,+y,-z,+z) ") + '\n')
		file.write(str(" '**** Error Tolerance ****'") + '\n')
		file.write(str(" 1.00e-5 = TOL = MAX ALLOWED (NORM OF |G>=AC|E>-ACA|X>)/(NORM OF AC|E>)") + '\n')
		file.write(str(" '**** Maximum number of iterations ****'") + '\n')
		file.write(str(" 2370     = MXITER") + '\n')
		file.write(str(" '**** Integration cutoff parameter for PBC calculations ****'") + '\n')
		file.write(str(" 1.00e-2 = GAMMA (1e-2 is normal, 3e-3 for greater accuracy)") + '\n')
		file.write(str(" '**** Angular resolution for calculation of <cos>, etc. ****'") + '\n')
		file.write(str(" 0.5    = ETASCA (number of angles is proportional to [(3+x)/ETASCA]^2 )") + '\n')
		file.write(str(" '**** Vacuum wavelengths (micron) ****'") + '\n')
		file.write(str(" ") + str(wavelength) + str(" ") + str(wavelength) + str(" 1 'INV' = wavelengths (first,last,how many,how=LIN,INV,LOG)") + '\n')
		file.write(str(" '**** Gaussian beam parameters (unit = dipole spacing)'") + '\n')
		file.write(str(" 1  = FLGWAV: Option for wavefront: 0 -- Plane wave; 1 -- Gaussian beam") + '\n')
		file.write(str(" 0.00, ")+str(gauss_center[0]) +str(" ") + str(gauss_center[1]) + (" = xyzc0, center of Gaussian beam waist, unit = dipole spacing") + '\n')
		file.write(str(" 150 = w0, Gaussian beam waist, unit = dipole spacing") + '\n')
		file.write(str(" '**** Refractive index of ambient medium'") + '\n')
		file.write(str(" 1.473 = NAMBIENT") + '\n')
		file.write(str(" '**** Effective Radii (micron) **** '") + '\n')
		file.write(str(" ") + str(eff_r) + str(" ") + str(eff_r) + str(" 1 'LIN' = aeff (first,last,how many,how=LIN,INV,LOG)") + '\n')
		file.write(str(" '**** Define Incident Polarizations ****'") + '\n')
		file.write(str(" (0,0) (1.,0.) (0.,0.) = Polarization state e01 (k along x axis)") + '\n')
		file.write(str(" 1 = IORTH  (=1 to do only pol. state e01; =2 to also do orth. pol. state)") + '\n')
		file.write(str(" '**** Specify which output files to write ****'") + '\n')
		file.write(str(" ") + str(iwrksc) + str(" = IWRKSC (=0 to suppress, =1 to write .sca file for each target orient.") + '\n')
		file.write(str(" '**** Specify Target Rotations ****'") + '\n')
		file.write(str(" 0.    0.   1  = BETAMI, BETAMX, NBETA  (beta=rotation around a1)") + '\n')
		file.write(str(" 0.    0.   1  = THETMI, THETMX, NTHETA (theta=angle between a1 and k)") + '\n')
		file.write(str(" 0.    0.   1  = PHIMIN, PHIMAX, NPHI (phi=rotation angle of a1 around k)") + '\n')
		file.write(str(" '**** Specify first IWAV, IRAD, IORI (normally 0 0 0) ****'") + '\n')
		file.write(str(" 0   0   0    = first IWAV, first IRAD, first IORI (0 0 0 to begin fresh)") + '\n')
		file.write(str(" '**** Select Elements of S_ij Matrix to Print ****'") + '\n')
		file.write(str(" 6       = NSMELTS = number of elements of S_ij to print (not more than 9)") + '\n')
		file.write(str(" 11 12 21 22 31 41       = indices ij of elements to print") + '\n')
		file.write(str(" '**** Specify Scattered Directions ****'") + '\n')
		file.write(str(" 'LFRAME' = CMDFRM (LFRAME, TFRAME for Lab Frame or Target Frame)") + '\n')
		if int(nplanes) == 0:
			file.write(" 0 = NPLANES = number of scattering planes\n")
		if int(nplanes) > 0:
			file.write(str(" ") + str(nplanes) + str(" = NPLANES = number of scattering planes") + '\n')
		for i in range(int(nplanes)):
			theta_min=0; theta_max=35
			phi = -180 + i * 360 / (int(nplanes) - 1)
			file.write(" %r %r %r %r = phi, theta_min, theta_max (deg) for plane %r\n" % ( int(np.round(phi)), int(theta_min), int(theta_max), int(1), i+1))
		file.close()	
		


	def write_varpar(self):
		shape = np.loadtxt(self.shapefile, skiprows=7)
		xvals=shape[:,1]; yvals=shape[:,2]; zvals=shape[:,3] 
		xmin = int(min(xvals)-2)
		xmax = int(max(xvals)+2)
		ymin = int(min(yvals)-2)
		ymax = int(max(yvals)+2)
		zmin = int(min(zvals)-2)
		zmax = int(max(zvals)+2)
		xplane = int((min(xvals)-max(xvals))/2-1)
		file = open(str('var.par'),'w')
		if self.shell == False: num_k = 1
		if self.shell == True: num_k = 2
		file.write(str('num_k: ') + str(num_k) + '\n')
		file.write(str('k_out: ') + str(self.k_out) + '\n') 
		file.write(str('k_in: ') + str(self.k_in) + '\n') 
		if self.shell == True:	
			file.write(str('k_in: ') + str(self.k_shell) + '\n') 
		file.write(str('k_sub: ') + str(self.k_sub) + '\n')
		file.write(str('lambda: ') + str(np.round(self.wave_pump*10,4))+str('e-07') + '\n')
		file.write(str('n_m: ') + str(self.n_back) + '\n')
		file.write(str('I_0: ') + str(self.I0) + '\n')
		file.write(str('unit: ') + str(self.lat_space) + '\n' + '\n')
		file.write(str('d: 1') + '\n')
		file.write(str('x_min: ') + str(xmin) + '\n')
		file.write(str('x_max: ') + str(xmax) + '\n')
		file.write(str('y_min: ') + str(ymin) + '\n')
		file.write(str('y_max: ') + str(ymax) + '\n')
		file.write(str('z_min: ') + str(zmin) + '\n')
		file.write(str('z_max: ') + str(zmax) + '\n' + '\n')

		file.write(str('x_plane: ') + str(xplane) + '\n')
		file.write(str('input_mode: 1') + '\n')
		file.write(str('total_points: ') + str(int(len(xvals)))+'\n')

		file.close()





	def prepare_initial_calculations(self):
		### Prepare folders (including shape file, ddscat.par, and var.par) 

		for i in range(0, len(self.ypoints)):
			directory = str("y")+str(int(self.ypoints[i]))+str("_z")+str(int(self.zpoints[i]))
			os.mkdir(str("raster_data_H/")+str(directory))
			shutil.copyfile(self.shapefile, str("raster_data_H/")+str(directory)+str('/shape.dat'))
			self.write_ddscatpar(   shapefile=self.shapefile,
									wavelength=self.wave_pump, # um
			                        diel_paths=self.diel_paths_RT,
			                        nrfld=2,
			                        iwrksc=0,
			                        nplanes=0,
			                        gauss_center=np.array([self.ypoints[i], self.zpoints[i]])
			                       )
			os.rename('ddscat.par', str("raster_data_H/")+str(directory)+str('/ddscat.par'))

			## Prepare var.par file
			self.write_varpar()
			os.rename('var.par', str("raster_data_H/")+str(directory)+str('/var.par'))

		### Prepare launch files
		new_string_y = str('yarray=( ')+' '.join(repr(int(i)) for i in self.ypoints).replace("'", '"') + str(' )') + str('\n')
		new_string_z = str('zarray=( ')+' '.join(repr(int(i)) for i in self.zpoints).replace("'", '"') + str(' )') + str('\n')
		num=len(self.ypoints)

		##### Launch 1 #####
		new_launch1 = open('launch_temp1.slurm')
		lines1 = new_launch1.readlines()
		lines1[21] = str('#SBATCH --array=0-')+str(num-1)
		lines1[25] = new_string_y
		lines1[27] = new_string_z
		new_launch1 = open(str('launch1.slurm'),"w")
		new_launch1.writelines(lines1)
		new_launch1.close()

		##### Launch 2 #####
		shutil.copyfile('launch_temp2.slurm', 'launch2.slurm')

		##### Launch 3 #####
		new_launch3 = open('launch_temp3.slurm')
		lines3 = new_launch3.readlines()
		lines3[21] = str('#SBATCH --array=0-')+str(num-1)
		lines3[25] = new_string_y
		lines3[27] = new_string_z
		new_launch3 = open(str('launch3.slurm'),"w")
		new_launch3.writelines(lines3)
		new_launch3.close()

		##### Launch 4 #####
		shutil.copyfile('launch_temp4.slurm', 'launch4.slurm')

		##### Launch 5 #####
		new_launch5 = open('launch_temp5.slurm')
		lines5 = new_launch5.readlines()
		lines5[21] = str('#SBATCH --array=0-')+str(num-1)
		lines5[25] = new_string_y
		lines5[27] = new_string_z
		new_launch5 = open(str('launch5.slurm'),"w")
		new_launch5.writelines(lines5)
		new_launch5.close()

		##### Launch 6 #####
		shutil.copyfile('launch_temp6.slurm', 'launch6.slurm')


	def prepare_hot_probe_calcs(self):
		### Prepare folders (including extended shape file and ddscat.par) 
		    
		shape = np.loadtxt(self.shapefile,skiprows=7)
		shape_points = shape[:,1:4]
		x = np.arange(-self.r_thermal, self.r_thermal+1)
		y = np.arange(-self.r_thermal, self.r_thermal+1)
		z = np.arange(-self.r_thermal, self.r_thermal+1)

		newx= []; newy=[]; newz=[]

		for xval in x:
		    for yval in y:
		        for zval in z:
		            if xval**2 + yval**2 + zval**2 <= self.r_thermal**2:
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

		for i in range(0, len(self.ypoints)):
		    directory = str('raster_data_H/')+str("y")+str(int(self.ypoints[i]))+str("_z")+str(int(self.zpoints[i]))
		    temps = np.loadtxt(str(directory)+str('/temp.out'))
		    maxT = int(np.round(np.max(temps[:,4])))
		    if maxT == 0: 
		        shutil.copyfile(str(directory)+str('/shape.dat_pump'), str(directory)+str('/shape.dat'))
		        diel_paths = self.diel_paths_RT
		    else:
		        shutil.copyfile('shape.dat_extended', str(directory)+str('/shape.dat'))
		        diel_paths = [self.diel_paths_RT,
		                       str('/home/caw97/rds/hpc-work/diels/temp_dep_gold/glyc_')+str(maxT)+str('K.txt')]
		        
		    self.write_ddscatpar(	shapefile=str('shape.dat_extended'),
		    						wavelength=self.wave_probe,
		                            diel_paths=diel_paths,
		                            nrfld=0,
		                            iwrksc=1,
		                            nplanes=101,
		                            gauss_center=np.array([self.ypoints[i], self.zpoints[i]])
		                           )

		    os.rename('ddscat.par', str(directory)+str('/ddscat.par'))
		    
		    
		    
	def prepare_room_probe_calcs(self):
	### Prepare folders (including extended shape file and ddscat.par) 
		for i in range(0, len(self.ypoints)):
			directory_new = str("y")+str(int(self.ypoints[i]))+str("_z")+str(int(self.zpoints[i]))
			os.mkdir(str("raster_data_R/")+str(directory_new))

			hot_directory = str('raster_data_H/')+str("y")+str(int(self.ypoints[i]))+str("_z")+str(int(self.zpoints[i]))
			room_directory = str('raster_data_R/')+str("y")+str(int(self.ypoints[i]))+str("_z")+str(int(self.zpoints[i]))

			shutil.copyfile(str(hot_directory)+str('/shape.dat_pump'), str(room_directory)+str('/shape.dat'))

			self.write_ddscatpar(shapefile=self.shapefile,
									wavelength=self.wave_probe, # um
									diel_paths=self.diel_paths_RT,
									nrfld=0,
									iwrksc=1,
									nplanes=101,
									gauss_center=np.array([self.ypoints[i], self.zpoints[i]])
									)
			os.rename('ddscat.par', str(room_directory)+str('/ddscat.par'))

