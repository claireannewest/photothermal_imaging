import numpy as np
import argparse 
import sys

class Mg_Shapes:
	def __init__(self, lat_space, shell):
		"""Defines the different system parameters.
        
        Keyword arguments:
        lat_space -- [1/nm] nm per lattice cite 
		length -- [nm] y-dir, length of rod
		width -- [nm] z-dir, width of rod
		thick -- [nm] x-dir, thickness of rod
        """
		self.lat_space = lat_space
		self.shell = shell


	def write_ddscatpar(self, shapefile, wavelength, diel_paths, nrfld, iwrksc, nplanes, gauss_center):
		data = np.loadtxt(str(shapefile),skiprows=7)
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
			file.write(str(" '")+str(val)+str("' = file with refractive index ") + str(idx+1) + '\n')
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
		
	# def rewrite_shapefile_forT(self):
	# 	data = np.loadtxt(str(self.new_dir_spec)+str('/shape.dat'),skiprows=7)
	# 	xnew = data[:,1] - max(data[:,1]) - 1
	# 	ynew = data[:,2]
	# 	znew = data[:,3]
	# 	ICOMP = data[:,4]
	# 	if int(max(xnew)) != -1: print('Did not shift correctly!')
	# 	# Rewrite shape file ###
	# 	N = len(xnew)		
	# 	file = open(str(self.new_dir_temp)+str('/shape.dat'),'w')
	# 	if self.shell == False:
	# 		file.write(str(' ')+str(self.kind)+str(' ') + str(self.length)+str(' nm, ')+str(self.lat_space)+str(' DS') + '\n')
	# 	if self.shell == True:
	# 		file.write(str(' ')+str(self.kind)+str(' ') + str(self.length)+str(' nm, ')+str(self.lat_space)+str(' DS with shell') + '\n')
	# 	file.write('\t' + str(N) + str(' = number of dipoles in target') + '\n')
	# 	file.write(str(' 1.000000 0.000000 0.000000 = A_1 vector') + '\n')
	# 	file.write(str(' 0.000000 1.000000 0.000000 = A_2 vector') + '\n')
	# 	file.write(str(' 1.000000 1.000000 1.000000 = (d_x,d_y,d_z)/d') + '\n')
	# 	file.write(str(' 0.000000 0.000000 0.000000 = (x,y,z)/d') + '\n')
	# 	file.write(str(' JA  IX  IY  IZ ICOMP(x,y,z)') + '\n')
	# 	count = 0
	# 	for j in range(0, N):
	# 	    count = count+1
	# 	    file.write('\t' + str(count) + '\t' + str(int(xnew[j])) + '\t' + str(int(ynew[j])) + '\t' + str(int(znew[j])) + '\t' + str(int(ICOMP[j])) + '\t' + str(int(ICOMP[j])) + '\t' + str(int(ICOMP[j])) + '\n')
	# 	file.close()
	# 	return xnew, ynew, znew

	# def write_ddscatpar_forT(self, data_spec, which_cluster):
	# 	data = np.loadtxt(str(data_spec))
	# 	data_spectra =  data[data[:,1].argsort(),]
	# 	waves = data_spectra[:,1]
	# 	abs_cross = data_spectra[:,3]*np.pi*data_spectra[:,0]**2
	# 	idx = np.where(abs_cross == max(abs_cross))[0]
	# 	peak_abs = waves[idx]
	# 	### Update wavelength
	# 	new_ddscatpar = open(str(self.new_dir_temp)+str('/ddscat.par'))
	# 	lines = new_ddscatpar.readlines()
	# 	new_wave = str(' ') + str("%.4f" % peak_abs) + str(' ') + str("%.4f" % peak_abs) + str(" 1 'INV' = wavelengths (first,last,how many,how=LIN,INV,LOG)" + '\n')
	# 	new_setting = str(' 2 = NRFLD (=0 to skip nearfield calc., =1 to calculate nearfield E, =2 to calculate nearfield E and B)') +'\n'
	# 	if self.shell == False: shift=0
	# 	if self.shell == True: shift=1
	# 	if which_cluster == 'hpc':
	# 		diel_path = str('/home/')+str(self.CRSid)+str('/rds/hpc-work/diels/')
	# 	if which_cluster == 'esc':
	# 		diel_path = str('/home/')+str(self.CRSid)+str('/diels/')			
	# 	lines[13] = str(" '")+str(diel_path)+str("Mg_Palik.txt' = file with refractive index 1") + '\n'
	# 	if self.shell == True: lines[14] = str(" '")+str(diel_path)+str("MgO.txt' = file with refractive index 2") + '\n'
	# 	lines[26+shift] = new_wave
	# 	lines[15+shift] = new_setting
	# 	new_ddscatpar = open(str(self.new_dir_temp)+str('/ddscat.par'), 'w')
	# 	new_ddscatpar.writelines(lines)
	# 	new_ddscatpar.close()
	# 	return peak_abs

	# def write_launchfile_forT(self, which_cluster):
	# 	if which_cluster == 'hpc':
	# 		new_launch = open('template_files_spectra-shell-hpc/launch_temp.slurm')
	# 		lines = new_launch.readlines()
	# 		lines[2] = str('#SBATCH -J ')+str(self.kind)+str('-temp') + '\n'
	# 		lines[19] = '\n'
	# 		lines[20] = '\n'
	# 		lines[21] = '\n'
	# 		lines[22] = '\n'
	# 		lines[23] = '\n'
	# 		lines[24:35] = str('')
	# 		lines[20] = str('/home/')+str(self.CRSid)+str('/codes/g-dda/ddscat') + '\n'
	# 		lines[21] = str('wait') + '\n'
	# 		lines[22] = str('mv tdda_input_w000_ddscat.par tdda_input') + '\n'
	# 		lines[23] = str('wait') + '\n'
	# 		lines[24] = str('/home/')+str(self.CRSid)+str('/codes/t-dda/source_code/Lattice_Diffusion /home/')+str(self.CRSid)+str('/codes/t-dda/lattice_greenfunction/Green_grid300.txt var.par tdda_input temp.out') + '\n'
	# 		lines[25] = '\n'
	# 		lines[26] = '\n'
	# 		new_launch = open(str(self.new_dir_temp)+str('/launch.slurm'), 'w')
	# 		new_launch.writelines(lines)
	# 		new_launch.close()
	# 	if which_cluster == 'esc':
	# 		new_launch = open('template_files_spectra-shell-esc/launch_temp.slurm')
	# 		lines = new_launch.readlines()
	# 		lines[1] = str('#SBATCH -J ')+str(self.kind)+str('-temp') + '\n'
	# 		lines[6] = str('/home/')+str(self.CRSid)+str('/source_code/g-dda/source_code/ddscat') + '\n'
	# 		lines[7] = str('wait') + '\n'
	# 		lines[8] = str('mv tdda_input_w000_ddscat.par tdda_input') + '\n'
	# 		lines[9] = str('wait') + '\n'
	# 		lines[10] = str('/home/')+str(self.CRSid)+str('/source_code/t-dda/source_code/Lattice_Diffusion /home/')+str(self.CRSid)+str('/source_code/t-dda/lattice_greenfunction/Green_grid300.txt var.par tdda_input temp.out') + '\n'
	# 		lines[11:20] = str('')
	# 		new_launch = open(str(self.new_dir_temp)+str('/launch.slurm'), 'w')
	# 		new_launch.writelines(lines)
	# 		new_launch.close()






	def write_varpar(self, wavelength, xvals, yvals, zvals):
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
		file.write(str('k_out: 0.137') + '\n') # IPA
		file.write(str('k_in: 156') + '\n') # Mg
		if self.shell == True:	
			file.write(str('k_in: 42') + '\n') # MgO
		file.write(str('k_sub: 0.137') + '\n')
		file.write(str('lambda: ') + str(np.round(wavelength*10,4))+str('e-07') + '\n')
		file.write(str('n_m: 1.473') + '\n')
		file.write(str('I_0: 1.0e+9') + '\n')
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


