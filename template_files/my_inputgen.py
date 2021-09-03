import numpy as np


class Photothermal_Image:
	def __init__(self):
		'''BLAH
		                                                                                                   
		Keyword arguments:                                                                                         
		lat_space -- BLAH                                                              
		'''

		# Obtain all the parameters
		par = []
		val = []
		with open(str("parameters.py")) as file:
			for line in file:
				if "#" != line[0] and len(line.split()) != 0:
					par.append(line.split()[0][:-1])
					if 'theta_info' in line:
						val.append(line.split()[1:4])
					else:
						val.append(line.split()[1])

		self.DS = int(val[par.index("DS")])
		self.rt_dir = val[par.index("rt_dir")]
		self.heat_dir = val[par.index("heat_dir")]
		self.n_R = float(val[par.index("n_R")])
		self.k_back = float(val[par.index("k_back")])
		self.k_in = float(val[par.index("k_in")])
		self.k_sub = float(val[par.index("k_sub")])
		self.pump_um = float(val[par.index("pump_um")])
		self.P_pu = float(val[par.index("P_pu")])
		self.P_pu_pol = val[par.index("P_pu_pol")]
		self.P_pu_offset = float(val[par.index("P_pu_offset")])
		self.probe_um = float(val[par.index("probe_um")])
		self.P_pr = float(val[par.index("P_pr")])
		self.P_pr_pol = val[par.index("P_pr_pol")]
		self.P_pr_offset = float(val[par.index("P_pr_offset")])
		self.nplanes = int(val[par.index("nplanes")])
		self.theta_info = val[par.index("theta_info")]
		self.NA = float(val[par.index("NA")])

	def make_ddscatpar(self, 
		shapefile,
		step,
		):
		##### PUMP SCATTERING #####
		if step == "pump":
			wavelength = self.pump_um
			diel_files = str(' "') + str(self.rt_dir) + str('" = file with metal refractive index\n')
			ncomp = 1
			nrfld = 2
			focal_offset = self.P_pu_offset
			n_back = self.n_R
			inc_pol = self.P_pu_pol
			wrsc = 0
			nplanes = 1

		##### HOT PROBE SCATTERING #####
		if step == "probe_hot":
			wavelength = self.probe_um
			with open(str("ddscat_filler"), 'r') as heated_material:
				diel_files = heated_material.readlines()
				ncomp = len(diel_files)
			nrfld = 0
			focal_offset = self.P_pr_offset
			with open(str("n_T_of_temp_max"),'r') as file:
				n_back = file.readlines()
			n_back = n_back[0]
			inc_pol = self.P_pr_pol
			wrsc = 1
			nplanes = self.nplanes

		##### ROOM PROBE SCATTERING #####
		if step == "probe_room":
			wavelength = self.probe_um
			diel_files = self.rt_dir
			ncomp = 1
			nrfld = 0
			focal_offset = self.P_pr_offset
			n_back = self.n_R
			inc_pol = self.P_pr_pol
			wrsc = 1
			nplanes = self.nplanes

		#################
		# memory allocation 
		x = []; y = []; z = []
		with open(shapefile) as file:
		     data = file.readlines ()
		for line in data:
		    line = line.split()
		    if len(line) == 7 and '=' not in line:
		       x.append(int(line[1]))
		       y.append(int(line[2]))
		       z.append(int(line[3]))
		mem_allo_x = max(x) - min(x) + 10
		mem_allo_y = max(y) - min(y) + 10
		mem_allo_z = max(z) - min(z) + 10
		# Waist of focused Gaussian 
		waist = wavelength * 10**3 * 0.6 / (self.NA) / int(self.DS)  
		waist = "{0:.4f}".format(waist)

		# Effective radius
		effR = (3 * len(x) / (4 * np.pi))**(1 / 3.0) * int(self.DS) * 10**(-3)
		effR = "{0:.4f}".format(effR)

		# Make ddscat.par file 
		f = open("ddscat.par", "w")
		f.write(" ' ========== Parameter file for v7.3 ==================='\n")
		f.write(" '**** Preliminaries ****'\n")
		f.write(" 'NOTORQ' = CMTORQ*6 (DOTORQ, NOTORQ) -- either do or skip torque calculations\n")
		f.write(" 'PBCGS2' = CMDSOL*6 (PBCGS2, PBCGST, GPBICG, QMRCCG, PETRKP) -- CCG method\n")
		f.write(" 'GPFAFT' = CMETHD*6 (GPFAFT, FFTMKL) -- FFT method\n")
		f.write(" 'GKDLDR' = CALPHA*6 (GKDLDR, LATTDR, FLTRCD) -- DDA method\n")
		f.write(" 'NOTBIN' = CBINFLAG (NOTBIN, ORIBIN, ALLBIN)\n")
		f.write(" '**** Initial Memory Allocation ****'\n")
		f.write(" %r %r %r = dimensioning allowance for target generation\n" % (mem_allo_x, mem_allo_y, mem_allo_z))
		f.write(" '**** Target Geometry and Composition ****'\n")
		f.write(" 'FROM_FILE' = CSHAPE*9 shape directive\n")
		f.write(" no SHPAR parameters needed\n")
		f.write(" %r         = NCOMP = number of dielectric materials\n" % ncomp)
		for line in diel_files:
			f.write( line)
		f.write(" '**** Additional Nearfield calculation? ****'\n")
		f.write(" %r = NRFLD (=0 to skip nearfield calc., =1 to calculate nearfield E, =2 to calculate nearfield E and B)\n" % nrfld)
		f.write(" 0.0 0.0 0.0 0.0 0.0 0.0 (fract. extens. of calc. vol. in -x,+x,-y,+y,-z,+z)\n")
		f.write(" '**** Error Tolerance ****'\n")
		f.write(" 1.00e-5 = TOL = MAX ALLOWED (NORM OF |G>=AC|E>-ACA|X>)/(NORM OF AC|E>)\n")
		f.write(" '**** Maximum number of iterations ****'\n")
		f.write(" 2370     = MXITER\n")
		f.write(" '**** Integration cutoff parameter for PBC calculations ****'\n")
		f.write(" 1.00e-2 = GAMMA (1e-2 is normal, 3e-3 for greater accuracy)\n")
		f.write(" '**** Angular resolution for calculation of <cos>, etc. ****'\n")
		f.write(" 0.5    = ETASCA (number of angles is proportional to [(3+x)/ETASCA]^2 )\n")
		f.write(" '**** Vacuum wavelengths (micron) ****'\n")
		f.write(" %r %r 1 'INV' = wavelengths (first,last,how many,how=LIN,INV,LOG)\n" % (wavelength, wavelength))
		f.write(" '**** Gaussian beam parameters (unit = dipole spacing)'\n")
		f.write(" 1  = FLGWAV: Option for wavefront: 0 -- Plane wave; 1 -- Gaussian beam\n")
		f.write(" %r, 0.00, 0.00 = xyzc0, center of Gaussian beam waist, unit = dipole spacing\n" % focal_offset)
		f.write(" %r = w0, Gaussian beam waist, unit = dipole spacing\n" % float(waist))
		f.write(" '**** Refractive index of ambient medium'\n")
		f.write(" %r = NAMBIENT\n" % (float(n_back)))
		f.write(" '**** Effective Radii (micron) **** '\n")
		f.write(" %r %r 1 'LIN' = eff. radii (first, last, how many, how=LIN,INV,LOG)\n" % (float(effR), float(effR)))
		f.write(" '**** Define Incident Polarizations ****'\n")
		if inc_pol == "y":
		   f.write( " (0,0) (1.,0.) (0.,0.) = Polarization state e01 (k along x axis)\n")
		elif inc_pol == "z":
		   f.write( " (0,0) (0.,0.) (1.,0.) = Polarization state e01 (k along x axis)\n")
		f.write(" 1 = IORTH  (=1 to do only pol. state e01; =2 to also do orth. pol. state)\n")
		f.write(" '**** Specify which output files to write ****'\n")
		f.write(" %r = IWRKSC (=0 to suppress, =1 to write \".sca\" file for each target orient.\n" % wrsc)
		f.write(" '**** Specify Target Rotations ****'\n")
		f.write(" 0.    0.   1  = BETAMI, BETAMX, NBETA  (beta=rotation around a1)\n")
		f.write(" 0.    0.   1  = THETMI, THETMX, NTHETA (theta=angle between a1 and k)\n")
		f.write(" 0.    0.   1  = PHIMIN, PHIMAX, NPHI (phi=rotation angle of a1 around k)\n")
		f.write(" '**** Specify first IWAV, IRAD, IORI (normally 0 0 0) ****'\n")
		f.write(" 0   0   0    = first IWAV, first IRAD, first IORI (0 0 0 to begin fresh)\n")
		f.write(" '**** Select Elements of S_ij Matrix to Print ****'\n")
		f.write(" 6       = NSMELTS = number of elements of S_ij to print (not more than 9)\n")
		f.write(" 11 12 21 22 31 41       = indices ij of elements to print\n")
		f.write(" '**** Specify Scattered Directions ****'\n")
		f.write(" 'LFRAME' = CMDFRM (LFRAME, TFRAME for Lab Frame or Target Frame)\n")
		for i in range(int(nplanes)):
			if int(nplanes) == 1: 
				f.write(" 0 = NPLANES = number of scattering planes\n")
			else:
				phi = -180 + i * 360 / (int(nplanes) - 1)
				f.write(" %r %r. %r. %r = phi, theta_min, theta_max (deg) for plane %r\n" % (float("{0:.4f}".format(phi)), int(self.theta_info[0]), int(self.theta_info[1]), int(self.theta_info[2]), i+1))
		f.close()

	def make_varpar(self, shapefile):
		# tDDA window
		x = []; y = []; z = []
		with open(shapefile) as file:
			data = file.readlines()
		for line in data:
			line = line.split()
			if len(line) == 7 and '=' not in line:
			   x.append(int(line[1]))
			   y.append(int(line[2]))
			   z.append(int(line[3]))
		window = 2 # [DS] since we're not making glycerol shells, no need to extend target
		x_min = min(x) - int(window) 
		x_max = max(x) + int(window) 
		y_min = min(y) - int(window) 
		y_max = max(y) + int(window) 
		z_min = min(z) - int(window) 
		z_max = max(z) + int(window) 
		x_plane = int( (x_max+x_min) / 2)
		## Convert Power to Intensity 
		waist = float(self.pump_um) * 10**3 * 0.6 / (self.NA) / self.DS
		waist = float(waist)
		I_0 = float(self.P_pu) / ( np.pi * (waist * self.DS * 10**-9)**2 )*10**-9 #units of nW/m^2, then will get converted to W/m^2 for var.par
		I_0 = "{:.4E}".format(I_0)

		f = open("var.par", "w")
		f.write("num_k: 1\n")
		f.write("k_out: %r\n" % self.k_back)
		f.write("k_in: %r\n" % self.k_in)
		f.write("k_sub: %r\n" % self.k_sub)
		f.write("lambda: %r\n" % (self.pump_um*10**(-6)))
		f.write("n_m: %r\n" % self.n_R)
		f.write("I_0: %re+9\n" % float(I_0))
		f.write("unit: %r\n\n" % self.DS)
		f.write("d: 1\n")
		f.write("x_min: %r\n" % int(x_min))
		f.write("x_max: %r\n" % int(x_max))
		f.write("y_min: %r\n" % int(y_min))
		f.write("y_max: %r\n" % int(y_max))
		f.write("z_min: %r\n" % int(z_min))
		f.write("z_max: %r\n\n" % int(z_max))
		f.write("x_plane: %r\n" % int(x_plane))
		f.write("input_mode: 1\n")
		f.close()

	def make_makemetal(self, shapefile):
		with open('makemetal_temp.py', 'r') as file:
			data = file.readlines()  
		for i in range(len(data)):
			if "nback =" in data[i]:
				message_nback = ("nback = %r #refractive index of background at room temperature\n" % float(self.n_R))
				data[i] = message_nback
			if "diel_path =" in data[i]:
				message_psf_dc_dir = ("diel_path = %r #location to store dielectric files\n" % str(self.rt_dir))
				data[i] = message_psf_dc_dir 
			if "shapefile =" in data[i]:
				message_shapefile = ("shapefile = np.loadtxt(%r,skiprows=7)\n" % str(shapefile))
				data[i] = message_shapefile
		return data






pt = Photothermal_Image()
# pt.make_ddscatpar(shapefile=str("shape.dat"),step=str("probe_room"))
# pt.make_varpar(shapefile=str("shape.dat"))
# pt.make_makemetal(shapefile="shape.dat")

