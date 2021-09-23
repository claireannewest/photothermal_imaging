import numpy as np
import os.path

hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]
c = 2.998E+10 # speed of light [cm/s]

class Photothermal_Image:
        def __init__(self):
                # Obtain all the parameters
                par = []
                val = []
                with open(str("parameters.py")) as file:
                        for line in file:
                                if "#" != line[0] and len(line.split()) != 0:
                                        par.append(line.split()[0][:-1])
                                        if 'theta_info' in line:
                                                val.append(line.split()[1:4])
                                        elif 'phi_info' in line:
                                                val.append(line.split()[1:4])
                                        else:
                                                val.append(line.split()[1])

                self.DS = int(val[par.index("DS")])
                self.radius = int(val[par.index("radius")])
                self.rt_dir = val[par.index("rt_dir")]
                self.heat_dir = val[par.index("heat_dir")]
                self.n_R = float(val[par.index("n_R")])
                self.num_k = int(val[par.index("num_k")])
                self.k_back = float(val[par.index("k_back")])
                self.k_in = float(val[par.index("k_in")])
                self.k_sub = float(val[par.index("k_sub")])
                self.pump_um = float(val[par.index("pump_um")])
                self.P_pu = float(val[par.index("P_pu")])
                self.P_pu_pol = val[par.index("P_pu_pol")]
                self.P_pu_offset = float(val[par.index("P_pu_offset")])*self.DS*1E-7 # units of cm 
                self.probe_um = float(val[par.index("probe_um")])
                self.P_pr = float(val[par.index("P_pr")])
                self.P_pr_pol = val[par.index("P_pr_pol")]
                self.P_pr_offset = float(val[par.index("P_pr_offset")])*self.DS*1E-7 # units of cm
                self.phi_info = val[par.index("phi_info")]
                self.theta_info = val[par.index("theta_info")]
                self.NA = float(val[par.index("num_app")])
                self.n_amb_heat = float(val[par.index("n_ambient_heated")])
                self.image_width = int(val[par.index("image_width")])
                self.ss = int(val[par.index("ss")])
                self.num_theta = (float(self.theta_info[1])-float(self.theta_info[0]))/float(self.theta_info[2])
                self.num_phi= (float(self.phi_info[1])-float(self.phi_info[0]))/float(self.phi_info[2])

        def waist(self,wave):
                # [nm]
                return wave * 10**3 * 0.6 / self.NA 

        def return_params(self):
                print(self.image_width)
                print(self.ss)
                print(self.DS)
                print(self.radius)

        def make_ddscatpar(self, shapefile,step,):
		##### PUMP SCATTERING #####
                if step == "pump":
                        wavelength = self.pump_um
                        g_waist_DS = self.waist(self.pump_um)/self.DS
                        diel_files = str(' "') + str(self.rt_dir) + str('" = file with metal refractive index\n')
                        ncomp = 1
                        nrfld = 2
                        focal_offset = self.P_pu_offset/(self.DS*1E-7) # DS
                        n_back = self.n_R
                        inc_pol = self.P_pu_pol
                        wrsc = 0
                        nplanes = 0

		##### HOT PROBE SCATTERING #####
                if step == "probe_hot":
                        wavelength = self.probe_um
                        g_waist_DS = self.waist(self.probe_um)/self.DS
                        with open(str("ddscat_filler"), 'r') as heated_material:
                                diel_files = heated_material.readlines()
                                ncomp = len(diel_files)
                        nrfld = 0
                        focal_offset = self.P_pr_offset/(self.DS*1E-7) # DS  
                        with open(str("n_T_of_temp_max"),'r') as file:
                                n_back = file.readlines()
                        n_back = n_back[0]
                        inc_pol = self.P_pr_pol
                        wrsc = 1
                        nplanes = self.num_phi + 1

                ##### ROOM PROBE SCATTERING #####
                if step == "probe_room":
                        wavelength = self.probe_um
                        g_waist_DS = self.waist(self.probe_um)/self.DS
                        diel_files = str(' "') + str(self.rt_dir) + str('" = file with metal refractive index\n')
                        ncomp = 1
                        nrfld = 0
                        focal_offset = self.P_pr_offset/(self.DS*1E-7) # DS  
                        n_back = self.n_R
                        inc_pol = self.P_pr_pol
                        wrsc = 1
                        nplanes = self.num_phi + 1

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
                f.write(" %.4f = w0, Gaussian beam waist, unit = dipole spacing\n" % float(g_waist_DS))
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
                if int(nplanes) == 0:
                        f.write(" 0 = NPLANES = number of scattering planes\n")
                if int(nplanes) > 0:
                        f.write(" %r = NPLANES = number of scattering planes\n" % int(nplanes))
                        for i in range(int(nplanes)):
                                phi = -180 + i * 360 / (int(nplanes) - 1)
                                f.write(" %r %r %r %r = phi, theta_min, theta_max (deg) for plane %r\n" % ( float(phi), float(self.theta_info[0]), float(self.theta_info[1]), float(self.theta_info[2]), i+1))
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
                I_0 = float(self.P_pu) / ( np.pi * ( float(self.waist(self.pump_um)) * 10**-9)**2 )*10**-9 #units of nW/m^2, then will get converted to W/m^2 for var.par
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
                file = open('makemetal.py', 'r')
                list_of_lines = file.readlines()  
                list_of_lines[2] = str("nback=")+str(self.n_R)+str("\n")
                list_of_lines[3] = str("diel_path='") + str(self.heat_dir)+str("' ")+str("\n")
                list_of_lines[4] = str("shapefile=np.loadtxt('")+str(shapefile)+str("', skiprows=7)")+str("\n")
                file = open('makemetal.py', 'w')
                file.writelines(list_of_lines)
                file.close

        ### Collect FML functions ###
        
        def file_len(self, fname):
                with open(fname) as f:
                        for i, l in enumerate(f):
                                pass
                return i + 1

        def field_MM(self, filename,skip):
            fmlfile = np.loadtxt(filename, skiprows=skip)
            theta_fml = (fmlfile[:,0])*np.pi/180 # radians
            phi_fml = fmlfile[:,1]*np.pi/180 # radians
            weights = np.zeros(len(theta_fml))+1
            f11 = fmlfile[:, 2] + 1j*fmlfile[:, 3]
            f21 = fmlfile[:, 4] + 1j*fmlfile[:, 5]
            f12 = 0; f22 = 0
            w = 1.240/self.probe_um # probe wavelength, eV
            k = w/hbar_eVs/c
            I0_SI = self.P_pr / ( np.pi * (self.waist(self.probe_um) * 2 * 10**-9)**2 ) # W/m^2 ## COME BACK TO HERE
            I0 = I0_SI*10**3 # g^1/2 / ( cm^1/2 s)
            E0 = np.sqrt(I0*8*np.pi/(c*self.n_R))

            magr = 1 # cm
            magk = np.linalg.norm(k)
            expon = np.exp(1j*magk*magr)
            E_theta = expon/(magk*magr) * ( f11 ) * E0
            E_phi = expon/(magk*magr) * ( f21 ) * E0
            E_r = 0
            Ex = np.zeros(len(theta_fml),dtype=complex)
            Ey = np.zeros(len(theta_fml),dtype=complex)
            Ez = np.zeros(len(theta_fml),dtype=complex)
            for i in range(0, len(theta_fml)):
                theta = theta_fml[i]
                phi = phi_fml[i]
                Ex[i] = E_r*np.cos(theta) - E_theta[i]*np.sin(theta)
                Ey[i] = E_r*np.sin(theta)*np.cos(phi) + E_theta[i]*np.cos(theta)*np.cos(phi) - E_phi[i]*np.sin(phi)
                Ez[i] = E_r*np.sin(theta)*np.sin(phi) + E_theta[i]*np.cos(theta)*np.sin(phi) + E_phi[i]*np.cos(phi)
            x = magr*np.cos(theta_fml)
            y = magr*np.sin(theta_fml)*np.cos(phi_fml)
            z = magr*np.sin(theta_fml)*np.sin(phi_fml)

            idx_thetamin = np.where(theta_fml*180/np.pi == float(self.theta_info[0]))
            idx_thetamax = np.where(theta_fml*180/np.pi == float(self.theta_info[1]))
            idx_phimin = np.where(phi_fml*180/np.pi == float(self.phi_info[0]))
            idx_phimax = np.where(phi_fml*180/np.pi == float(self.phi_info[1]))
            weights[idx_thetamin] = 0.5
            weights[idx_thetamax] = 0.5
            weights[idx_phimin] = 0.5
            weights[idx_phimax] = 0.5

            ### Convert theta_fml to theta 
            theta = np.arccos(z/magr)
            phi = np.arctan2(y, x)

            return np.array([Ex, Ey, Ez]), weights, theta, phi, theta_fml, phi_fml

        def E_gaussian(self,filename,skip):
                # for one beam position, calculate the field on the partial hemi
                fmlfile = np.loadtxt(filename, skiprows=skip)
                theta_fml = (fmlfile[:,0])*np.pi/180 # radians
                phi_fml = fmlfile[:,1]*np.pi/180 # radians
                magr = 1 # cm
                w = 1.240/self.probe_um # probe wavelength
                k = w*self.n_R/hbar_eVs/c
                waist_cm = self.waist(self.probe_um) * 1E-7 # [cm] beam waist radius
                ## position dependence 
                x_fml = magr*np.cos(theta_fml)
                y_fml = magr*np.sin(theta_fml)*np.cos(phi_fml)
                z_fml = magr*np.sin(theta_fml)*np.sin(phi_fml)
                x = x_fml-self.P_pr_offset # axial distance from beam's focus / waist 
                xR = np.pi*waist_cm**2*self.n_R/(self.probe_um*1E-4) # Rayleigh range, at distance xR, width of beam = np.sqrt(2) times larger than it is at beam waist
                wx = waist_cm*np.sqrt(1+(x/xR)**2) # radius at which field amp. falls to 1/e of their axial values
                Rx = x*(1 + (xR/x)**2) # radius of curvature
                phi_gouy = np.arctan(x/xR) # Gouy phase
                r = np.sqrt(y_fml**2 + z_fml**2) # radial distance from center of beam 
                Emag = waist_cm/wx*np.exp(-r**2/wx**2)*np.exp(-1j*(k*x+k*r**2/(2*Rx)-phi_gouy))
                return np.array([np.zeros(len(Emag)), Emag, np.zeros(len(Emag))])

        def calculated_IPT(self):
                IPT_TOT_HMR = np.zeros((int(self.image_width*2/(self.ss)+1), int(self.image_width*2/(self.ss)+1)))
                IPT_TOT_INT = np.zeros(IPT_TOT_HMR.shape)
                IPT_R = np.zeros(IPT_TOT_HMR.shape)
                IPT_one = np.zeros(IPT_TOT_HMR.shape)
                ycoords = np.zeros(IPT_TOT_HMR.shape)
                zcoords = np.zeros(IPT_TOT_HMR.shape)
                for valy in range(-self.image_width,self.image_width+self.ss,self.ss):
                        for valz in range(-self.image_width,self.image_width+self.ss,self.ss):
                                print(valy, valz)
                                base_file = str('raster_data/x0_y')+str(int(valy))+str('_z')+str(int(valz)) + \
                                            str('/fml_x0y')+str(int(valy))+str('z')+str(int(valz))+str('_')
                                lines_H = self.file_len(base_file+str('H'))
                                lines_R = self.file_len(base_file+str('R'))
                                base = 29; ndiff = lines_H-lines_R
                                E_H, weights, theta, phi, theta_fml, phi_fml = self.field_MM(filename=str(base_file+str('H')),skip=base+ndiff) 
                                E_R, _, _, _,_,_ = self.field_MM(filename=str(base_file+str('R')),skip=base)
                                E_p = self.E_gaussian(filename=str(base_file+str('R')),skip=base)
                                hot_minus_room = np.zeros(E_p.shape[1])
                                interf_term = np.zeros(E_p.shape[1])
                                one_term =  np.zeros(E_p.shape[1])+1

                                modEH_sqrd = np.sum(E_H.real**2 + E_H.imag**2, axis=0)
                                modER_sqrd = np.sum(E_R.real**2 + E_R.imag**2, axis=0)
                                hot_minus_room = modEH_sqrd - modER_sqrd
                                interf_term = 2*np.real( np.sum(E_p*np.conj(E_H - E_R),axis=0))

                                yi = int((valy + self.image_width)/self.ss)
                                zi = int((valz + self.image_width)/self.ss)
                                ycoords[yi, zi] = valy
                                zcoords[yi, zi] = valz

                                dtheta = float(self.theta_info[2])*np.pi/180 
                                dphi = float(self.phi_info[2])*np.pi/180

                                # Integrate 
                                IPT_TOT_HMR[yi, zi] = c*self.n_R/(8*np.pi)*( np.sum(hot_minus_room*np.sin(theta_fml)*dtheta*dphi*weights ) )
                                IPT_TOT_INT[yi, zi] = c*self.n_R/(8*np.pi)*( np.sum(interf_term*np.sin(theta_fml)*dtheta*dphi*weights ) )
                                IPT_one[yi, zi] = np.sum(np.abs(one_term)*np.sin(theta_fml)*dtheta*dphi*weights)
                return ycoords, zcoords, IPT_TOT_HMR, IPT_TOT_INT, IPT_one

        def collect_fml(self):
                ycoords, zcoords, IPT_HMR, IPT_INT, IPT_one = self.calculated_IPT()
                ywrite = np.ravel(ycoords); zwrite = np.ravel(zcoords)
                IPT_HMR_write = np.ravel(IPT_HMR)
                IPT_INT_write = np.ravel(IPT_INT)
                IPT_one_write = np.ravel(IPT_one)
                file = open(str('pt_signal.txt'),'w')
                file.write(str('y') + '\t' + str('z') + '\t' + str('H-R') + '\t' + str('INT') + '\n')
                for i in range(0, len(ywrite)):
                        file.write("%d \t %d \t %2.4e \t %2.4e \t %2.4e \n" % (ywrite[i],  zwrite[i], IPT_HMR_write[i],  IPT_INT_write[i], IPT_one_write[i]))





