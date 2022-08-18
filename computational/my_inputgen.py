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

                self.DS = int(val[par.index("lat_space")])
                self.n_R = float(val[par.index("n_back")])
                self.image_width = int(int(val[par.index("raster_length")])/2)
                self.ss = int(val[par.index("stepsize")])
                self.probe_um = float(val[par.index("wave_probe")])
                self.NA = float(val[par.index("NA")])
                self.P_pr = float(val[par.index("P_pr")])





                # self.rt_dir = val[par.index("rt_dir")]
                # self.heat_dir = val[par.index("heat_dir")]

                # self.num_k = int(val[par.index("num_k")])
                # self.k_back = float(val[par.index("k_back")])
                # self.k_in = float(val[par.index("k_in")])
                # self.k_sub = float(val[par.index("k_sub")])
                # self.pump_um = float(val[par.index("pump_um")])
                # self.P_pu = float(val[par.index("P_pu")])
                # self.P_pu_pol = val[par.index("P_pu_pol")]
                # self.P_pu_offset = float(val[par.index("P_pu_offset")])*self.DS*1E-7 # units of cm 
                # self.P_pr_pol = val[par.index("P_pr_pol")]
                self.P_pr_offset = float(val[par.index("P_pr_offset")])*self.DS*1E-7 # units of cm

                
                self.phi_info = val[par.index("phi_info")]
                self.theta_info = val[par.index("theta_info")]
                self.num_theta = (float(self.theta_info[1])-float(self.theta_info[0]))/float(self.theta_info[2])
                self.num_phi= (float(self.phi_info[1])-float(self.phi_info[0]))/float(self.phi_info[2])




                # self.n_amb_heat = float(val[par.index("n_ambient_heated")])

        def waist(self,wave):
                # [nm]
                return wave * 10**3 * 0.6 / self.NA 

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
                                filename_H = str('raster_data_H/y')+str(int(valy))+str('_z')+str(int(valz)) + \
                                            str('/w000r000k000.fml')
                                filename_R = str('raster_data_R/y')+str(int(valy))+str('_z')+str(int(valz)) + \
                                            str('/w000r000k000.fml')

                                lines_H = self.file_len(filename_H)
                                lines_R = self.file_len(filename_R)
                                base = 30; ndiff = lines_H-lines_R
                                E_H, weights, theta, phi, theta_fml, phi_fml = self.field_MM(filename=str(filename_H),skip=base+ndiff) 
                                E_R, _, _, _,_,_ = self.field_MM(filename=str(filename_R),skip=base)
                                E_p = self.E_gaussian(filename=str(filename_R),skip=base)
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
                return ycoords, zcoords, IPT_TOT_HMR, IPT_TOT_INT

        def collect_fml(self):
                ycoords, zcoords, IPT_HMR, IPT_INT = self.calculated_IPT()
                ywrite = np.ravel(ycoords); zwrite = np.ravel(zcoords)
                IPT_HMR_write = np.ravel(IPT_HMR)
                IPT_INT_write = np.ravel(IPT_INT)
                file = open(str('pt_signal.txt'),'w')
                file.write(str('y') + '\t' + str('z') + '\t' + str('H-R') + '\t' + str('INT') + '\n')
                for i in range(0, len(ywrite)):
                        file.write("%d \t %d \t %2.4e \t %2.4e \n" % (ywrite[i],  zwrite[i], IPT_HMR_write[i],  IPT_INT_write[i]))
