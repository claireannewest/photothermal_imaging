import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%.0f"  # Give format here

e = 4.80326E-10 # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10 # speed of light [cm/s]
hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]

class Mie_Theory:
    def __init__(self, radius, nb, selected_waves):
        """Defines the different system parameters.
        
        Keyword arguments:
        radius -- [cm] radius of NP 
        n -- [unitless] refractive index of background 
        wave_pump -- [cm] wavelength of driving laser 
        """
        self.radius = radius
        self.selected_waves = selected_waves
        self.nb = nb

    def psi(self, n, rho):
        return rho*spherical_jn(n, rho)

    def psi_prime(self, n, rho):
        return spherical_jn(n, rho) + rho*spherical_jn(n, rho, derivative=True)

    def hankel(self, n, rho):
        return spherical_jn(n, rho) + 1j*spherical_yn(n, rho)

    def hankel_prime(self, n, rho):
        return spherical_jn(n, rho, derivative=True) + 1j*spherical_yn(n, rho,derivative=True)

    def xi(self, n, rho):
        return rho*self.hankel(n, rho)

    def xi_prime(self, n, rho):
        return self.hankel(n, rho) + rho*self.hankel_prime(n, rho)

    def mie_coefficent(self, n):
        ''' Calculates Mie coefficients, a and b. 

        n: multipole label (n = 1 is the dipole)
        '''
        JCdata = np.loadtxt('auJC_interp.tab',skiprows=3)
        wave_raw = JCdata[:,0]*1E-4 # cm
        n_re_raw = JCdata[:,1]
        n_im_raw = JCdata[:,2]
        idx = np.where( np.in1d( np.round(wave_raw,7), np.round(self.selected_waves,7) ))[0]
        n_re = n_re_raw[idx]
        n_im = n_im_raw[idx]
        m = (n_re + 1j*n_im)/self.nb
        k = 2*np.pi*self.nb/self.selected_waves
        x = k*self.radius
        numer_a = m*self.psi(n,m*x)*self.psi_prime(n,x) - self.psi(n,x)*self.psi_prime(n,m*x)
        denom_a = m*self.psi(n,m*x)*self.xi_prime(n,x) - self.xi(n,x)*self.psi_prime(n,m*x)
        numer_b = self.psi(n,m*x)*self.psi_prime(n,x) - m*self.psi(n,x)*self.psi_prime(n,m*x)
        denom_b = self.psi(n,m*x)*self.xi_prime(n,x) - m*self.xi(n,x)*self.psi_prime(n,m*x)
        an = numer_a/denom_a
        bn = numer_b/denom_b
        return an, bn, self.selected_waves 

    def cross_sects(self, nTOT):
        ''' Calculates Mie cross-sections. 

        nTOT: total number of multipoles (nTOT = 1 is just dipole)
        '''
        a_n = np.zeros((nTOT,len(self.selected_waves)), dtype=complex)
        b_n = np.zeros((nTOT,len(self.selected_waves)), dtype=complex)
        ext_insum = np.zeros((len( np.arange(1,nTOT+1) ),len(self.selected_waves)))
        sca_insum = np.zeros((len( np.arange(1,nTOT+1) ),len(self.selected_waves)))
        nTOT_ar = np.arange(1, nTOT+1)
        for idx, ni in enumerate(nTOT_ar):
            a_n[idx,:], b_n[idx,:], _ = self.mie_coefficent(n=ni)
            ext_insum[idx,:] = (2*(ni+1)+1)*np.real(a_n[idx,:]+b_n[idx,:])
            sca_insum[idx,:] = (2*(ni+1)+1)*(np.abs(a_n[idx,:])**2+np.abs(b_n[idx,:])**2)
        k = 2*np.pi*self.nb/self.selected_waves
        C_ext = 2 * np.pi/(k**2)*np.sum(ext_insum, axis=0)
        C_sca = 2 * np.pi/(k**2)*np.sum(sca_insum, axis=0)
        C_abs = C_ext - C_sca
        return C_abs, C_sca, self.selected_waves

    def find_resonance(self,nTOT):
        C_abs, C_sca, _ = self.cross_sects(nTOT=nTOT)
        idx_abs = np.where(C_abs == max(C_abs))
        idx_sca = np.where(C_sca == max(C_sca))
        return self.selected_waves[idx_abs][0], C_abs[idx_abs][0], self.selected_waves[idx_sca][0], C_sca[idx_sca][0]


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

class Photothermal_Image:
    def __init__(self, 
                    radius, 
                    wave_pump,
                    abs_cross, 
                    P0h, 
                    wave_probe,
                    kind,
                    nb_T0,
                    define_zp=np.array([]),
                    ):
        """Defines the different system parameters.

        Keyword arguments:
        radius -- [cm] radius of NP 
        n -- [unitless] refractive index of background 
        wave_pump -- [cm] wavelength of driving laser 
        """
        self.radius = radius # cm
        self.wave_pump = wave_pump # cm 
        self.abs_cross = abs_cross # cm^2
        self.P0h = P0h # erg/s 
        self.wave_probe = wave_probe # cm
        self.kind = kind
        self.define_zp = define_zp 
        self.nb_T0 = nb_T0
        self.kappa = 0.6*(1E7/100) # erg/ (s cm K)
        self.C = (1.26*2.35*1E7) # erg / (cm^3 K)
        self.Omega = 1E5 # 1/s (100 kHz)
        self.rth = np.sqrt(2*self.kappa/(self.Omega*self.C))
        self.shell_radius = self.rth 
        self.dnbdT = -10**(-4)

    def waist(self, wave): # [cm]
        # if wave == self.wave_pump: return 360*1E-7
        # if wave == self.wave_probe: return 630*1E-7
        NA = 1.25
        return wave * 0.6 / NA 

    def convert_k(self, wave):
        return 2*np.pi*self.nb_T0/wave

    def zp(self):
        if len(self.define_zp) == 0:
            return np.pi*self.waist(wave=self.wave_probe)**2*self.nb_T0/self.wave_probe
        else:
            return self.define_zp

    def zR(self):
        return np.pi*self.waist(self.wave_probe)**2*self.nb_T0/self.wave_probe

    def eps_gold_room(self, selected_waves,drude=False):
        JCdata = np.loadtxt('auJC_interp.tab',skiprows=3)
        wave = np.round(JCdata[:,0]*1E-4,7) # cm
        n_re = JCdata[:,1]
        n_im = JCdata[:,2]
        idx = np.where(np.in1d(wave, selected_waves))[0]
        n = n_re[idx] + 1j*n_im[idx]
        eps = n_re[idx]**2 - n_im[idx]**2 +1j*(2*n_re[idx]*n_im[idx])
        if drude == True:
            drude = np.loadtxt('agDrudeFit_101421.tab',skiprows=3)
            wave = np.round(drude[:,0]*1E-4,7) # cm
            eps_re = drude[:,1]
            eps_im = drude[:,2]
            eps_c = eps_re + 1j*eps_im
            mod_eps_c = np.sqrt(eps_re**2 + eps_im**2)
            n_re = np.sqrt( (mod_eps_c + eps_re )/ 2)
            n_im = np.sqrt( (mod_eps_c - eps_re )/ 2)

            idx = np.where(np.in1d(wave, selected_waves))[0]
            n = n_re[idx] + 1j*n_im[idx]
            eps = eps_c[idx] + 1j*eps_c[idx]
        return n, eps

    def qi(self, ri):
        xi = 2*np.pi*ri/self.wave_probe
        return 1/3*(1-(xi)**2 - 1j*2/3*(xi)**3)

    #############################################################
    #### ALPHA ####

    def alpha_sph_QS(self,n,r):
        nb = self.nb_T0
        return r**3*nb**2*(n**2-nb**2)/(n**2+2*nb**2)

    def alpha_sph_MW(self,n, r, q):
        nb = self.nb_T0
        return 1/3*r**3*nb**2*(n**2-nb**2)/(nb**2 + q*(n**2 - nb**2))

    def alpha_CS_QS(self, n1, n2, r1, r2):
        nb = self.nb_T0
        f = r1**3/r2**3
        return r2**3*nb**2*( (n2**2 - nb**2)*(n1**2 + 2*n2**2) + f*(n1**2 - n2**2)*(nb**2 + 2*n2**2) ) \
                    / ( (n2**2 + 2*nb**2)*(n1**2 + 2*n2**2) + 2*f*(n2**2 - nb**2)*(n1**2 - n2**2) )
         
    def alpha_CS_MW(self, n1, n2, r1, r2, q1, q2):
        e1 = n1**2;
        e2 = n2**2
        eb = self.nb_T0**2
        return 1/3*r2**3*eb*( (e2-eb)*(e1*q1-e2*(q1-1))*r2**3 - (e1-e2)*(e2*(q2-1)-eb*q2)*r1**3 )/\
                  ( (e1*q1-e2*(q1-1))*(e2*q2-eb*(q2-1))*r2**3 - (e1-e2)*(e2-eb)*q2*(q2-1)*r1**3 )

    #############################################################
    #### d alpha / dn ####

    def d_alpha_sphQS_dn(self,n,r):
        nb = self.nb_T0
        return 6*n*nb**4*r**3 / (n**2+2*nb**2)**2

    def d_alpha_sphMW_dn(self, n, r, q):
        nb = self.nb_T0
        return 2*n*nb**4*r**3 / (3*(nb**2*(-1+q) - n**2*q)**2)

    #############################################################
    #### d alpha / dn1 ####

    def d_alpha_CS_QS_dn1(self, n1, n2, r1, r2):
        nb = self.nb_T0
        f = r1**3/r2**3
        return 54*f*n1*n2**4*nb**4*r2**3 /\
            ((-2*(-1+f)*n2**4+2*(2+f)*n2**2*nb**2+n1**2*((1+2*f)*n2**2-2*(-1+f)*nb**2))**2)

    def d_alpha_CS_MW_dn1(self, n1, n2, r1, r2, q1, q2):
        nb = self.nb_T0
        return 2*n1*n2**4*nb**4*r1**3*r2**6 / \
                (3*((n1-n2)*(n1+n2)*(n2-nb)*(n2+nb)*(-1+q2)*q2*r1**3 +\
                (n2**2*(-1+q1)-n1**2*q1)*(-nb**2*(-1+q2)+n2**2*q2)*r2**3)**2)

   #############################################################
    #### d alpha / dn2 at T0 ####

    def d_alpha_CS_QS_dn2(self, n1, n2, r1, r2):
        nb = self.nb_T0
        f = r1**3/r2**3
        return -(6*(-1+f)*n2*((1+2*f)*n1**4 - 4*(-1+f)*n1**2*n2**2 + 2*(2+f)*n2**4)*nb**4*r2**3)/\
                ((-2*(-1+f)*n2**4 + 2*(2+f)*n2**2*nb**2+n1**2*((1+2*f)*n2**2-2*(-1+f)*nb**2))**2)

    def d_alpha_CS_MW_dn2(self, n1, n2, r1, r2, q1, q2):
        nb=self.nb_T0
        return (2*n2*nb**4*r2**3*((n1**2-n2**2)**2*(-1+q2)*q2*r1**6 + \
            (n1**4*q1*(1-2*q2)+n2**4*(-1 + q1 + 2*q2 - 2*q1*q2) + \
            2*n1**2*n2**2*(-q2+q1*(-1+2*q2)))*r1**3*r2**3 + \
            (n2**2*(-1+q1)-n1**2*q1)**2*r2**6)) /\
            (3*((n1-n2)*(n1+n2)*(n2-nb)*(n2+nb)*(-1+q2)*q2*r1**3 +\
            (n2**2*(-1+q1)-n1**2*q1)*(-nb**2*(-1+q2)+n2**2*q2)*r2**3)**2)

    ########################################################

    def alpha_atT0(self):
        ng_T0, _ = self.eps_gold_room(selected_waves=self.wave_probe)

        if self.kind == 'glyc_sph_QS':
            alpha_atT0 = self.alpha_sph_QS(n=self.nb_T0, r=self.shell_radius)

        if self.kind == 'gold_sph_QS':
            alpha_atT0 = self.alpha_sph_QS(n=ng_T0, r=self.radius)

        if self.kind == 'glyc_sph_MW':
            alpha_atT0 = self.alpha_sph_MW(n=self.nb_T0, r=self.shell_radius,q=self.qi(self.shell_radius))

        if self.kind == 'gold_sph_MW':
            alpha_atT0 = self.alpha_sph_MW(n=ng_T0, r=self.radius,q=self.qi(self.radius))

        if self.kind == 'coreshell_QS':
            alpha_atT0 = self.alpha_CS_QS(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.shell_radius)

        if self.kind == 'coreshell_MW':
            alpha_atT0 = self.alpha_CS_MW(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.shell_radius,
                                          q1=self.qi(self.radius), q2=self.qi(self.shell_radius))
        return alpha_atT0

    ########################################################

    def dalphadn_atT0(self):
        ng_T0, _ = self.eps_gold_room(selected_waves=self.wave_probe)
        dn1_dT = -1E-4 - 1j*1E-4
        dn2_dT = self.dnbdT

        if self.kind == 'glyc_sph_QS':
            dgold_dT_atT0 = 0
            dglyc_dT_atT0 = self.d_alpha_sphQS_dn(n=self.nb_T0, r=self.shell_radius)

        if self.kind == 'gold_sph_QS':
            dgold_dT_atT0 = self.d_alpha_sphQS_dn(n=ng_T0, r=self.radius)
            dglyc_dT_atT0 = 0

        if self.kind == 'glyc_sph_MW':
            dgold_dT_atT0 = 0
            dglyc_dT_atT0 = self.d_alpha_sphMW_dn(n=self.nb_T0, r=self.shell_radius, q=self.qi(ri=self.shell_radius))

        if self.kind == 'gold_sph_MW':
            dgold_dT_atT0 = self.d_alpha_sphMW_dn(n=ng_T0, r=self.radius, q=self.qi(ri=self.radius))
            dglyc_dT_atT0 = 0

        if self.kind == 'coreshell_QS':
            dgold_dT_atT0 = self.d_alpha_CS_QS_dn1(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.shell_radius)
            dglyc_dT_atT0 = self.d_alpha_CS_QS_dn2(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.shell_radius)

        if self.kind == 'coreshell_MW':
            dgold_dT_atT0 = self.d_alpha_CS_MW_dn1(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.shell_radius,
                                                   q1=self.qi(self.radius), q2=self.qi(self.shell_radius))
            dglyc_dT_atT0 = self.d_alpha_CS_MW_dn2(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.shell_radius,
                                                   q1=self.qi(self.radius), q2=self.qi(self.shell_radius))

        return (dgold_dT_atT0*dn1_dT + dglyc_dT_atT0*dn2_dT)

    def Pabs(self):
        w_pump = self.waist(wave=self.wave_pump) # cm
        I0 = 2*self.P0h/(np.pi*w_pump**2)
        return self.abs_cross*I0

    def pt_signal(self, which, norm, P0_probe=500*10):
        ''' Photothermal Signal
        which: 'sin' or 'cos'
        norm: True or False. True = normalize it as usual, False = don't normalize it
        '''
        w_probe = self.waist(wave=self.wave_probe) # cm
        k = self.convert_k(self.wave_probe) # cm^-1
        zR = self.zR()
        zp = self.zp()
        if which == 'sin': 
            c_lin = 0.0366
            c_quad = 0.0183
        if which == 'cos': 
            c_lin = 0.0525
            c_quad = 0.0263
        if self.kind == 'gold_sph_QS' or self.kind == 'gold_sph_MW':
            V = 4/3*np.pi*self.radius**3
        else:
            V = 4/3*np.pi*self.shell_radius**3
        th=0; ph=0
        f_thph = 1 - np.sin(th)**2 * np.cos(ph)**2
        g_thph = f_thph**2
        guoy = 1/np.sqrt(1+(-zp/zR)**2) + 1j*(-zp/zR)/np.sqrt(1+(-zp/zR)**2)
        interf_term = 4*k*f_thph/(w_probe**2*np.sqrt(1+(zp/zR)**2))*\
                      np.imag(guoy*np.conj(self.dalphadn_atT0()))*\
                      self.Pabs()*self.rth**2/(self.kappa*V)*c_lin
        scatt_term1 = k**4/(zR**2+zp**2)*\
                     2*np.real(np.conj(self.alpha_atT0())*self.dalphadn_atT0())*\
                     self.Pabs()*self.rth**2/(self.kappa*V)*c_lin
        scatt_term2 = k**4/(zR**2+zp**2)*\
                      np.abs(self.dalphadn_atT0())**2*\
                     (self.Pabs()*self.rth**2/(self.kappa*V))**2*c_quad
        if norm == False:
            w_probe = self.waist(wave=self.wave_probe) # cm
            zR = self.zR()
            z = 1
            I0_probe = P0_probe/(np.pi*w_probe**2)
            E0_probe = np.sqrt(I0_probe*8*np.pi/(c*self.nb_T0))
            E_ref_sqrd = np.abs(zR/z*E0_probe)**2
            interf_term = interf_term*E_ref_sqrd
            scatt_term1 = scatt_term1*E_ref_sqrd
            scatt_term2 = scatt_term2*E_ref_sqrd

        return interf_term, scatt_term1, scatt_term2, zR



    # def h_minus_r(self, P0_probe=500*10):
    #     w_probe = self.waist(wave=self.wave_probe) # cm
    #     k = self.convert_k(self.wave_probe) # cm^-1
    #     zR = self.zR()
    #     zp = self.zp()
    #     z=1
    #     guoy = 1/np.sqrt(1+(-zp/zR)**2) + 1j*(-zp/zR)/np.sqrt(1+(-zp/zR)**2)
    #     I0_probe = P0_probe/(np.pi*w_probe**2)
    #     E0 = np.sqrt(I0_probe*8*np.pi/(c*self.nb_T0))
    #     E_atNP = E0/np.sqrt(1+(zp/zR)**2)*np.exp(-1j*k*zp)*guoy
    #     r=z
    #     G = k**2*np.exp(1j*k*r)/r
    #     T = self.Pabs()/(8*np.pi*self.kappa*self.rth)
    #     alpha_H = self.alpha_atT0() + self.dalphadn_atT0()*T
    #     alpha_R = np.zeros(len(alpha_H))+self.alpha_atT0()
    #     EH = G*alpha_H*E_atNP
    #     ER = G*alpha_R*E_atNP
    #     Epr = np.zeros(len(EH))+E0/(z/zR)*np.exp(1j*k*(z-zp))*(-1j)
    #     return c*self.nb_T0/(8*np.pi)*( np.abs(EH)**2 - np.abs(ER)**2 + \
    #             2*np.real(np.dot(Epr, np.conj(EH-ER)) ) )



    def cobri(self, cobri_or_scat, P0_probe=500*10):
        w_probe = self.waist(wave=self.wave_probe) # cm
        k = self.convert_k(self.wave_probe) # cm^-1
        zR = self.zR()
        zp = self.zp()
        z=10
        guoy = 1/np.sqrt(1+(-zp/zR)**2) + 1j*(-zp/zR)/np.sqrt(1+(-zp/zR)**2)
        I0_probe = P0_probe/(np.pi*w_probe**2)
        E0 = np.sqrt(I0_probe*8*np.pi/(c*self.nb_T0))
        E_atNP = E0/np.sqrt(1+(zp/zR)**2)*np.exp(-1j*k*zp)*guoy
        r=z
        G = k**2*np.exp(1j*k*r)/r
        alpha_R = self.alpha_atT0()
        ER = G*alpha_R*E_atNP
        Epr = E0/(z/zR)*np.exp(1j*k*(z-zp))*(-1j)
        if cobri_or_scat == 'cobri':
            return c*self.nb_T0/(8*np.pi)*(np.abs(Epr + ER)**2 - np.abs(Epr)**2)
        if cobri_or_scat == 'scat':
            return c/(self.nb_T0*8*np.pi)*(np.abs(ER)**2)


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

class Plot_Everything:
    def __init__(self, 
                    radius, 
                    nb
                    ):
        self.radius = radius
        self.nb = nb



############################################################################################################
###### Photothermal Spectra ########
############################################################################################################
    # def pi_si_terms(self, 
    #                 norm_sig,
    #                 wave_pump, 
    #                 wave_probe, 
    #                 whichalpha, 
    #                 nTOT_abs, 
    #                 nTOT_sca,
    #                 power, 
    #                 fig, 
    #                 ax,
    #                 plot_self_diff, # True = two self terms, False = added together
    #                 sep_sincos,
    #                 P0_probe=500*10,
    #                 waverange=np.round(np.arange(400, 700, 1)*1E-7, 7),
    #                 define_zp=np.array([]),
    #                 ):
    #     radius = self.radius
    #     for idx, rad_val in enumerate(radius):
    #         mt_single = Mie_Theory(rad_val, wave_pump)
    #         abs_cross, _, _ = mt_single.cross_sects(nTOT=nTOT_abs)
    #         ############################################################
    #         pt = Photothermal_Image(rad_val, np.round(wave_pump,7), abs_cross, 
    #                                 power*10, np.round(wave_probe,7), whichalpha, define_zp)
    #         PI_sin, SI1_sin, SI2_sin, _ = pt.pt_signal(which='sin',norm=norm_sig,P0_probe=P0_probe)
    #         PI_cos, SI1_cos, SI2_cos, _ = pt.pt_signal(which='cos',norm=norm_sig,P0_probe=P0_probe)
    #         total_signal = np.sqrt( (PI_sin + SI1_sin + SI2_sin)**2 + (PI_cos + SI1_cos + SI2_cos)**2)
    #         ############################################################
    #         #### Plot Mie Theory ####
    #         mt = Mie_Theory(rad_val, waverange)
    #         abs_cross_MIE, _, _ = mt.cross_sects(nTOT=nTOT_abs)
    #         _, sca_cross_MIE, _ = mt.cross_sects(nTOT=nTOT_sca)
    #         ax[idx].fill_between(waverange*1E7, sca_cross_MIE, 
    #                          color='mediumpurple',zorder=0, alpha=.6)
    #         ax[idx].fill_between(waverange*1E7, abs_cross_MIE, 
    #                          color='gray',zorder=1, alpha=.6)
    #         idx_abs = np.where(abs_cross_MIE == max(abs_cross_MIE))
    #         max_abs = waverange[idx_abs]
    #         idx_sca = np.where(sca_cross_MIE == max(sca_cross_MIE))
    #         max_sca = waverange[idx_sca]
    #         ax[idx].plot(np.array([max_abs, max_abs])*1E7, 
    #             np.array([-1, 1]), color='k', linewidth=0.75, linestyle='dashed')
    #         ax[idx].plot(np.array([max_sca, max_sca])*1E7, 
    #             np.array([-1, 1]), color='k', linewidth=0.75, linestyle='dashed')

    #         ### Format Figure ###
    #         if max(sca_cross_MIE) > max(abs_cross_MIE): whichbig = sca_cross_MIE
    #         if max(sca_cross_MIE) < max(abs_cross_MIE): whichbig = abs_cross_MIE
    #         ax[idx].set_ylim([-max(whichbig)*1.1, max(whichbig)*1.1])
    #         ax[idx].set_xlim(400, 700)
    #         ax[idx].set_title(str('r = ')+str(int(np.round(rad_val*1E7)))+str(' nm'), pad=20)
    #         ax[idx].set_yticks([-max(whichbig), 0, max(whichbig)])
    #         yfmt = ScalarFormatterForceFormat()
    #         yfmt.set_powerlimits((0,0))
    #         ax[idx].yaxis.set_major_formatter(yfmt)
    #         #### Plot Photothermal ####
    #         ax2 = ax[idx].twinx()
    #         ax2.plot(waverange*1E7, total_signal,'k', zorder=2,linewidth=1.5)
    #         if sep_sincos == False:
    #             ax2.plot(waverange*1E7, PI_sin+PI_cos,'tab:blue',linewidth=1.5,zorder=0)
    #             if plot_self_diff == True:
    #                 ax2.plot(waverange*1E7, SI1_sin+SI1_cos,'tab:green',linewidth=1.5,zorder=0)
    #                 ax2.plot(waverange*1E7, SI2_sin+SI2_cos,'tab:orange',linewidth=1.5,zorder=0)
    #             if plot_self_diff == False:
    #                 ax2.plot(waverange*1E7, SI1_sin+SI1_cos+SI2_sin+SI2_cos,'tab:red',linewidth=1.5,zorder=0)
    #         if sep_sincos == True:
    #             ax2.plot(waverange*1E7, PI_sin,'tab:blue',linewidth=1.5,zorder=0)
    #             ax2.plot(waverange*1E7, PI_cos,'tab:orange',linewidth=1.5,zorder=0)
    #             ax2.plot(waverange*1E7, SI1_sin,'tab:green',linewidth=1.5,zorder=0)
    #             ax2.plot(waverange*1E7, SI1_cos,'tab:red',linewidth=1.5,zorder=0)
    #             ax2.plot(waverange*1E7, SI2_sin,'tab:purple',linewidth=1.5,zorder=0)
    #             ax2.plot(waverange*1E7, SI2_cos,'tab:brown',linewidth=1.5,zorder=0)
    #         ax2.axhline(0,0,color='k', linewidth=1)
    #         ax2.set_ylim([-max(total_signal)*1.6, max(total_signal)*1.6])
    #         ax2.set_yticks([-np.round(max(total_signal),1), 0, np.round(max(total_signal),1)])

    #     plt.subplots_adjust(left=.05, bottom=.2, right=.95, 
    #                         top=.8, wspace=.75) 
    #     plt.show()
    #     fig.savefig(str('sweep.png'), 
    #         dpi=500, bbox_inches='tight'
    #         )




############################################################################################################
######### Sweep zp #################
############################################################################################################
    # def sweep_zp(self, 
    #             norm_sig,
    #             wave_pump, 
    #             wave_probe, 
    #             whichalpha, 
    #             nTOT, 
    #             power, 
    #             fig, 
    #             ax,
    #             zp,
    #             P0_probe=500*10,
    #             ):
    #     for idx, val_rad in enumerate(self.radius):
    #         mt_single = Mie_Theory(val_rad, wave_pump)
    #         abs_cross, _, _ = mt_single.cross_sects(nTOT=nTOT)
    #         pt = Photothermal_Image(val_rad, wave_pump, abs_cross,
    #                                 power*10, wave_probe, whichalpha, zp)
    #         PI_sin, SI1_sin, SI2_sin, _ = pt.pt_signal(which='sin',norm=norm_sig,P0_probe=P0_probe)
    #         PI_cos, SI1_cos, SI2_cos, _ = pt.pt_signal(which='cos',norm=norm_sig,P0_probe=P0_probe)
    #         total_signal = np.sqrt( (PI_sin + SI1_sin + SI2_sin)**2 + (PI_cos + SI1_cos + SI2_cos)**2)
    #         ax[idx].plot(zp*1E7, total_signal,
    #                 'k', linewidth=1.5)
    #         ax[idx].axhline(0,color='k',linewidth=1)
    #         ax[idx].axvline(0,color='k',linewidth=1)
    #         ax[idx].set_title(str('r = ')+str(int(np.round(val_rad*1E7)))+str(' nm'))
    #     plt.subplots_adjust(left=.05, bottom=.2, right=.95, 
    #                     top=.8, wspace=.6,hspace=.2) 
    #     fig.savefig(str('sweep_zp.png'), 
    #         dpi=500, bbox_inches='tight'
    #         )
    #     plt.show()




############################################################################################################
###### zp Spectrum Images #########
############################################################################################################
    # def sweep_zp_image(self,
    #                 norm_sig, 
    #                 wave_pump, 
    #                 wave_probe, 
    #                 whichalpha, 
    #                 nTOT, 
    #                 power, 
    #                 fig, 
    #                 ax,
    #                 zp,
    #                 scalemin=0,
    #                 scalemax=1,
    #                 P0_probe=500*10,
    #                 ):
    #     radius=self.radius
    #     if len(wave_pump) > 1:
    #         wavesweep = wave_pump
    #         ax.set_xlabel('Pump Wavelength [nm]')
    #     if len(wave_pump) == 1:
    #         wavesweep = wave_probe
    #         ax.set_xlabel('Probe Wavelength [nm]')
    #     if len(wave_pump) > 1 & len(wave_probe) > 1:
    #         ax.set_xlabel('Pump and Probe Wavelength [nm]')
    #     total_signal = np.zeros((len(zp), len(wavesweep)))
    #     for idx, val_zp in enumerate(zp):
    #         mt_single = Mie_Theory(radius, wave_pump)
    #         abs_cross, _, _ = mt_single.cross_sects(nTOT=nTOT)
    #         pt = Photothermal_Image(radius, wave_pump, abs_cross,
    #                                 power*10, wave_probe, whichalpha, np.array([val_zp]))
    #         PI_sin, SI1_sin, SI2_sin, _ = pt.pt_signal(which='sin',norm=norm_sig,P0_probe=P0_probe)
    #         PI_cos, SI1_cos, SI2_cos, _ = pt.pt_signal(which='cos',norm=norm_sig,P0_probe=P0_probe)
    #         total_signal[idx,:] = np.sqrt( (PI_sin + SI1_sin + SI2_sin)**2 + (PI_cos + SI1_cos + SI2_cos)**2)
    #     idx_max = np.where(total_signal == np.max(total_signal))
    #     print('Global maximum at zp:', int(np.round(zp[idx_max[0]][0]*1E7)),\
    #         'nm, wavelength:', int(np.round(wave_pump[idx_max[1]][0]*1E7)), 'nm')
    #     im = ax.imshow(total_signal,
    #             extent=[min(wavesweep)*1E7,max(wavesweep)*1E7,min(zp)*1E7,max(zp)*1E7],
    #             cmap=plt.get_cmap('RdPu'),
    #             aspect='auto',
    #             vmin=0+scalemin,
    #             vmax=np.max(total_signal)*scalemax,
    #             )
    #     ax.set_ylabel('$z_p$ [nm]')
    #     plt.colorbar(im,ax=ax,label='PT Signal')
    #     ax.set_title(str('r = ')+str(int(np.round(radius*1E7)))+str(' nm'))
    #     plt.subplots_adjust(left=.2)
    #     fig.savefig(str('zp_image.png'), 
    #         dpi=500, bbox_inches='tight'
    #         )
    #     plt.show()




############################################################################################################
###### zp Find Max #########
############################################################################################################
    def sweepwave_atzpmax(self, 
                norm_sig, # True = normalize it, False = don't normalize
                wave_pump, 
                wave_probe, 
                whichalpha, 
                nTOT_abs, 
                nTOT_sca,                
                power, 
                fig, 
                ax,
                plot_scatt_diff, # True = two self terms, False = added together
                sep_sincos,
                include_zpmax,
                waverange,
                zp=np.arange(-1000, 1000, 10)*1E-7,
                P0_probe=500*10,
                ):
        radius = self.radius
        if include_zpmax == True:
            fig_zp, ax_zp = plt.subplots(1, 5, figsize=(9.3,1.8),sharex=True)
        if len(wave_pump) > 1:
            wave_pump_array = wave_pump
            wave_probe_array = np.zeros(len(waverange))+wave_probe
            ax[0].set_xlabel('Pump Wavelength [nm]')
        if len(wave_pump) == 1:
            wave_pump_array = np.zeros(len(waverange))+wave_pump
            wave_probe_array = wave_probe
            ax[0].set_xlabel('Probe Wavelength [nm]')
        if ((len(wave_pump)) > 1 and (len(wave_probe) > 1)):
            wave_probe_array = wave_pump
            wave_probe_array = wave_probe
            ax[0].set_xlabel('Pump and Probe Wavelength [nm]')
        ############################################################
        ### Loop Through the Radii ###
        for idxr, rad_val in enumerate(radius):
            total_signal_max = np.zeros(len(waverange))
            ISIN_max = np.zeros(len(waverange))
            ICOS_max = np.zeros(len(waverange))
            S1SIN_max = np.zeros(len(waverange))
            S2SIN_max = np.zeros(len(waverange))
            S1COS_max = np.zeros(len(waverange))
            S2COS_max = np.zeros(len(waverange))
            zp_max = np.zeros(len(waverange))
            ############################################################
            #### Plot Mie Theory ####
            mt = Mie_Theory(radius=rad_val, nb=self.nb, selected_waves=waverange)
            abs_cross_MIE, _, _ = mt.cross_sects(nTOT=nTOT_abs)
            _, sca_cross_MIE, _ = mt.cross_sects(nTOT=nTOT_sca)
            ax[idxr].fill_between(waverange*1E7, sca_cross_MIE, 
                             color='mediumpurple',zorder=0, alpha=.6)
            ax[idxr].fill_between(waverange*1E7, abs_cross_MIE, 
                             color='gray',zorder=1, alpha=.6)
            idx_abs = np.where(abs_cross_MIE == max(abs_cross_MIE))
            max_abs = waverange[idx_abs]
            idx_sca = np.where(sca_cross_MIE == max(sca_cross_MIE))
            max_sca = waverange[idx_sca]
            ax[idxr].plot(np.array([max_abs, max_abs])*1E7, 
                np.array([-1, 1]), color='k', linewidth=0.75, linestyle='dashed')
            ax[idxr].plot(np.array([max_sca, max_sca])*1E7, 
                np.array([-1, 1]), color='k', linewidth=0.75, linestyle='dashed')
            ############################################################
            ### Format Figure ###
            if max(sca_cross_MIE) > max(abs_cross_MIE): whichbig = sca_cross_MIE
            if max(sca_cross_MIE) < max(abs_cross_MIE): whichbig = abs_cross_MIE
            ax[idxr].set_ylim([0, max(whichbig)*1.1])
            ax[idxr].set_xlim(min(waverange*1E7), max(waverange*1E7))
            ax[idxr].set_title(str('r = ')+str(int(np.round(rad_val*1E7)))+str(' nm'), pad=20)
            ax[idxr].set_yticks([0, max(whichbig)])
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0,0))
            ax[idxr].yaxis.set_major_formatter(yfmt)
            ###########################################################
            ### Find zp to Maximize PT ####
            for idxw, val_wave in enumerate(waverange):
                mt_single = Mie_Theory(rad_val, self.nb, np.array([wave_pump_array[idxw]]))
                abs_cross, _, _ = mt_single.cross_sects(nTOT=nTOT_abs)
                pt = Photothermal_Image(radius=rad_val, wave_pump=np.array([wave_pump_array[idxw]]),
                                         abs_cross=abs_cross, P0h=power*10, wave_probe=wave_probe_array[idxw], 
                                         kind=whichalpha, nb_T0=self.nb, define_zp=zp)
                I_sin, S1_sin, S2_sin, _ = pt.pt_signal(which='sin', norm=norm_sig, P0_probe=P0_probe)
                I_cos, S1_cos, S2_cos, _ = pt.pt_signal(which='cos',norm=norm_sig, P0_probe=P0_probe)
                total_signal = np.sqrt( (I_sin + S1_sin + S2_sin)**2 + 
                                       (I_cos + S1_cos + S2_cos)**2)
                zp_idx_max = np.where(total_signal == max(total_signal))
                zp_max[idxw] = zp[zp_idx_max]
                total_signal_max[idxw] = total_signal[zp_idx_max]
                ISIN_max[idxw] = I_sin[zp_idx_max]
                ICOS_max[idxw] = I_cos[zp_idx_max]
                S1SIN_max[idxw] = S1_sin[zp_idx_max]
                S2SIN_max[idxw] = S2_sin[zp_idx_max]
                S1COS_max[idxw] = S1_cos[zp_idx_max]
                S2COS_max[idxw] = S2_cos[zp_idx_max]
            ############################################################
            #### Plot Photothermal ####
            ax2 = ax[idxr].twinx()
            if idxr == len(radius)-1:
                ax2.set_ylabel('PT Signal')

            ax2.plot(waverange*1E7, total_signal_max,'k', zorder=2,linewidth=1.5)

            if sep_sincos == False:
                ax2.plot(waverange*1E7, ISIN_max+ICOS_max,'tab:blue',linewidth=1.5,zorder=0)

                if plot_scatt_diff == True:
                    ax2.plot(waverange*1E7, S1SIN_max+S1COS_max,'tab:green',linewidth=1.5,zorder=0)
                    ax2.plot(waverange*1E7, S2SIN_max+S2COS_max,'tab:orange',linewidth=1.5,zorder=0)

                if plot_scatt_diff == False:
                    ax2.plot(waverange*1E7, S1SIN_max+S1COS_max+S2SIN_max+S2COS_max,'tab:red',linewidth=1.5,zorder=0)

            if sep_sincos == True:
                ax2.plot(waverange*1E7, ISIN_max,'tab:blue',linewidth=1.5,zorder=0)
                ax2.plot(waverange*1E7, ICOS_max,'tab:orange',linewidth=1.5,zorder=0)
                ax2.plot(waverange*1E7, S1SIN_max,'tab:green',linewidth=1.5,zorder=0)
                ax2.plot(waverange*1E7, S1COS_max,'tab:red',linewidth=1.5,zorder=0)
                ax2.plot(waverange*1E7, S2SIN_max,'tab:purple',linewidth=1.5,zorder=0)
                ax2.plot(waverange*1E7, S2COS_max,'tab:brown',linewidth=1.5,zorder=0)
            

            ax[0].set_ylabel('Cross-sections [$\mu$m$^2$]')
            ax2.set_yticks([0, max(total_signal_max)])
            ax2.set_yticklabels([0, max(total_signal_max)])
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0,0))
            ax2.yaxis.set_major_formatter(yfmt)

            if include_zpmax == True:
                ax_zp[idxr].plot(waverange*1E7, zp_max*1E7)
                ax_zp[idxr].set_title(str('r = ')+str(int(np.round(rad_val*1E7)))+str(' nm'), pad=20)
                ax_zp[idxr].set_ylabel('$z_p$ [nm]')

        fig.subplots_adjust(left=.05, bottom=.2, right=.95, 
                            top=.8, wspace=.75) 
        fig_zp.subplots_adjust(left=.05, bottom=.2, right=.95, 
                            top=.8, wspace=.75) 
        fig.savefig(str('sweep_atzpmax.png'), 
            dpi=500, bbox_inches='tight')
        fig_zp.savefig(str('sweep_atzpmax_showzp.png'), 
            dpi=500, bbox_inches='tight')



# ###################################################################################
# ###### Compare time dynamics to steady state ######
# ###################################################################################
#     def compare_td_to_ss(self, 
#                     norm_sig,
#                     wave_pump, 
#                     wave_probe, 
#                     whichalpha, 
#                     nTOT_abs, 
#                     nTOT_sca,
#                     power, 
#                     fig, 
#                     ax,
#                     P0_probe=500*10,
#                     waverange=np.round(np.arange(400, 700, 100)*1E-7, 7),
#                     define_zp=np.array([]),
#                     ):
#         radius = self.radius
#         for idx, rad_val in enumerate(radius):
#             mt_single = Mie_Theory(rad_val, wave_pump)
#             abs_cross, _, _ = mt_single.cross_sects(nTOT=nTOT_abs)
#             ############################################################
#             ### Time Dynamics Explicit ###
#             pt = Photothermal_Image(rad_val, np.round(wave_pump,7), abs_cross, 
#                                     power*10, np.round(wave_probe,7), whichalpha, define_zp)
#             PI_sin, SI1_sin, SI2_sin, _ = pt.pt_signal(which='sin',norm=norm_sig,P0_probe=P0_probe)
#             PI_cos, SI1_cos, SI2_cos, _ = pt.pt_signal(which='cos',norm=norm_sig,P0_probe=P0_probe)
#             total_signal = np.sqrt( (PI_sin + SI1_sin + SI2_sin)**2 + (PI_cos + SI1_cos + SI2_cos)**2)
#             ############################################################
#             #### Plot Photothermal ####
#             ax2 = ax[idx].twinx()
#             ax2.plot(waverange*1E7, total_signal,'k', zorder=2,linewidth=1.5)
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0,0))
#             ax2.yaxis.set_major_formatter(yfmt)
#             ############################################################
#             ### Steady-State ###
#             steady_sig = pt.h_minus_r()
#             ax[idx].plot(waverange*1E7, steady_sig)
#             yfmt = ScalarFormatterForceFormat()
#             yfmt.set_powerlimits((0,0))
#             ax[idx].yaxis.set_major_formatter(yfmt)
#             ax[idx].set_title(str('r = ')+str(int(np.round(rad_val*1E7)))+str(' nm'), pad=20)
#         fig.subplots_adjust(left=.05, bottom=.2, right=.95, 
#                             top=.8, wspace=.75) 
#         plt.show()
#         fig.savefig(str('sweep.png'), 
#             dpi=500, bbox_inches='tight'
#             )

