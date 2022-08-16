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
            ext_insum[idx,:] = (2*ni+1)*np.real(a_n[idx,:]+b_n[idx,:])
            sca_insum[idx,:] = (2*ni+1)*(np.abs(a_n[idx,:])**2+np.abs(b_n[idx,:])**2)
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
                    delta_z, 
                    zprobe_focus,
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
        self.nb_T0 = nb_T0
        self.delta_z = delta_z
        self.zprobe_focus = zprobe_focus
        self.kappa = 0.6*(1E7/100) # erg/ (s cm K)
        self.C = (1.26*2.35*1E7) # erg / (cm^3 K)
        self.Omega = 1E5 # 1/s (100 kHz)
        self.rth = np.sqrt(2*self.kappa/(self.Omega*self.C))
        self.shell_radius = self.rth 
        self.dnbdT = -10**(-4)

    def waist_at_focus(self, wave):
        NA = 1.25
        return wave * 0.6 / NA

    def waist_at_z(self, wave, z, z_at_focus): # [cm]
        waist_at_focus = self.waist_at_focus(wave=wave)
        waist_z = waist_at_focus*np.sqrt(1+( (z-z_at_focus)/self.zR(wave=wave))**2 )
        return waist_z

    def convert_k(self, wave):
        return 2*np.pi*self.nb_T0/wave

    def zR(self, wave):
        waist_at_focus = self.waist_at_focus(wave=wave)
        return np.pi*waist_at_focus**2*self.nb_T0/wave

    def eps_gold_room(self, selected_waves,drude=False):
        JCdata = np.loadtxt('auJC_interp.tab',skiprows=3)
        wave = np.round(JCdata[:,0]*1E-4,7) # cm
        n_re = JCdata[:,1]
        n_im = JCdata[:,2]
        idx = np.where(np.in1d(wave, selected_waves))[0]
        n = n_re[idx] + 1j*n_im[idx]
        eps = n_re[idx]**2 - n_im[idx]**2 +1j*(2*n_re[idx]*n_im[idx])
        return n, eps

    def qi(self, ri):
        xi = 2*np.pi*ri/self.wave_probe
        return 1/3*(1-(xi)**2 - 1j*2/9*(xi)**3)

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

    def dalphadT_atT0(self):
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
        zpump_focus = self.zprobe_focus - self.delta_z
        waist_at_focus_pump = self.waist_at_focus(wave=self.wave_pump)
        waist_at_NP_pump = self.waist_at_z(wave=self.wave_pump, z=0, z_at_focus=zpump_focus)
        I_at_focus = 2*self.P0h/(np.pi*waist_at_focus_pump**2)
        I = I_at_focus*(waist_at_focus_pump/waist_at_NP_pump)**2
        return self.abs_cross*I

    def pt_signal(self, which, norm, P0_probe=500*10, testing=False):
        ''' Photothermal Signal
        which: 'sin' or 'cos'
        norm: True or False. True = normalize it as usual, False = don't normalize it
        '''
        waist_at_focus_probe = self.waist_at_focus(wave=self.wave_probe)
        k = self.convert_k(self.wave_probe) # cm^-1
        zR = self.zR(wave=self.wave_probe)
        zp = self.zprobe_focus
        I1 = (np.exp(1) - 2*np.cos(1)) / (4*np.exp(1)) 
        I2 = (2*np.sin(1) - np.cos(1)) / (4*np.exp(1)) 
        if which == 'sin': I = I1 
        if which == 'cos': I = I2
        if self.kind == 'gold_sph_QS' or self.kind == 'gold_sph_MW':
            V = 4/3*np.pi*self.radius**3
        else:
            V = 4/3*np.pi*self.shell_radius**3
        guoy = np.exp(1j*np.arctan2(-zp, zR))
        interf_term = 4*k/(waist_at_focus_probe**2*np.sqrt(1+(zp/zR)**2))*\
                      np.imag(guoy*np.conj(self.dalphadT_atT0()))*\
                      self.Pabs()*self.rth**2/(2*self.kappa*V)*I


        scatt_term1 = k**4/(zR**2+zp**2)*\
                     2*np.real(np.conj(self.alpha_atT0())*self.dalphadT_atT0())*\
                     self.Pabs()*self.rth**2/(2*self.kappa*V)*I


        scatt_term2 = k**4/(zR**2+zp**2)*\
                      np.abs(self.dalphadT_atT0())**2*\
                     (self.Pabs()*self.rth**2/(2*self.kappa*V))**2*I

        return interf_term, scatt_term1, scatt_term2, zR


