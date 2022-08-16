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

    def dalphadn_atT0(self):
        ng_T0, _ = self.eps_gold_room(selected_waves=self.wave_probe)
        dn1_dT = 1E-4 + 1j*1E-4
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

    def pt_signal(self, which, norm, P0_probe=500*10, testing=False):
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
        if testing == True:
            return self.abs_cross, self.P0h, V, self.kappa,k, zR, zp, self.rth, self.waist(wave=self.wave_pump),self.waist(wave=self.wave_probe), interf_term, scatt_term1, scatt_term2
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
