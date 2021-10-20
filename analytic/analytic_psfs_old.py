import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec

e = 4.80326E-10 # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10 # speed of light [cm/s]
hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]

class Mie_Theory:
    def __init__(self, radius, selected_waves=np.arange(3.50e-05, 0.000100, 1E-7)):
        """Defines the different system parameters.
        
        Keyword arguments:
        radius -- [cm] radius of NP 
        n -- [unitless] refractive index of background 
        wave_pump -- [cm] wavelength of driving laser 
        """
        self.radius = radius
        self.selected_waves = selected_waves
        self.n = 1.473

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
        JCdata = np.loadtxt('auJC_interp.tab',skiprows=3)
        wave_raw = JCdata[:,0]*1E-4 # cm
        n_re_raw = JCdata[:,1]
        n_im_raw = JCdata[:,2]
        idx = np.where( np.in1d( np.round(wave_raw,7), np.round(self.selected_waves,7) ))[0]
        n_re = n_re_raw[idx]
        n_im = n_im_raw[idx]
        m = (n_re + 1j*n_im)/self.n
        k = 2*np.pi*self.n/self.selected_waves
        x = k*self.radius
        numer_a = m*self.psi(n,m*x)*self.psi_prime(n,x) - self.psi(n,x)*self.psi_prime(n,m*x)
        denom_a = m*self.psi(n,m*x)*self.xi_prime(n,x) - self.xi(n,x)*self.psi_prime(n,m*x)
        numer_b = self.psi(n,m*x)*self.psi_prime(n,x) - m*self.psi(n,x)*self.psi_prime(n,m*x)
        denom_b = self.psi(n,m*x)*self.xi_prime(n,x) - m*self.xi(n,x)*self.psi_prime(n,m*x)
        an = numer_a/denom_a
        bn = numer_b/denom_b
        return an, bn, self.selected_waves 

    def cross_sects(self, nTOT):
        a_n = np.zeros((nTOT,len(self.selected_waves)), dtype=complex)
        b_n = np.zeros((nTOT,len(self.selected_waves)), dtype=complex)
        ni = np.arange(1,nTOT+1)
        for i in range(nTOT):
            a_n[i,:], b_n[i,:], _ = self.mie_coefficent(n=(i+1))
            ext_insum = (2*ni[i]+1)*np.real(a_n+b_n)
            sca_insum = (2*ni[i]+1)*(np.real(a_n*np.conj(a_n)+b_n*np.conj(b_n)))
        k = 2*np.pi*self.n/self.selected_waves
        C_ext = 2 * np.pi/(k**2)*np.sum(ext_insum, axis=0)
        C_sca = 2 * np.pi/(k**2)*np.sum(sca_insum, axis=0)
        C_abs = C_ext - C_sca
        return C_abs, C_sca, self.selected_waves

    def find_dipole_res(self):
        C_abs_1, C_sca_1,_ = self.cross_sects(nTOT=1)
        C_abs_10, C_sca_10,_ = self.cross_sects(nTOT=10)
        idx_abs = np.where(C_abs_1 == max(C_abs_1))
        idx_sca = np.where(C_sca_1 == max(C_sca_1))
        return self.selected_waves[idx_abs][0], C_abs_10[idx_abs][0], self.selected_waves[idx_sca][0], C_sca_10[idx_sca][0]


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
                    define_zp='single',
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
        self.nb_T0 = 1.473
        self.kappa = 0.6*(1E7/100) # erg/ (s cm K)
        self.C = (1.26*2.35*1E7) # erg / (cm^3 K)
        self.Omega = 1E5 # 1/s (100 kHz)
        self.f_thph = 1 
        self.g_thph=1
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
        if self.define_zp == 'single':
            return np.pi*self.waist(wave=self.wave_probe)**2*self.nb_T0/self.wave_probe
        else:
            return self.define_zp

    def eps_gold_room(self, selected_waves):
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
        f=r1**3/r2**3
        return r2**3*nb**2*( (n2**2 - nb**2)*(n1**2 + 2*n2**2) + f*(n1**2 - n2**2)*(nb**2 + 2*n2**2)) \
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

    def d_alpha_sph_MW_dn(self, n, r, q):
        nb = self.nb_T0
        return 2*n*nb**4*r**3 / (nb**2*(-1+q) - n**2*q)**2

    #############################################################
    #### d alpha / dn1 ####

    def d_alpha_CS_QS_dn1(self, n1, n2, r1, r2):
        nb = self.nb_T0
        f = r1**3/r2**3
        return 54*f*n1*n2**4*nb**4*r2**3 /\
            (-2*(-1+f)*n2**4+2*(2+f)*n2**2*nb**2+n1**2*((1+2*f)*n2**2-2*(-1+f)*nb**2))**2

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
        return (6*(-1+f)*n2*((1+2*f)*n1**4-4*(-1+f)*n1**2*n2**2+2*(2+f)*n2**4)*nb**4*r2**3)/\
                ((-2(-1+f)*n2**4+2*(2+f)*n2**2*nb**2+n1**2((1+2*f)*n2**2-2*(-1+f)*nb**2))**2)

    def d_alpha_CS_MW_dn2(self, n1, n2, r1, r2, q1, q2):
        nb=self.nb_T0
        return (2*n2*nb**4*r2**3*((n1**2-n2**2)**2*(-1+q2)*q2*r1**6 + \
            (n1**4*q1*(1-2*q2)+n2**4*(-1 + q1 + 2 q2 - 2 q1 q2) + \
            2*n1**2*n2**2*(-q2+q1*(-1+2*q2)))*r1**3*r2**3 + \
            (n2**2*(-1+q1)-n1**2*q1)**2*r2**6)) /\
            (3*((n1-n2)*(n1+n2)*(n2-nb)*(n2+nb)*(-1+q2)*q2*r1**3 +\
            (n2**2*(-1+q1)-n1**2*q1)*(-nb**2*(-1+q2)+n2**2*q2)*r2**3)**2)

    ########################################################

    def alpha_atT0(self):
        ng_T0, _ = self.eps_gold_room(selected_waves=self.wave_probe)

        if self.kind == 'glyc_sph_QS':
            alpha_atT0 = self.alpha_sph_QS(n=self.nb_T0, r=self.rth)

        if self.kind == 'gold_sph_QS':
            alpha_atT0 = self.alpha_sph_QS(n=ng_T0, r=self.radius)

        if self.kind == 'glyc_sph_MW':
            alpha_atT0 = self.alpha_sph_MW(n=self.nb_T0, r=self.rth,q=self.q(self.rth))

        if self.kind == 'gold_sph_MW':
            alpha_atT0 = self.alpha_sph_MW(n=ng_T0, r=self.radius,q=self.qi(self.radius))

        if self.kind == 'coreshell_QS':
            alpha_atT0 = self.alpha_CS_QS(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.rth)

        if self.kind == 'coreshell_MW':
            alpha_atT0 = self.alpha_CS_MW(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.rth,
                                          q1=self.qi(self.radius), q2=self.qi(self.rth))
        return alpha_atT0

    ########################################################

    def dalphadT0(self):
        ng_T0, _ = self.eps_gold_room(selected_waves=self.wave_probe)
        dn1_dT = -1E-4 - 1j*1E-4;  
        dn2_dT = self.dnbdT

        if self.kind == 'glyc_sph_QS':
            dgold_dT_atT0 = 0
            dglyc_dT_atT0 = self.d_alpha_sphQS_dn(n=self.nb_T0, r=self.rth)

        if self.kind == 'gold_sph_QS':
            dgold_dT_atT0 = self.d_alpha_sphQS_dn(n=ng_T0, r=self.radius)
            dglyc_dT_atT0 = 0

        if self.kind == 'glyc_sph_MW':
            dgold_dT_atT0 = 0
            dglyc_dT_atT0 = self.d_alpha_sphMW_dn(n=self.nb_T0, r=self.rth, q=self.qi(r=self.rth))

        if self.kind == 'gold_sph_MW':
            dgold_dT_atT0 = self.d_alpha_sphMW_dn(n=self.ng_T0, r=self.radius, q=self.qi(r=self.radius))
            dglyc_dT_atT0 = 0

        if self.kind == 'coreshell_QS':
            dgold_dT_atT0 = self.d_alpha_CS_QS_dn1(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.rth)
            dglyc_dT_atT0 = self.d_alpha_CS_QS_dn2(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.rth)

        if self.kind == 'coreshell_MW':
            dgold_dT_atT0 = self.d_alpha_CS_MW_dn1(n1=ng_T0, n2=self.nb_T0, r1=self.radius, r2=self.rth,
                                                   q1=self.qi(self.radius), q2=self.qi(self.rth))
        return (dgold_dT_atT0*dn1_dT + dglyc_dT_atT0*dn2_dT)

    def Pabs(self):
        w_pump = self.waist(wave=self.wave_pump) # cm
        I0 = 2*self.P0h/(np.pi*w_pump**2)
        return self.abs_cross*I0

    def alphaPT(self, which):
        if which == 'sin': const2 = 0.0183; const1=0.366
        if which == 'cos': const2 = 0.0263; const1=0.053
        alpha0 = self.alpha_T0()
        C1=const1; C2=const2
        V = 4/3*np.pi*self.shell_radius**3
        return alpha0*C1/np.sqrt(C2), self.dalphadT0()*self.Pabs()*self.rth**2/self.kappa*np.sqrt(C2)/V

    def probe_inteference(self,which):
        w_probe = self.waist(wave=self.wave_probe) # cm
        k = self.convert_k(self.wave_probe) # cm^-1
        zR = np.pi*w_probe**2*self.nb_T0/self.wave_probe
        zp = self.zp()
        if which == 'sin': const = 0.366
        if which == 'cos': const = 0.053
        V = 4/3*np.pi*self.shell_radius**3

        guoy = 1/np.sqrt(1+(-zp/zR)**2) + 1j*(-zp/zR)/np.sqrt(1+(-zp/zR)**2)

        signal = 8*np.pi*self.f_thph*self.nb_T0/(self.wave_probe*w_probe**2)*\
                    1/np.sqrt(1+(zp/zR)**2)*self.Pabs()*self.rth**2/self.kappa*\
                    np.imag(-1*guoy*np.conj(self.dalphadT0())*const/V\
                    )
        alpha_PT1, alpha_PT2 = self.alphaPT(which=which)
        alphaPT_tot = alpha_PT1 + alpha_PT2
        if which == 'sin': const2 = 0.0183; const1=0.366
        if which == 'cos': const2 = 0.0263; const1=0.053
        C1=const1; C2=const2

        woa = 4*self.f_thph/(w_probe**2)*1/np.sqrt(1+(zp/zR)**2)*k*\
                np.imag(-guoy* (np.conj(alpha_PT2) * C1/np.sqrt(C2) ))

        return woa#signal

    def self_inteference(self,which):
        w_probe = self.waist(wave=self.wave_probe) # cm
        k = self.convert_k(self.wave_probe) # cm^-1

        zR = np.pi*w_probe**2*self.nb_T0/self.wave_probe
        zp=self.zp()
        if which == 'sin': const2 = 0.0183; const1=0.366
        if which == 'cos': const2 = 0.0263; const1=0.053
        V = 4/3*np.pi*self.shell_radius**3
        signal = 16*np.pi**2*self.g_thph*self.nb_T0**2/(self.wave_probe**2*w_probe**4)*\
                    1/(1+(zp/zR)**2)*self.Pabs()*self.rth**2/self.kappa*(\
                    2*np.real( np.conj( self.alpha_T0() ) * self.dalphadT0() )*const1/V +\
                    np.abs(self.dalphadT0())**2 * self.Pabs()*self.rth**2/self.kappa * const2/V**2)

        alpha_PT1, alpha_PT2 = self.alphaPT(which=which)

        if which == 'sin': const2 = 0.0183; const1=0.366
        if which == 'cos': const2 = 0.0263; const1=0.053
        C1=const1; C2=const2


        alphaPT_tot = alpha_PT1 + alpha_PT2
        woa = self.g_thph*k**4/(zR**2+zp**2)*\
                ( np.abs(alphaPT_tot)**2-np.abs(alpha_PT1)**2 )

        return signal

    def polarizability_res(self, whichalpha, wave_probe):
        ng_T0, _ = self.eps_gold_room(selected_waves=wave_probe)
        k = self.convert_k(wave_probe) # cm^-1
        r1 = self.radius
        r2 = self.shell_radius
        if whichalpha == 'CM':
            alpha0 = self.alpha_T0()

        if whichalpha == 'pt':
            V = 4/3*np.pi*self.shell_radius**3
            alpha0 = self.alpha_T0()
            which = 'sin'
            if which == 'sin': const2 = 0.0183; const1=0.366
            if which == 'cos': const2 = 0.0263; const1=0.053
            C1=const1; C2=const2
            alpha = alpha0*C1/np.sqrt(C2) + self.dalphadT0()*self.Pabs()*self.rth**2/self.kappa*np.sqrt(C2)/V

        # scat = 8*np.pi/3 * (k/self.nb_T0)**4 * ( np.abs(alpha)**2-np.abs(alpha0*C1/np.sqrt(C2))**2)
        # ext = 4*np.pi *  (k/self.nb_T0) * np.imag(alpha-alpha0*C1/np.sqrt(C2))

        scat = 8*np.pi/3 * (k/self.nb_T0)**4 * ( np.abs(alpha)**2 )
        ext = 4*np.pi * (k/self.nb_T0) * np.imag(alpha)

        return scat, ext


##################################################################################################
##################################################################################################
### JUST PLOT ALPHA SCAT & ABS ###
##################################################################################################
##################################################################################################

def scat_ext_of_alpha_PT():
    val_rad = 75.E-7; nR = 1.473

    waverange = np.round(np.arange(450,800,10)*1E-7,7)
    sca_varypu = np.zeros(len(waverange)); ext_varypu = np.zeros(len(waverange))
    sca_varypr = np.zeros(len(waverange)); ext_varypr = np.zeros(len(waverange))

    # Find resonance 
    mt_atres = Mie_Theory(val_rad, nR)
    wave_abs_res, C_abs_res, wave_sca_res, C_sca_res = mt_atres.find_dipole_res()

    for idx, val_wave_pump in enumerate(waverange):
        wave_probe_fixed = np.round(wave_sca_res,7)
        mt_atpump = Mie_Theory(val_rad, nR, np.array([val_wave_pump]))
        abs_cross_val, _, _ = mt_atpump.cross_sects(nTOT=10)

        pt = Photothermal_Image(val_rad, nR, val_wave_pump, abs_cross_val, 500*10, wave_probe_fixed, 'core')
        sca_varypu[idx], ext_varypu[idx] = pt.polarizability_res(whichalpha='pt', wave_probe=wave_probe_fixed)

    for idx, wave_probe in enumerate(waverange):
        wave_pump_fixed = wave_abs_res
        pt_MW = Photothermal_Image(val_rad, nR, wave_pump_fixed, C_abs_res, 500*10, wave_probe, 'core')
        sca_varypr[idx], ext_varypr[idx]  = pt.polarizability_res(whichalpha='pt', wave_probe=wave_probe)

    fig, ax = plt.subplots(2, 1, figsize=(3,5))# sharey=True)

    # Mie theory
    mt = Mie_Theory(val_rad, 1.473, waverange)
    abs_cross, sca_cross, _ = mt.cross_sects(nTOT=1)

    ax[1].plot(waverange*1E7, abs_cross/max(abs_cross), color='tab:red', label='abs 10')
    ax[1].plot(waverange*1E7, sca_cross/max(sca_cross), color='tab:purple',label='sca 10')


    ax2 = ax[0].twinx()
    ax[0].plot(waverange*1E7, ext_varypu, alpha=.5, color='tab:blue', label='ext')
    ax2.plot(waverange*1E7, sca_varypu, alpha=.5, color='tab:orange', label='sca')

    ax[0].set_xlabel('Pump Wavelength [nm]')
    ax[0].set_ylabel('Extinction [$\mu$m$^2$]',color='tab:blue')
    ax[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax2.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    ax2.set_ylabel('Scattering [$\mu$m$^2$]',color='tab:orange')

    ax3 = ax[1].twinx()
    ax[1].plot(waverange*1E7, ext_varypr/max(ext_varypr), alpha=.5, color='tab:blue', label='ext')
    ax3.plot(waverange*1E7, sca_varypr/max(sca_varypr), alpha=.5, color='tab:orange', label='sca')


    ax[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[1].set_xlabel('Probe Wavelength [nm]')
    ax[1].set_ylabel('Extinction [$\mu$m$^2$]',color='tab:blue')
    ax3.set_ylabel('Scattering [$\mu$m$^2$]',color='tab:orange')

    plt.suptitle(str(int(val_rad*1E7)) + ' nm')
    plt.subplots_adjust(hspace=.4, left=.25, right=.8,top=.95)
    plt.show()

# scat_ext_of_alpha_PT()


##################################################################################################
##################################################################################################
### SWEEP POWER ### 
##################################################################################################
##################################################################################################

def sweep_power(nR):
    # a = np.arange(10, 51, 10)*1E-7
    a=np.array([10, 20, 75])*1E-7
    pump_pow = np.arange(0, 1001, 200)*1E-6*1E7 # microWatts*(convertion to erg/s)
    # pump_pow = np.array([200*10])
    PI_sin_NM = np.zeros((len(a), len(pump_pow)))
    PI_cos_NM = np.zeros((len(a), len(pump_pow)))
    SI_sin_NM = np.zeros((len(a), len(pump_pow)))
    SI_cos_NM = np.zeros((len(a), len(pump_pow)))

    for idx_rad, val_rad in enumerate(a):
        mt = Mie_Theory(val_rad, nR)
        wave_abs_res, C_abs_res, wave_sca_res, C_sca_res = mt.find_dipole_res()
        wave_probe = wave_sca_res

        wave_pump = wave_abs_res
        for idx_pow, val_pow in enumerate(pump_pow):
            pt_NM = Photothermal_Image(val_rad, nR, wave_pump, C_abs_res, val_pow, wave_probe, 'core')
            PI_sin[idx_rad, idx_pow] = pt.probe_inteference(which='sin')
            PI_cos[idx_rad, idx_pow] = pt.probe_inteference(which='cos')
            SI_sin[idx_rad, idx_pow] = pt.self_inteference(which='sin')
            SI_cos[idx_rad, idx_pow] = pt.self_inteference(which='cos')


    fig = plt.figure(1, figsize=[5,6])
    gs = GridSpec(3,1, figure=fig, hspace=.35)
    gs0 = gs[0:2].subgridspec(2,2)
    ax1 = fig.add_subplot(gs0[0,0])
    ax2 = fig.add_subplot(gs0[0,1])
    ax3 = fig.add_subplot(gs0[1,0])
    ax4 = fig.add_subplot(gs0[1,1])
    gs1 = gs[2].subgridspec(1,1)
    ax5 = fig.add_subplot(gs1[0,0])

    n = 10#len(a)
    c = plt.cm.tab10(np.linspace(0,1,n))
    for idx, val in enumerate(a):
        ax1.plot(pump_pow*0.1, PI_sin_NM[idx,:], alpha=.5,color=c[idx])
        ax2.plot(pump_pow*0.1, PI_cos_NM[idx,:], alpha=.5,color=c[idx])
        ax3.plot(pump_pow*0.1, SI_sin_NM[idx,:], alpha=.5,color=c[idx])
        ax4.plot(pump_pow*0.1, SI_cos_NM[idx,:], alpha=.5,color=c[idx])

        ax1.plot(pump_pow*0.1, PI_sin_QS[idx,:],linestyle='dashed', alpha=.7, color=c[idx])
        ax2.plot(pump_pow*0.1, PI_cos_QS[idx,:],linestyle='dashed', alpha=.7, color=c[idx])
        ax3.plot(pump_pow*0.1, SI_sin_QS[idx,:],linestyle='dashed', alpha=.7, color=c[idx])
        ax4.plot(pump_pow*0.1, SI_cos_QS[idx,:],linestyle='dashed', alpha=.7, color=c[idx])

        ax1.plot(pump_pow*0.1, PI_sin_MW[idx,:], linestyle='dotted', color=c[idx])
        ax2.plot(pump_pow*0.1, PI_cos_MW[idx,:], linestyle='dotted', color=c[idx])
        ax3.plot(pump_pow*0.1, SI_sin_MW[idx,:], linestyle='dotted', color=c[idx])
        ax4.plot(pump_pow*0.1, SI_cos_MW[idx,:], linestyle='dotted', color=c[idx])

        ax5.plot(pump_pow*0.1, np.sqrt( (PI_sin_NM[idx,:]+SI_sin_NM[idx,:])**2 + \
                                        (PI_cos_NM[idx,:]+SI_cos_NM[idx,:])**2 ),\
                                        #/ max(np.sqrt( (PI_sin_NM[idx,:]+SI_sin_NM[idx,:])**2 + \
                                        #(PI_cos_NM[idx,:]+SI_cos_NM[idx,:])**2 )),
                                        alpha=.5, color=c[idx],  
                                        )

        ax5.plot(pump_pow*0.1, np.sqrt( (PI_sin_QS[idx,:]+SI_sin_QS[idx,:])**2 + \
                                        (PI_cos_QS[idx,:]+SI_cos_QS[idx,:])**2 ),\
                                        #/ max(np.sqrt( (PI_sin_QS[idx,:]+SI_sin_QS[idx,:])**2 + \
                                        #(PI_cos_QS[idx,:]+SI_cos_QS[idx,:])**2 )),
                                        linestyle='dashed',color=c[idx], alpha=.7, 
                                        )

        ax5.plot(pump_pow*0.1, np.sqrt( (PI_sin_MW[idx,:]+SI_sin_MW[idx,:])**2 + \
                                        (PI_cos_MW[idx,:]+SI_cos_MW[idx,:])**2 ),\
                                        #/ max(np.sqrt( (PI_sin_MW[idx,:]+SI_sin_MW[idx,:])**2 + \
                                        #(PI_cos_MW[idx,:]+SI_cos_MW[idx,:])**2 )),
                                        linestyle='dotted', color=c[idx],
                                        )

    ax5.plot(-1, 1, 'k',alpha=.5,label='no metal')
    ax5.plot(-1, 1, 'k', linestyle='dashed', label='QS')
    ax5.plot(-1, 1, 'k', linestyle='dashdot',label='MLWA')

    ax1.set_title('PI Sin')
    ax1.set_xticklabels([])
    ax2.set_title('PI Cos')
    ax2.set_xticklabels([])
    ax3.set_title('SI Sin')
    ax3.set_xticklabels([])
    ax4.set_title('SI Cos')
    ax4.set_xticklabels([])

    plt.legend()
    ax5.set_xlabel('Power [$\mu W$]')
    ax5.set_ylabel('Signal')
    plt.subplots_adjust(hspace=.2)
    plt.show()
# sweep_power(nR=1.473) 


#####################################################################################
#####################################################################################
### PT SPECTRA (SWEEP PROBE) AGAINST MIE ###
#####################################################################################
#####################################################################################

def plot_spectra_sweep(radius, ax, whichalpha, pump, probe):
    waverange = np.arange(400, 750, 5)*1E-7
    PI_sin = np.zeros(len(waverange)); 
    PI_cos = np.zeros(len(waverange))
    SI_sin = np.zeros(len(waverange)); 
    SI_cos = np.zeros(len(waverange))

    for idx, wave_val in enumerate(waverange):
        if pump == 'fixed': 
            wave_pump = 532.E-7
        if pump == 'sweep': 
            wave_pump = np.round(wave_val, 7)
        if pump == 'track_res': 
            mt = Mie_Theory(radius, waverange)
            w_abs_res, _, _, _ = mt.find_dipole_res()
            wave_pump = w_abs_res

        mt_single = Mie_Theory(radius, np.array([wave_pump]))
        abs_cross, _, _ = mt_single.cross_sects(nTOT=10)

        if probe == 'fixed':
            wave_probe = 1000.E-7
        if probe == 'sweep':
            wave_probe = np.round(wave_val, 7)
        if probe == 'track_res': 
            mt = Mie_Theory(radius, waverange)
            _, _, w_sca_res, _ = mt.find_dipole_res()
            wave_probe = w_sca_res

        pt = Photothermal_Image(radius, np.round(wave_pump,7), abs_cross, 
                                1000*10, np.round(wave_probe,7), whichalpha)

        PI_sin[idx] = pt.probe_inteference(which='sin'); 
        PI_cos[idx] = pt.probe_inteference(which='cos')
        SI_sin[idx] = pt.self_inteference(which='sin'); 
        SI_cos[idx] = pt.self_inteference(which='cos')

    # Mie theory
    mt = Mie_Theory(radius, waverange)
    abs_cross_MIE, _, _ = mt.cross_sects(nTOT=10)
    _, sca_cross_MIE, _ = mt.cross_sects(nTOT=1)

    ax.plot(waverange*1E7, abs_cross_MIE/max(abs_cross_MIE), color='tab:blue', label='Abs $l$=10')
    ax.plot(waverange*1E7, sca_cross_MIE/max(sca_cross_MIE), color='tab:orange',label='Sca $l$=1')

    # Photothermal
    ax2 = ax.twinx()
    if pump == 'sweep': c='tab:purple'
    if probe == 'sweep': c='tab:red'
    ax2.plot(waverange*1E7, np.sqrt( (SI_sin+PI_sin)**2 + (SI_cos+PI_cos)**2),
            color=c, label='PT Signal')
    ax.plot(1,1,color=c, label='PT Signal')


    ax.set_xlim(400, 750)
    ax.set_title(str(int(np.round(radius*1E7)))+str(' nm'))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
    return c
################################################################################################
################################################################################################

def make_figure_of_plots():
    radii = np.array([5, 10])*1E-7
    fig, ax = plt.subplots(int(len(radii)), 1, figsize=(4,7),sharex=True,)
    for idx, radii in enumerate(radii):
        print(int(radii*1E7))
        c = plot_spectra_sweep(radius=radii, ax=ax[idx], 
                            whichalpha='core_QS', 
                            pump='fixed', # sweep, fixed, or track_res
                            probe='sweep'
                            )
    ax[0].legend(frameon=False)

    if c == 'tab:purple':
        ax[-1].set_xlabel('Pump Wavelength [nm]',fontsize=12)
    if c == 'tab:red':
        ax[-1].set_xlabel('Probe Wavelength [nm]',fontsize=12)

    plt.subplots_adjust(left=.2,hspace=.3, top=.965, bottom=.07,right=.9,wspace=.2)
    plt.show()

make_figure_of_plots() 
