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
    def __init__(self, radius, selected_waves=np.arange(450.e-07, 700.e-07, 1E-7)):
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

    def find_resonance(self,nTOT):
        C_abs, C_sca, _ = self.cross_sects(nTOT=nTOT)
        idx_abs = np.where(C_abs == max(C_abs))
        idx_sca = np.where(C_sca == max(C_sca))
        return self.selected_waves[idx_abs][0], C_abs[idx_abs][0], self.selected_waves[idx_sca][0], C_sca[idx_sca][0]


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
        if len(self.define_zp) == 0:
            return 1*np.pi*self.waist(wave=self.wave_probe)**2*self.nb_T0/self.wave_probe
        else:
            return self.define_zp

    def zR(self):
        return np.pi*self.waist(self.wave_probe)**2*self.nb_T0/self.wave_probe

    def eps_gold_room(self, selected_waves):
        JCdata = np.loadtxt('auJC_interp.tab',skiprows=3)
        wave = np.round(JCdata[:,0]*1E-4,7) # cm
        n_re = JCdata[:,1]
        n_im = JCdata[:,2]
        idx = np.where(np.in1d(wave, selected_waves))[0]
        n = n_re[idx] + 1j*n_im[idx]
        eps = n_re[idx]**2 - n_im[idx]**2 +1j*(2*n_re[idx]*n_im[idx])
        # drude = np.loadtxt('agDrudeFit_101421.tab',skiprows=3)
        # wave = np.round(drude[:,0]*1E-4,7) # cm
        # eps_re = drude[:,1]
        # eps_im = drude[:,2]
        # eps_c = eps_re + 1j*eps_im
        # mod_eps_c = np.sqrt(eps_re**2 + eps_im**2)
        # n_re = np.sqrt( (mod_eps_c + eps_re )/ 2)
        # n_im = np.sqrt( (mod_eps_c - eps_re )/ 2)

        # idx = np.where(np.in1d(wave, selected_waves))[0]
        # n = n_re[idx] + 1j*n_im[idx]
        # eps = eps_c[idx] + 1j*eps_c[idx]
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

    def pt_signal(self,which):
        w_probe = self.waist(wave=self.wave_probe) # cm
        k = self.convert_k(self.wave_probe) # cm^-1
        zR = self.zR()
        zp = self.zp()
        if which == 'sin': const2 = 0.0183; const1=0.366
        if which == 'cos': const2 = 0.0263; const1=0.053
        if self.kind == 'gold_sph_QS' or self.kind == 'gold_sph_MW':
            V = 4/3*np.pi*self.radius**3
        else:
            V = 4/3*np.pi*self.shell_radius**3
        guoy = 1/np.sqrt(1+(-zp/zR)**2) + 1j*(-zp/zR)/np.sqrt(1+(-zp/zR)**2)

        probe_signal = 8*np.pi*self.f_thph*self.nb_T0/(self.wave_probe*w_probe**2)*\
                    1/np.sqrt(1+(zp/zR)**2)*self.Pabs()*self.rth**2/self.kappa*\
                    np.imag(-1*guoy*np.conj(self.dalphadn_atT0())*const1/V\
                    )

        self_signal = 16*np.pi**2*self.g_thph*self.nb_T0**2/(self.wave_probe**2*w_probe**4)*\
                    1/(1+(zp/zR)**2)*self.Pabs()*self.rth**2/self.kappa*(\
                    2*np.real( np.conj( self.alpha_atT0() ) * self.dalphadn_atT0() )*const1/V +\
                    np.abs(self.dalphadn_atT0())**2 * self.Pabs()*self.rth**2/self.kappa * const2/V**2
                    )
        return probe_signal, self_signal, zR

#####################################################################################
#####################################################################################
def pi_si_terms():
    radius = np.array([10, 20, 30, 40, 50])*1E-7
    fig, ax = plt.subplots(1, 5, figsize=(9.3,1.8),sharex=True)

    # radius = np.array([10, 20, 75])*1E-7
    # fig, ax = plt.subplots(1, 3, figsize=(7,1.8),sharex=True)

    waverange = np.round(np.arange(400, 700, 1)*1E-7, 7)
    pump = 'sweep'; probe = 'fixed'
    whichalpha='coreshell_MW'
    nTOT = 10
    power = 500 #microWatts
    for idx, rad_val in enumerate(radius):
        # Set pump and probe wavelengths 
        if pump == 'fixed': 
            wave_pump = np.array([532.E-7])
        if pump == 'sweep': 
            wave_pump = waverange
        if pump == 'track_res': 
            mt = Mie_Theory(rad_val, waverange)
            w_abs_res, _, _, _ = mt.find_resonance(nTOT=nTOT)
            wave_pump = np.array([w_abs_res])

        if probe == 'fixed':
            wave_probe = 1000.E-7
        if probe == 'sweep':
            wave_probe = waverange
        if probe == 'track_res': 
            mt = Mie_Theory(rad_val, waverange)
            _, _, w_sca_res, _ = mt.find_resonance()
            wave_probe = w_sca_res
        mt_single = Mie_Theory(rad_val, wave_pump)
        abs_cross, _, _ = mt_single.cross_sects(nTOT=nTOT)
        ############################################################
        pt = Photothermal_Image(rad_val, np.round(wave_pump,7), abs_cross, 
                                power*10, np.round(wave_probe,7), whichalpha)
        PI_sin, SI_sin, _ = pt.pt_signal(which='sin')
        PI_cos, SI_cos, _ = pt.pt_signal(which='cos')
        total_signal = np.sqrt( (PI_sin + SI_sin)**2 + (PI_cos + SI_cos)**2)
        just_SI = np.sqrt( (SI_sin)**2 + (SI_cos)**2)
        just_PI = np.sqrt( (PI_sin)**2 + (PI_cos)**2)
        ############################################################
        #### Mie theory ####
        mt = Mie_Theory(rad_val, waverange)
        abs_cross_MIE, _, _ = mt.cross_sects(nTOT=nTOT)
        _, sca_cross_MIE, _ = mt.cross_sects(nTOT=nTOT)
        w_abs_res, abs_cross, w_sca_res, sca_cross = mt.find_resonance(nTOT=nTOT)
        ax[idx].plot(np.array([w_abs_res, w_abs_res])*1E7, 
            np.array([-1, 1]), color='k', linewidth=0.75, linestyle='dashed')
        ax[idx].plot(np.array([w_sca_res, w_sca_res])*1E7, 
            np.array([-1, 1]), color='k', linewidth=0.75, linestyle='dashed')
        # Shade regions 
        ax[idx].fill_between(waverange*1E7, sca_cross_MIE, 
                         color='mediumpurple',zorder=0, alpha=.6)
        ax[idx].fill_between(waverange*1E7, abs_cross_MIE, 
                         color='gray',zorder=1, alpha=.6)
        if max(sca_cross_MIE) > max(abs_cross_MIE):
            whichbig = sca_cross_MIE
        if max(sca_cross_MIE) < max(abs_cross_MIE):
            whichbig = abs_cross_MIE
        ## Format Figure 
        ax[idx].set_ylim([-max(whichbig)*1.1, max(whichbig)*1.1])
        ax[idx].set_xlim(400, 700)
        ax[idx].set_title(str('r = ')+str(int(np.round(rad_val*1E7)))+str(' nm'), pad=20)
        ax[idx].set_yticks([-max(whichbig), 0, max(whichbig)])
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0,0))
        ax[idx].yaxis.set_major_formatter(yfmt)

        #### Plot Photothermal ####
        ax2 = ax[idx].twinx()
        ax2.plot(waverange*1E7, total_signal,
                'k',
                zorder=2,
                linewidth=1.5,
                )
        ax2.plot(waverange*1E7, PI_sin+PI_cos, 
                'tab:blue',
                linewidth=1.5,
                 #alpha=0.7,
                zorder=0)
        ax2.plot(waverange*1E7, SI_sin+SI_cos,
                'tab:red',
                linewidth=1.5,
                #alpha=0.7,
                zorder=0)
        ax2.axhline(0,0,color='k', linewidth=1)
        ax2.set_ylim([-max(total_signal)*1.6, max(total_signal)*1.6])
        ax2.set_yticks([-np.round(max(total_signal),1), 0, np.round(max(total_signal),1)])

    if pump == 'sweep' and probe != 'sweep':
        ax[0].set_xlabel('Pump Wavelength [nm]') 
        whichsweep = str('pu')
    if pump != 'sweep' and probe == 'sweep':
        ax[0].set_xlabel('Probe Wavelength [nm]') 
        whichsweep = str('pr')
    if pump == 'sweep' and probe == 'sweep':
        ax[0].set_xlabel('Pump & Probe Wavelength [nm]') 
        whichsweep = str('pupr')
    plt.subplots_adjust(left=.05, bottom=.2, right=.95, 
                        top=.8, wspace=.75) 
    plt.show()
    fig.savefig(str('parts_')+str(whichalpha)+str('_')+str('sweep_')+str(whichsweep)+str('.png'), 
        dpi=500, bbox_inches='tight'
        )


pi_si_terms()






def sweep_power():
    fig, ax = plt.subplots(1, 1, figsize=(2.8,2.8),sharex=True)
    waverange = np.round(np.arange(450, 700, 1)*1E-7, 7)
    whichalpha='coreshell_MW'
    norm='yes'
    nTOT = 10
    power_range = np.arange(0, 1001, 100) # microwatts
    radius = np.array([10, 20, 75])*1E-7
    total_signal = np.zeros((len(radius), len(power_range)))

    for idx_r, val_rad in enumerate(radius):

        # mt = Mie_Theory(val_rad, waverange)
        # wave_res, _, _, _ = mt.find_resonance(nTOT=nTOT)
        # wave_pump = np.array([wave_res])

        wave_pump = np.array([532.E-7])
        mt_single = Mie_Theory(val_rad, wave_pump)
        abs_cross, _, _ = mt_single.cross_sects(nTOT=nTOT)


        for idx_p, val_pow in enumerate(power_range):
            pt = Photothermal_Image(val_rad, wave_pump, np.array([abs_cross]),
                                    val_pow*10, 785.E-7, whichalpha)

            PI_sin, SI_sin, _ = pt.pt_signal(which='sin')
            PI_cos, SI_cos, _ = pt.pt_signal(which='cos')
            total_signal[idx_r, idx_p] = np.sqrt( (PI_sin + SI_sin)**2 + (PI_cos + SI_cos)**2)

    for idx_r, val_rad in enumerate(radius):
        if norm == 'yes':
            plt.plot(power_range, total_signal[idx_r,:]\
                /max(total_signal[idx_r,:]),
                label=str('r = ')+str(int(val_rad*1E7))+str(' nm'))
            plt.ylabel('Normalized PT Signal')

        if norm == 'no':
            plt.plot(power_range, total_signal[idx_r,:]\
                ,label=str('r = ')+str(int(val_rad*1E7))+str(' nm'))
            plt.ylabel('PT Signal')

    plt.xlabel('Power [$\mu$W]')
    plt.xlim([0, 1000])
    # plt.ylim([0, 1.1])
    plt.legend(frameon=False)
    plt.subplots_adjust(left=.2, bottom=.2)
    fig.savefig('power_zR.png',
        dpi=500, bbox_inches='tight'
        )

    plt.show()

# sweep_power()


def sweep_zp():
    radius = np.array([10, 20, 30, 40, 75])*1E-7
    fig, ax = plt.subplots(2, 5, figsize=(10,3.5),sharex=True)
    waverange = np.round(np.arange(450, 700, 1)*1E-7, 7)
    whichalpha='coreshell_MW'
    nTOT = 10
    val_pow = 1000 # microwatts
    zp = np.arange(-2000, 2000)*1E-7

    for idx, val_rad in enumerate(radius):
        wave_pump = np.array([532.E-7])
        mt_single = Mie_Theory(val_rad, wave_pump)
        abs_cross, _, _ = mt_single.cross_sects(nTOT=10)
        pt = Photothermal_Image(val_rad, wave_pump, np.array([abs_cross]),
                                val_pow*10, 785.E-7, whichalpha, zp)
 
        PI_sin, SI_sin, zR = pt.pt_signal(which='sin')
        PI_cos, SI_cos, _ = pt.pt_signal(which='cos')
        sinterm = PI_sin[0,:] + SI_sin[0,:]
        costerm = PI_cos[0,:] + SI_cos[0,:]
        self_int = SI_sin[0,:]+SI_cos[0,:]
        probe_int = PI_sin[0,:]+PI_cos[0,:]
        total_signal = np.sqrt( sinterm**2 + costerm**3 )
    
        min_idx_sig = np.where(total_signal == min(total_signal))
        max_idx_si = np.where(self_int == max(self_int))

        ax[0, idx].plot(zp*1E7, total_signal,
                'k', linewidth=1.5)
        ax[0, idx].plot(zp*1E7, PI_sin[0,:]+PI_cos[0,:],
                'tab:blue',
                linewidth=1.5,)
        ax[0, idx].plot(zp*1E7, self_int,
                'tab:red',
                linewidth=1.5,)


        ax[0, idx].axhline(0,color='k',linewidth=1)
        ax[0, idx].axvline(0,color='k',linewidth=1)

        ax[0, idx].set_title(str('r = ')+str(int(np.round(val_rad*1E7)))+str(' nm'))

        ax[1, idx].plot(zp*1E7, total_signal,
                'k', linewidth=1.5)
        ax[1, idx].plot(zp*1E7, np.arctan2(sinterm, costerm),
                'tab:purple')
        ax[1, idx].axhline(0,color='k',linewidth=1)
        ax[1, idx].axvline(0,color='k',linewidth=1)

        ax[1, idx].set_xlabel('$z_p$ [nm]')
        # Draw zR
        ax[0, idx].axvline(-zR*1E7, color='magenta', linewidth=1)
        ax[0, idx].axvline(zR*1E7, color='lime',linewidth=1)
        ax[1, idx].axvline(-zR*1E7, color='magenta',linewidth=1)
        ax[1, idx].axvline(zR*1E7, color='lime',linewidth=1)

    plt.xlim([min(zp)*1E7, max(zp)*1E7])
    plt.xticks([-1500, 0, 1500])
    plt.subplots_adjust(left=.05, bottom=.2, right=.95, 
                    top=.8, wspace=.6,hspace=.2) 
    fig.savefig(str('zoffset')+str(whichalpha)+str('_')+str('.png'), 
    dpi=500, bbox_inches='tight'
    )

    plt.show()
# sweep_zp()






def sweep_power_zpdiff():
    fig, ax = plt.subplots(1, 1, figsize=(2.8,2.8),sharex=True)
    waverange = np.round(np.arange(450, 700, 1)*1E-7, 7)
    whichalpha='coreshell_MW'
    nTOT = 10
    power_range = np.arange(0, 1001, 100) # microwatts
    radius = 75.*1E-7
    total_signal = np.zeros((3, len(power_range)))

    wave_pump = np.array([532.E-7])
    mt_single = Mie_Theory(radius, wave_pump)
    abs_cross, _, _ = mt_single.cross_sects(nTOT=nTOT)

    pt = Photothermal_Image(radius, wave_pump, np.array([abs_cross]),
                            200*10, 785.E-7, whichalpha)
    _, _, zR = pt.pt_signal(which='sin')

    colors = ['magenta', 'black', 'lime']
    legendlabels = ['-zR', '0', 'zR']
    zR_range = np.array([-zR, 0, zR])
    for idx_zR, val_zR in enumerate(zR_range):
        for idx_p, val_pow in enumerate(power_range):
            pt = Photothermal_Image(radius, wave_pump, np.array([abs_cross]),
                                    val_pow*10, 785.E-7, whichalpha, np.array([val_zR]))
            PI_sin, SI_sin, _ = pt.pt_signal(which='sin')
            PI_cos, SI_cos, _ = pt.pt_signal(which='cos')
            total_signal[idx_zR, idx_p] = np.sqrt( (PI_sin + SI_sin)**2 + (PI_cos + SI_cos)**2)
        plt.plot(power_range, total_signal[idx_zR,:], color=colors[idx_zR], label=legendlabels[idx_zR])

    plt.axvline(200)

    plt.ylabel('PT Signal')

    plt.xlabel('Power [$\mu$W]')
    plt.xlim([0, 1000])
    # plt.ylim([0, 1.1])
    plt.legend(frameon=False)
    plt.subplots_adjust(left=.2, bottom=.2)
    fig.savefig('power_zR_diff.png',
        dpi=500, bbox_inches='tight'
        )

    plt.show()

# sweep_power_zpdiff()
