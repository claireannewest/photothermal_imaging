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
        return 2*n*nb**4*r**3 / (nb**2*(-1+q) - n**2*q)**2

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
        # print('1 ', dgold_dT_atT0)
        # print('2 ', dglyc_dT_atT0)
        return (dgold_dT_atT0*dn1_dT + dglyc_dT_atT0*dn2_dT)

    def Pabs(self):
        w_pump = self.waist(wave=self.wave_pump) # cm
        I0 = 2*self.P0h/(np.pi*w_pump**2)
        return self.abs_cross*I0

    def probe_inteference(self,which):
        w_probe = self.waist(wave=self.wave_probe) # cm
        k = self.convert_k(self.wave_probe) # cm^-1
        zR = self.zR()
        zp = self.zp()
        if which == 'sin': const = 0.366
        if which == 'cos': const = 0.053
        if self.kind == 'gold_sph_QS' or self.kind == 'gold_sph_MW':
            V = 4/3*np.pi*self.radius**3
        else:
            V = 4/3*np.pi*self.shell_radius**3
        guoy = 1/np.sqrt(1+(-zp/zR)**2) + 1j*(-zp/zR)/np.sqrt(1+(-zp/zR)**2)

        signal = 8*np.pi*self.f_thph*self.nb_T0/(self.wave_probe*w_probe**2)*\
                    1/np.sqrt(1+(zp/zR)**2)*self.Pabs()*self.rth**2/self.kappa*\
                    np.imag(-1*guoy*np.conj(self.dalphadn_atT0())*const/V\
                    )

        part = np.imag(-1*guoy*np.conj(self.dalphadn_atT0()))
        return signal

    def self_inteference(self,which):
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
        signal = 16*np.pi**2*self.g_thph*self.nb_T0**2/(self.wave_probe**2*w_probe**4)*\
                    1/(1+(zp/zR)**2)*self.Pabs()*self.rth**2/self.kappa*(\
                    2*np.real( np.conj( self.alpha_atT0() ) * self.dalphadn_atT0() )*const1/V +\
                    np.abs(self.dalphadn_atT0())**2 * self.Pabs()*self.rth**2/self.kappa * const2/V**2)
        return signal


#####################################################################################
#####################################################################################
### PT SPECTRA (SWEEP PROBE) AGAINST MIE ###
#####################################################################################
#####################################################################################

def plot_spectra_sweep(radius, ax, whichalpha, pump, probe, c):
    waverange = np.arange(450, 700, 10)*1E-7
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
                                5000*10, np.round(wave_probe,7), whichalpha)

        PI_sin[idx] = pt.probe_inteference(which='sin'); 
        PI_cos[idx] = pt.probe_inteference(which='cos')
        SI_sin[idx] = pt.self_inteference(which='sin'); 
        SI_cos[idx] = pt.self_inteference(which='cos')

        # Mie theory
        mt = Mie_Theory(radius, waverange)
        abs_cross_MIE, _, _ = mt.cross_sects(nTOT=10)
        _, sca_cross_MIE, _ = mt.cross_sects(nTOT=10)

    # ax2 = ax.twinx()
    # ax2.plot(waverange*1E7, np.sqrt( (SI_sin*0+PI_sin)**2 + (SI_cos*0+PI_cos)**2),
    #         color=c)
    # ax.plot(1,1,color=c, label=whichalpha)
    # ax.set_xlim(450, 700)
    # ax.set_title(str(int(np.round(radius*1E7)))+str(' nm'))
    # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)




    return waverange, abs_cross_MIE, sca_cross_MIE, SI_sin, PI_sin, SI_cos, PI_cos
################################################################################################
################################################################################################

def make_figure_of_plots():
    radii = np.array([50, 10])*1E-7
    fig, ax = plt.subplots(1, 1, figsize=(4,4),sharex=True,)
    ax2 = ax.twinx()
    radius = radii[0]
    waverange, abs_cross_MIE, sca_cross_MIE, SI_sin, PI_sin, SI_cos, PI_cos = plot_spectra_sweep(
                    radius=radii[0], ax=ax, c='',
                    whichalpha='glyc_sph_MW', 
                    pump='sweep', # sweep, fixed, or track_res
                    probe='sweep'
                    )

    # ax2.plot(waverange*1E7, np.sqrt( (SI_sin+PI_sin)**2 + (SI_cos+PI_cos)**2),
    #     color='tab:green', label='PT Signal')
    # ax.plot(1,1,color=c, label='PT Signal')

    ax.plot(waverange*1E7, abs_cross_MIE, color='tab:blue', label='Abs $l$=10')
    ax.plot(waverange*1E7, sca_cross_MIE, color='tab:orange',label='Sca $l$=1')


    waverange, _, _, SI_sin, PI_sin, SI_cos, PI_cos = plot_spectra_sweep(
                    radius=radii[0], ax=ax, c='',
                    whichalpha='coreshell_MW', 
                    pump='sweep', # sweep, fixed, or track_res
                    probe='sweep'
                    )

    ax2.plot(waverange*1E7, 3*(np.sqrt( (SI_sin+PI_sin)**2 + (SI_cos+PI_cos)**2)),
            color='tab:purple', linestyle='--', label='PT Signal')
    print(min(3*(np.sqrt( (SI_sin+PI_sin)**2 + (SI_cos+PI_cos)**2))))
    # ax.plot(1,1,color=c, linestyle='--',label='PT Signal')

    ax.set_xlim(450, 700)
    ax.set_title(str(int(np.round(radius*1E7)))+str(' nm'))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)






    ax.legend(frameon=False)
    plt.subplots_adjust(left=.2,hspace=.3, top=.965, bottom=.07,right=.9,wspace=.2)
    plt.show()

make_figure_of_plots() 


# def make_figure_of_plots():
#     radii = np.array([5, 10])*1E-7
#     fig, ax = plt.subplots(int(len(radii)), 1, figsize=(4,7),sharex=True,)
#     for idx, radii in enumerate(radii):
#         print(int(radii*1E7))
#         c = plot_spectra_sweep(radius=radii, ax=ax[idx], 
#                             whichalpha='glyc_sph_QS', 
#                             pump='sweep', # sweep, fixed, or track_res
#                             probe='fixed'
#                             )


#         break
#     ax[0].legend(frameon=False)

#     if c == 'tab:purple':
#         ax[-1].set_xlabel('Pump Wavelength [nm]',fontsize=12)
#     if c == 'tab:red':
#         ax[-1].set_xlabel('Probe Wavelength [nm]',fontsize=12)

#     plt.subplots_adjust(left=.2,hspace=.3, top=.965, bottom=.07,right=.9,wspace=.2)
#     plt.show()

# make_figure_of_plots() 
