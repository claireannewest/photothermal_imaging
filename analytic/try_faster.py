import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from scipy.interpolate import interp1d
from matplotlib.ticker import FormatStrFormatter

e = 4.80326E-10 # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10 # speed of light [cm/s]
hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]

class Mie_Theory:
    def __init__(self, radius, n, wave_pump=np.arange(300,1000)*1E-7):
        """Defines the different system parameters.
        
        Keyword arguments:
        radius -- [cm] radius of NP 
        n -- [unitless] refractive index of background 
        wave_pump -- [cm] wavelength of driving laser 
        """
        self.radius = radius
        self.n = n
        self.wave_pump = wave_pump

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
        JCdata = np.loadtxt('auJC.tab',skiprows=3)
        wave = JCdata[:,0]*1E-4 # cm
        n_re_raw = JCdata[:,1]
        n_im_raw = JCdata[:,2]
        wavenew = np.arange(np.round(min(wave),7)+1E-7, np.round(max(wave),7), 1E-7)
        f_nre= interp1d(wave, n_re_raw, "cubic")
        n_re = f_nre(wavenew)
        f_nim= interp1d(wave, n_im_raw, "cubic")
        n_im = f_nim(wavenew)
        m = (n_re + 1j*n_im)/self.n
        k = 2*np.pi*self.n/wavenew
        x = k*self.radius
        numer_a = m*self.psi(n,m*x)*self.psi_prime(n,x) - self.psi(n,x)*self.psi_prime(n,m*x)
        denom_a = m*self.psi(n,m*x)*self.xi_prime(n,x) - self.xi(n,x)*self.psi_prime(n,m*x)
        numer_b = self.psi(n,m*x)*self.psi_prime(n,x) - m*self.psi(n,x)*self.psi_prime(n,m*x)
        denom_b = self.psi(n,m*x)*self.xi_prime(n,x) - m*self.xi(n,x)*self.psi_prime(n,m*x)
        an = numer_a/denom_a
        bn = numer_b/denom_b
        return an, bn, wavenew 

    def cross_sects(self, nTOT):
        _, _, wavenew = self.mie_coefficent(n=(1))
        a_n = np.zeros((nTOT,1707), dtype=complex)
        b_n = np.zeros((nTOT,1707), dtype=complex)
        ni = np.arange(1,nTOT+1)
        for i in range(nTOT):
            a_n[i,:], b_n[i,:],_ = self.mie_coefficent(n=(i+1))
            ext_insum = (2*ni[i]+1)*np.real(a_n+b_n)
            sca_insum = (2*ni[i]+1)*(np.real(a_n*np.conj(a_n)+b_n*np.conj(b_n)))
        k = 2*np.pi*self.n/wavenew
        C_ext = 2 * np.pi/(k**2)*np.sum(ext_insum, axis=0)
        C_sca = 2 * np.pi/(k**2)*np.sum(sca_insum, axis=0)
       	C_abs = C_ext - C_sca
        return C_abs, C_sca, wavenew

    def find_dipole_res(self):
        C_abs_1, C_sca_1,_ = self.cross_sects(nTOT=1)
        C_abs_10, C_sca_10,_ = self.cross_sects(nTOT=10)
        idx_abs = np.where(C_abs_1 == max(C_abs_1))
        idx_sca = np.where(C_sca_1 == max(C_sca_1))
        return self.wave_pump[idx_abs][0], C_abs_10[idx_abs][0], self.wave_pump[idx_sca][0], C_sca_10[idx_sca][0]


##############################################################################################################################
##############################################################################################################################

def plot_spectra(radius, ax):
    wave = np.arange(400, 901, 10)*1E-7
    abs_cross_1 = np.zeros(len(wave))
    sca_cross_1 = np.zeros(len(wave))
    abs_cross_10 = np.zeros(len(wave))
    sca_cross_10 = np.zeros(len(wave))
    PI_sin_cs = np.zeros(len(wave))
    PI_cos_cs = np.zeros(len(wave))
    SI_sin_cs = np.zeros(len(wave))
    SI_cos_cs = np.zeros(len(wave))


    mt = Mie_Theory(radius, 1.473)
    mt.find_dipole_res()

    abs_cross_1, sca_cross_1, wave = mt.cross_sects(nTOT=1)
    abs_cross_10, sca_cross_10,_ = mt.cross_sects(nTOT=10)

    print(abs_cross_10.shape)
    print(wave.shape)

    plt.plot(wave*1E7, abs_cross_10*1E8, 
                color='tab:blue', 
                label='abs 10')
    plt.plot(wave*1E7, sca_cross_10*1E8, 
                color='tab:orange',
                label='sca 10')


    # ax.scatter(w_abs_res*1E7, C_abs_res*1E8, color='k')
    # ax.scatter(w_sca_res*1E7, C_sca_res*1E8, color='k')

    # ax2 = ax.twinx()


    # # ax2.plot(wave*1E7, np.sqrt( (PI_sin_cs+SI_sin_cs)**2+(PI_cos_cs+SI_cos_cs)**2)\
    # #                     /max(np.sqrt( (PI_sin_cs+SI_sin_cs)**2+(PI_cos_cs+SI_cos_cs)**2)),
    # #         color='tab:green',
    # #         # label='PI',
    # #         )



    # if C_sca_res > C_abs_res: which = C_sca_res*1E8
    # if C_sca_res < C_abs_res: which = C_abs_res*1E8

    # ax.set_xlim(400, 900)
    # ax.set_title(str(int(np.round(radius*1E7)))+str(' nm'))
    # ax.set_ylabel('$\sigma [ \mu m^2]$')
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
    plt.show()
plot_spectra(radius=100E-7, ax=0)



# def make_figure_of_plots():
#     radii = np.arange(10, 101, 10)*1E-7
#     fig, ax = plt.subplots(int(len(radii)/2), 2, figsize=(6,7),sharex=True,)# sharey=True)

#     for idx, radii in enumerate(radii):
#         if idx<4: col=0;
#         if idx>4: 
#             col=1; idx = idx-5
#         print(idx, radii, col)
#         plot_spectra(radius=radii, ax=ax[idx,col])

#     ax[-1].set_xlabel('Wavelength [nm]')
#     plt.subplots_adjust(left=.2,hspace=.3, top=.965, bottom=.07,right=.9)
#     plt.legend()
#     plt.show()


# make_figure_of_plots()

