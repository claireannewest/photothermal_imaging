import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from scipy.interpolate import interp1d

e = 4.80326E-10 # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10 # speed of light [cm/s]
hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]

class Mie_Theory:
    def __init__(self, radius, n, wave_pump):
        """Defines the different system parameters.
        
        Keyword arguments:
        radius -- [cm] radius of NP 
        n -- [unitless] refractive index of background 
        wave -- [cm] wavelength of driving laser 
        """
        self.radius = radius
        self.n = n
        self.wave_pump = wave_pump
        self.k = 2*np.pi*self.n/self.wave_pump

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
        idx = np.where(np.round(wavenew,7) == np.round(self.wave_pump,7))[0][0]
        m = (n_re[idx] + 1j*n_im[idx])/self.n
        x = self.k*self.radius
        numer_a = m*self.psi(n,m*x)*self.psi_prime(n,x) - self.psi(n,x)*self.psi_prime(n,m*x)
        denom_a = m*self.psi(n,m*x)*self.xi_prime(n,x) - self.xi(n,x)*self.psi_prime(n,m*x)
        numer_b = self.psi(n,m*x)*self.psi_prime(n,x) - m*self.psi(n,x)*self.psi_prime(n,m*x)
        denom_b = self.psi(n,m*x)*self.xi_prime(n,x) - m*self.xi(n,x)*self.psi_prime(n,m*x)
        an = numer_a/denom_a
        bn = numer_b/denom_b
        return an, bn 

    def cross_sects(self, nTOT):
        a_n = np.zeros(nTOT,dtype=complex)
        b_n = np.zeros(nTOT,dtype=complex)
        ni = np.arange(1,nTOT+1)
        for i in range(nTOT):
            a_n[i], b_n[i] = self.mie_coefficent(n=(i+1))
        C_ext = 2 * np.pi/(self.k**2)*np.sum((2*ni+1)*np.real(a_n+b_n))
        C_sca = 2 * np.pi/(self.k**2)*np.sum((2*ni+1)*(np.real(a_n*np.conj(a_n)+b_n*np.conj(b_n))))
       	C_abs = C_ext - C_sca
        return C_abs, C_sca

##############################################################################################################################

mt = Mie_Theory(5E-7, 1.473, 530E-7)
print(mt.cross_sects(10))


## System Paramters ###
a = np.arange(5,86,5)*1E-7 # cm, radius of sphere

n_R = 1.473 # room temperature background 
dndT = -10**(-4) # 1/K
kappa = 0.6*(1E7/100) # erg/ (s cm K)
C = (1.26*2.35*1E7) # erg / (cm^3 K)
abs_cross = 1.16*1E-14*(100**2) #cm^2
alpha0 = 0

### Pump Laser Parameters ###
wave_pump = np.array([531.0, 533.0, 534.0, 537.0, 542.0, 549.0, 555.0, 561.0, 561.0, 530.0, 527.0, 528.0, 531.0, 536.0, 540.0, 545.0, 548.0 ])
abs_cross = np.array([8e-05, 0.00063, 0.00217, 0.00503, 0.00904, 0.01327, 0.01647,  0.01777, 0.01739, 0.01769, 0.02012, 0.02336, 0.02723, 0.03135, 0.03514, 0.03805, 0.0398])

# P0h = 225*1E-6*1E7 # erg/s
P0h = np.arange(200, 1200, 10)*1E-6*1E7 
Omega = 1E6 # 1/s (1 MHz)

### Probe Laser Parameters ###
wave_probe = 785E-7 # cm

### Imaging Parameters ###
# zp = np.arange(-10000,10000,.1)*1E-7 # cm
f_thph = 1; g_thph=1

### Functions ###
def waist(wave): # [cm]
        NA = 1.25
        return wave * 0.6 / NA 



def signal(soc, abs_cross, wave_pump):
    w_pump = waist(wave=wave_pump)
    w_probe = waist(wave=wave_probe)

    k = 2*np.pi*n_R/wave_probe # cm^-1
    zR = np.pi*w_probe**2*n_R/wave_probe
    rth = np.sqrt(2*kappa/(Omega*C))
    zp=zR

    first = 4*f_thph*P0h*abs_cross/(np.pi*w_pump**2*kappa*wave_probe)
    second = n_R**2*dndT*rth**2/w_probe**2
    third = (-zp/zR)/(1+zp**2/zR**2)
    if soc == 'sin': const = 0.366
    if soc == 'cos': const = 0.053
    PI = first*second*third*const
    ######################################################
    first_prefix = g_thph*k**4/zR**2*1/(1+zp**2/zR**2)
    second_prefix = dndT*n_R/(2*np.pi)*P0h*abs_cross/(np.pi*w_pump**2)*rth**2/kappa
    part1 = 2*alpha0
    part2 = second_prefix
    if soc == 'sin': const1 = 0.366; const2 = 0.0183
    if soc == 'cos': const1 = 0.053; const2 = 0.0263
    SI = first_prefix*second_prefix*(part1*const1+part2*const2)
    return PI, SI

fig, ax = plt.subplots(2, 2,sharex=True,)# sharey=True)
n = 9
colors = plt.cm.gist_rainbow(np.linspace(0,1,n))

### PI Sin ###
count=0
for i in range(1, len(a),2):
    PI_sin, SI_sin = signal(soc='sin',abs_cross=abs_cross[i], wave_pump=wave_pump[i])
    print(PI_sin)
    ax[0, 0].plot(P0h*0.1, PI_sin, label=str(round(a[i]*1E7)), color=colors[count])
    ax[0, 1].plot(P0h*0.1, SI_sin, color=colors[count])
    PI_cos, SI_cos = signal(soc='cos',abs_cross=abs_cross[i], wave_pump=wave_pump[i])
    ax[1, 0].plot(P0h*0.1, PI_cos, color=colors[count])
    ax[1, 1].plot(P0h*0.1, SI_cos, color=colors[count])

    count=count+1

ax[1, 0].set_xlabel('$P_0$ [$\mu W$]')
ax[1, 1].set_xlabel('$P_0$ [$\mu W$]')

ax[0, 0].set_title('PI Sin')
ax[1, 0].set_title('PI Cos')
ax[0, 1].set_title('SI Sin')
ax[1, 1].set_title('SI Cos')
ax[0,0].legend(loc='upper left',frameon=False)
plt.show()


