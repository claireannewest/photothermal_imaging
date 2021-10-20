import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

JCdata = np.loadtxt('auJC.tab',skiprows=3)
wave = JCdata[:,0]*1E-4 # cm
n_re = JCdata[:,1]
n_im = JCdata[:,2]
plt.plot(wave*1E7, n_re)
plt.plot(wave*1E7, n_im)



JCdata = np.loadtxt('agDrudeFit_101421.tab',skiprows=3)
wave = JCdata[:,0]*1E-4 # cm
eps_re = JCdata[:,1]
eps_im = JCdata[:,2]

eps_c = eps_re + 1j*eps_im
mod_eps_c = np.sqrt(eps_re**2 + eps_im**2)
n_re = np.sqrt( (mod_eps_c + eps_re )/ 2)
n_im = np.sqrt( (mod_eps_c - eps_re )/ 2)


plt.plot(wave*1E7, n_re,'--')
plt.plot(wave*1E7, n_im,'--')




plt.legend()
plt.show()



wavenew = np.arange(np.round(min(wave),7)+1E-7, np.round(max(wave),7), 1E-7)
f_nre= interp1d(wave, n_re_raw, "cubic")
n_re = f_nre(wavenew)
f_nim= interp1d(wave, n_im_raw, "cubic")
n_im = f_nim(wavenew)

# file = open(str('auJC_interp.tab'),'w')
# file.write( str('Refractive index of gold (Made on 10-13-21)')+ '\n')
# file.write( str('1 2 3 0 0 = specifies whether N or epsilon are read in (see manual)') + '\n')
# file.write( str('lambda [um]') + '\t' + str('n') + '\t' + str('k') + '\n')
# for j in range(0, len(wavenew)):
#   file.write( "%.3f" % float(wavenew[j]*1E4) + '\t' + "%.5f" % n_re[j] + '\t' + "%.5f" % n_im[j] + '\n')
# file.close()


# plt.plot(wavenew*1E7, n_re,alpha=.4)
# plt.plot(wavenew*1E7, n_im,alpha=.4)

# ng = n_re + 1j*n_im
# nb=1.#473

# oneterm = np.imag(-np.conj(1/(ng**2+2*nb**2)**2))
# plt.plot(wavenew*1E7, oneterm/max(oneterm),label='im(-conj(1/denom))')
# idx = np.where(oneterm == min(oneterm))
# print(wavenew[idx]*1E7)

# oneterm = np.imag((1/(ng**2+2*nb**2)**2))
# plt.plot(wavenew*1E7, oneterm/max(oneterm),'--',label='absorption')
# idx = np.where(oneterm == max(oneterm))
# print(wavenew[idx]*1E7)


