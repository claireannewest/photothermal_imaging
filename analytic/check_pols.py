import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

nb=1.473

def eps(wavelength):
	JCdata = np.loadtxt('auJC.tab',skiprows=3)
	wave = JCdata[:,0]*1E-4 # cm
	n_re_raw = JCdata[:,1]
	n_im_raw = JCdata[:,2]
	wavenew = np.arange(np.round(min(wave),7)+1E-7, np.round(max(wave),7), 1E-7)
	f_nre= interp1d(wave, n_re_raw, "cubic")
	n_re = f_nre(wavenew)
	f_nim= interp1d(wave, n_im_raw, "cubic")
	n_im = f_nim(wavenew)
	idx = np.where(np.round(wavenew,7) == np.round(wavelength,7))[0][0]
	n = n_re[idx] + 1j*n_im[idx]
	eps = n_re[idx]**2 - n_im[idx]**2 +1j*(2*n_re[idx]*n_im[idx])
	return eps

def alpha(wavelength, r):
	xi = 2*np.pi*r/wavelength
	q = 1/3*(1-(xi)**2 - 1j*2/3*(xi)**3)
	eps1 = eps(wavelength)
	epsb = nb**2
	return epsb/3*r**3*(eps(wavelength)-nb**2)/( nb**2 +q*(eps(wavelength) - nb**2) )

def pi(wavelength):
	eps1 = eps(wavelength)
	epsb = nb
	return np.imag(1/(eps1+2*epsb)**2)

def term1(wavelength):
	eps1 = eps(wavelength)
	epsb = nb
	return np.real(1/np.conj(eps1+2*epsb) * 1/(eps1+2*epsb)**2)

def term2(wavelength):
	eps1 = eps(wavelength)
	epsb = nb
	return np.real(1/np.abs((eps1+2*epsb)**2)**2)

def scat(wavelength):
	eps1 = eps(wavelength)
	epsb = nb
	return np.real(1/np.abs(eps1+2*epsb))**2

def ext(wavelength):
	eps1 = eps(wavelength)
	epsb = nb
	return np.imag(-1/(eps1+2*epsb))

def compare_them_all():
	waverange = np.arange(400,700)*1E-7
	termalpha_1 = np.zeros(len(waverange))
	termalpha_2 = np.zeros(len(waverange))
	pi_term = np.zeros(len(waverange))
	sca = np.zeros(len(waverange))
	ext_c = np.zeros(len(waverange))


	for idx, wave in enumerate(waverange):
		termalpha_1[idx] = term1(wavelength=wave)
		termalpha_2[idx] = term2(wavelength=wave)
		pi_term[idx] = pi(wavelength=wave)
		sca[idx] = scat(wavelength=wave)
		ext_c[idx] = ext(wavelength=wave)


	k = 2*np.pi*nb/waverange
	fig, ax = plt.subplots(1, 1, figsize=(4,4),sharex=True)
	plt.plot(waverange*1E7, termalpha_1/max(termalpha_1),'tomato', label=str('SI 1'))
	plt.plot(waverange*1E7, termalpha_2/max(termalpha_2), 'magenta',label=str('SI 2'))
	plt.plot(waverange*1E7, pi_term/max(pi_term), 'tab:blue',label=str('PI'))
	plt.plot(waverange*1E7, sca/max(sca), 'k',linestyle='dotted', label=str('Sca'))
	plt.plot(waverange*1E7, ext_c/max(ext_c), 'k',linestyle='dashed', label=str('Ext'))


	# plt.plot(waverange*1E7, termalpha_1,'tomato', label=str('term1'))
	# plt.plot(waverange*1E7, termalpha_2, 'magenta',label=str('term2'))
	# plt.plot(waverange*1E7, pi_term, 'tab:blue',label=str('pi'))


	plt.axhline(0, color='k')


	idx_1_min = np.where(termalpha_1 == min(termalpha_1))
	idx_1_max = np.where(termalpha_1 == max(termalpha_1))
	idx_2_min = np.where(termalpha_2 == min(termalpha_2))
	idx_2_max = np.where(termalpha_2 == max(termalpha_2))
	idx_p_min = np.where(pi_term == min(pi_term))
	idx_p_max = np.where(pi_term == max(pi_term))
	idx_ext_max = np.where(ext_c == max(ext_c))
	idx_sca_max = np.where(sca == max(sca))


	print('Min SI1', int(waverange[idx_1_min][0]*1E7))
	print('Max SI1', int(waverange[idx_1_max][0]*1E7))

	print('Min SI2', int(waverange[idx_2_min][0]*1E7))
	print('Max SI2', int(waverange[idx_2_max][0]*1E7))

	print('Min PI', int(waverange[idx_p_min][0]*1E7))
	print('Max PI', int(waverange[idx_p_max][0]*1E7))

	print('Max Sca', int(waverange[idx_sca_max][0]*1E7))
	print('Max Ext', int(waverange[idx_ext_max][0]*1E7))


	plt.xlim([400, 700])
	plt.ylim([-9, 2])
	plt.xlabel('Probe Wavelength [nm]')
	plt.legend(frameon=False)
	plt.subplots_adjust(bottom=.2)
	plt.show()

	fig.savefig('check_pols.png',
	    dpi=500, bbox_inches='tight'
	    )

#################################

def vary_pump():
	waverange = np.arange(400,1000)*1E-7
	extinc = np.zeros(len(waverange))
	linear = np.zeros(len(waverange))
	quad = np.zeros(len(waverange))
	P=1
	r=100.E-7
	for idx, wave in enumerate(waverange):
		waist = wave * 0.6 / 1.25 
		A1 = 1E-4*P/waist**2
		A2 = 5E-4*P/waist**2
		A3 = (1E-4*P/waist**2)**2

		k = 2*np.pi/wave
		extinc[idx] = 4*np.pi*k*np.imag(alpha(wavelength=wave,r=r))
		scat = 8*np.pi/3*k**4*np.abs(alpha(wavelength=wave,r=r))**2

		linear[idx] = A1*extinc[idx]
		quad[idx] = A3*extinc[idx]**2


	fig, ax = plt.subplots(1, 1, figsize=(4,4),sharex=True)
	plt.plot(waverange*1E7, extinc, label='abs')
	# plt.plot(waverange*1E7, linear/max(linear), label='linear')
	# plt.plot(waverange*1E7, quad/max(quad), label='quad',linestyle='--')
	# tot = linear+quad 
	# plt.plot(waverange*1E7, tot/max(tot), label='Tot', linestyle='--')

	# plt.axhline(0, color='k')
	# idx_p_min = np.where(pi_term == min(pi_term))
	# idx_p_max = np.where(pi_term == max(pi_term))
	# idx_ext_max = np.where(ext_c == max(ext_c))

	# print('Min PI', int(waverange[idx_p_min][0]*1E7))
	# print('Max PI', int(waverange[idx_p_max][0]*1E7))
	# print('Max Ext', int(waverange[idx_ext_max][0]*1E7))

	# plt.xlim([400, 700])
	# # plt.ylim([-9, 2])
	# plt.xlabel('Probe Wavelength [nm]')
	plt.legend(frameon=False)
	# plt.subplots_adjust(bottom=.2)
	plt.show()

	# fig.savefig('check_pols.png',
	#     dpi=500, bbox_inches='tight'
	#     )


vary_pump()