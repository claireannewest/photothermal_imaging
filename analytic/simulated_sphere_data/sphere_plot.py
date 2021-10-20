import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "sans-serif",
#   "font.sans-serif": ["Helvetica"]})

radii = np.arange(5,101,5)
num_wave = 200
########################################################################################
### MIE ###
########################################################################################
mie_l10_JCvac_wave = np.zeros((len(radii),num_wave))
mie_l10_JCvac_ext = np.zeros((len(radii),num_wave))
mie_l10_JCvac_sca = np.zeros((len(radii),num_wave))
for count, val in enumerate(radii):
	datai = loadmat(str('sphere_analysis/Sph')+str(val)+str('nm_JC_ret_l10_n1.0.mat'))
	mie_l10_JCvac_wave[count,:] = datai['enei'][0]
	mie_l10_JCvac_ext[count,:] = datai['ext'][0]
	mie_l10_JCvac_sca[count,:] = datai['sca'][0]

########################################################################################
mie_l10_JCgly_wave = np.zeros((len(radii),num_wave))
mie_l10_JCgly_ext = np.zeros((len(radii),num_wave))
mie_l10_JCgly_sca = np.zeros((len(radii),num_wave))
for count, val in enumerate(radii):
	datai = loadmat(str('sphere_analysis/Sph')+str(val)+str('nm_JC_ret_l10_n1.473.mat'))
	mie_l10_JCgly_wave[count,:] = datai['enei'][0]
	mie_l10_JCgly_ext[count,:] = datai['ext'][0]
	mie_l10_JCgly_sca[count,:] = datai['sca'][0]

########################################################################################
def find_resonance():
	n = len(radii)
	colors = plt.cm.gist_rainbow(np.linspace(0,1,n))

	mie_l10_JCgly_abs = mie_l10_JCgly_ext - mie_l10_JCgly_sca
	for i, val in enumerate(radii):
		plt.plot(mie_l10_JCgly_wave[i,:], mie_l10_JCgly_abs[i,:], color=colors[i])
		idx = np.where(mie_l10_JCgly_abs[i,:] == max(mie_l10_JCgly_abs[i,:]))
		print(val, np.round(mie_l10_JCgly_wave[i, idx][0][0]), np.round(mie_l10_JCgly_abs[i, idx][0][0], 5))

	plt.show()
find_resonance()
########################################################################################
### BEM ###
########################################################################################
BEM_JCvac_wave = np.zeros((len(radii),num_wave))
BEM_JCvac_ext = np.zeros((len(radii),num_wave))
BEM_JCvac_sca = np.zeros((len(radii),num_wave))
for count, val in enumerate(radii):
	datai = loadmat(str('sphere_analysis/Sph')+str(val)+str('nm_JC_ret_BEM_n1.0.mat'))
	BEM_JCvac_wave[count,:] = datai['enei'][0]
	BEM_JCvac_ext[count,:] = datai['ext'][0]
	BEM_JCvac_sca[count,:] = datai['sca'][0]

BEM_JCgly_wave = np.zeros((len(radii),num_wave))
BEM_JCgly_ext = np.zeros((len(radii),num_wave))
BEM_JCgly_sca = np.zeros((len(radii),num_wave))
for count, val in enumerate(radii):
	datai = loadmat(str('sphere_analysis/Sph')+str(val)+str('nm_JC_ret_BEM_n1.4.mat'))
	BEM_JCgly_wave[count,:] = datai['enei'][0]
	BEM_JCgly_ext[count,:] = datai['ext'][0]
	BEM_JCgly_sca[count,:] = datai['sca'][0]


########################################################################################
### DDA ###
########################################################################################
num_wave=100
dda_JCvac_wave = np.zeros((len(radii),num_wave))
dda_JCvac_ext = np.zeros((len(radii),num_wave))
dda_JCvac_sca = np.zeros((len(radii),num_wave))
val = 5
count=0
datai = np.loadtxt(str('sphere_analysis/dda/Sph')+str(val)+str('nm_JC_DDA_1.0_0.5'))
dda_JCvac_wave[count,:] = datai[:,1]*1E3
dda_JCvac_ext[count,:] = datai[:,2]*np.pi*datai[:,0]**2
dda_JCvac_sca[count,:] = datai[:,4]*np.pi*datai[:,0]**2

########################################################################################
########################################################################################




# plt.plot(dda_JCvac_wave[0,:], dda_JCvac_ext[0,:]-dda_JCvac_sca[0,:],label='dda')

# datai = np.loadtxt(str('sphere_analysis/dda/Sph')+str(val)+str('nm_JC_DDA_1.0'))
# dda_JCvac_wave[count,:] = datai[:,1]*1E3
# dda_JCvac_ext[count,:] = datai[:,2]*np.pi*datai[:,0]**2
# dda_JCvac_sca[count,:] = datai[:,4]*np.pi*datai[:,0]**2
# plt.plot(dda_JCvac_wave[0,:], dda_JCvac_ext[0,:]-dda_JCvac_sca[0,:],label='dda 1.0')

# plt.plot(mie_l10_JCvac_wave[0,:],mie_l10_JCvac_ext[0,:]-mie_l10_JCvac_sca[0,:],label='mie')
# plt.plot(BEM_JCvac_wave[0,:],BEM_JCvac_ext[0,:]-BEM_JCvac_sca[0,:],label='bem')

# plt.legend()
# plt.show()




########################################################################################


# fig, ax = plt.subplots(3, 4,sharex=True, sharey=True)
# ax[0, 0].set_ylabel('BEM Ret.')
# ax[1, 0].set_ylabel('DDA')
# ax[2, 0].set_ylabel('Mie (l=10)')
# ax[0,0].set_title(r'$\sigma_{abs}, n=1$')
# ax[0,1].set_title(r'$\sigma_{sca}, n=1$')
# ax[0,2].set_title(r'$\sigma_{abs}, n=1.47$')
# ax[0,3].set_title(r'$\sigma_{sca}, n=1.47$')

# n = len(radii)
# colors = plt.cm.gist_rainbow(np.linspace(0,1,n))

# for i in range(0, len(radii)):
# 	ax[2,0].plot(mie_l10_JCvac_wave[i,:],mie_l10_JCvac_ext[i,:]-mie_l10_JCvac_sca[i,:], color=colors[i])

# for i in range(0, len(radii)):
# 	ax[2,1].plot(mie_l10_JCvac_wave[i,:],mie_l10_JCvac_sca[i,:],color=colors[i])


# for i in range(0, len(radii)):
# 	ax[2,2].plot(mie_l10_JCgly_wave[i,:],mie_l10_JCgly_ext[i,:]-mie_l10_JCgly_sca[i,:], color=colors[i])

# for i in range(0, len(radii)):
# 	ax[2,3].plot(mie_l10_JCgly_wave[i,:],mie_l10_JCgly_sca[i,:],color=colors[i])






# ax[0,0].set_xlim([400,700])
# ax[0,0].set_ylim([0,0.15])


# fig, ax = plt.subplots(1, 2,sharex=True, sharey=True)
# ax[0, 0].set_ylabel('BEM Ret.')
# ax[1, 0].set_ylabel('DDA')
# ax[2, 0].set_ylabel('Mie (l=10)')

# n = len(radii)
# colors = plt.cm.jet(np.linspace(0,1,n))

# fig, ax = plt.subplots(8,1,figsize=(3,8), sharex=True, sharey=True)
# count=0	

# for i in range(1, len(radii)-3,2):
# 	ax[count].plot(mie_l10_JCgly_wave[i,:],mie_l10_JCgly_ext[i,:]-mie_l10_JCgly_sca[i,:],
# 		label=radii[i], color=colors[i])
# 	# ax[count].plot(mie_l10_JCvac_wave[i,:],mie_l10_JCvac_sca[i,:],
# 		# linestyle='dashdot',color=colors[i])
# 	ax[count].set_title(str(radii[i]))
# 	count=count+1

# # for i in range(0, len(radii)):
# # 	ax[2,2].plot(mie_l10_JCgly_wave[i,:],mie_l10_JCgly_ext[i,:]-mie_l10_JCgly_sca[i,:], color=colors[i])

# # for i in range(0, len(radii)):
# # 	ax[2,3].plot(mie_l10_JCgly_wave[i,:],mie_l10_JCgly_sca[i,:],color=colors[i])


# # plt.legend()
# # plt.xlim([400,700])
# plt.subplots_adjust(top=.99, bottom=.01)
# plt.show()