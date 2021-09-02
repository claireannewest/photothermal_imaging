import numpy as np
import os.path

hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]
c = 2.998E+10 # speed of light [cm/s]

begin=BVAL
end=EVAL
stepsize=SSVAL

numtheta = 8
numphi = 100

lam_probe=LVAL

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def field_MM(filename,skip):
    fmlfile = np.loadtxt(filename, skiprows=skip)
    theta_fml = (fmlfile[:,0])*np.pi/180 # radians
    phi_fml = fmlfile[:,1]*np.pi/180 # radians
    weights = []
    f11 = fmlfile[:, 2] + 1j*fmlfile[:, 3]
    f21 = fmlfile[:, 4] + 1j*fmlfile[:, 5]
    f12 = 0; f22 = 0
    w = 1.240/lam_probe # probe wavelength, eV
    n = 1.473
    k = w/hbar_eVs/c
    waist = lam_probe * 10**3 * 0.6 / 1.25 
    I0_SI = 0.0002 / ( np.pi * (waist * 2 * 10**-9)**2 ) # W/m^2
    I0 = I0_SI*10**3 # g^1/2 / ( cm^1/2 s)
    E0 = np.sqrt(I0*8*np.pi/(c*n))
    magr = 1 # cm
    magk = np.linalg.norm(k)
    expon = np.exp(1j*magk*magr)
    E_theta = expon/(magk*magr) * ( f11 ) * E0
    E_phi = expon/(magk*magr) * ( f21 ) * E0
    E_r = 0
    Ex = np.zeros(len(theta_fml),dtype=complex)
    Ey = np.zeros(len(theta_fml),dtype=complex)
    Ez = np.zeros(len(theta_fml),dtype=complex)
    for i in range(0, len(theta_fml)):
        theta = theta_fml[i]
        phi = phi_fml[i]
        Ex[i] = E_r*np.cos(theta) - E_theta[i]*np.sin(theta)
        Ey[i] = E_r*np.sin(theta)*np.cos(phi) + E_theta[i]*np.cos(theta)*np.cos(phi) - E_phi[i]*np.sin(phi)
        Ez[i] = E_r*np.sin(theta)*np.sin(phi) + E_theta[i]*np.cos(theta)*np.sin(phi) + E_phi[i]*np.cos(phi)
    x = magr*np.cos(theta_fml)
    y = magr*np.sin(theta_fml)*np.cos(phi_fml)
    z = magr*np.sin(theta_fml)*np.sin(phi_fml)

    idx_thetamin = np.where(theta_fml*180/np.pi == 0)
    idx_thetamax = np.where(theta_fml*180/np.pi == 35)
    idx_phimin = np.where(phi_fml*180/np.pi == -180)
    idx_phimax = np.where(phi_fml*180/np.pi == 180)

    weights.append(idx_thetamin)
    weights.append(idx_thetamax)
    weights.append(idx_phimin)
    weights.append(idx_phimax)
    ### Convert theta_fml to theta 
    theta = np.arccos(z/magr)
    phi = np.arctan2(y, x)
    return np.array([Ex, Ey, Ez]), weights, theta, phi

def E_gaussian(filename,skip):
    # for one beam position, calculate the field on the partial hemi
    fmlfile = np.loadtxt(filename, skiprows=skip)
    theta_fml = (fmlfile[:,0])*np.pi/180 # radians
    phi_fml = fmlfile[:,1]*np.pi/180 # radians
    magr = 1 # cm
    n = 1.473
    w = 1.240/lam_probe # probe wavelength
    k = w*n/hbar_eVs/c
    waist = lam_probe * 0.6 / 1.25 * 1E-4 # [cm] beam waist radius
    ## position dependence 
    x_fml = magr*np.cos(theta_fml)
    y_fml = magr*np.sin(theta_fml)*np.cos(phi_fml)
    z_fml = magr*np.sin(theta_fml)*np.sin(phi_fml)
    x_focal = 0
    x = x_fml-x_focal # axial distance from beam's focus / waist 
    xR = np.pi*waist**2*n/(lam_probe*1E-4) # Rayleigh range, at distance xR, width of beam = np.sqrt(2) times larger than it is at beam waist
    wx = waist*np.sqrt(1+(x/xR)**2) # radius at which field amp. falls to 1/e of their axial values
    Rx = x*(1 + (xR/x)**2) # radius of curvature
    phi_gouy = np.arctan(x/xR) # Gouy phase
    r = np.sqrt(y_fml**2 + z_fml**2) # radial distance from center of beam 
    Emag = waist/wx*np.exp(-r**2/wx**2)*np.exp(-1j*(k*x+k*r**2/(2*Rx)-phi_gouy))
    return np.array([np.zeros(len(Emag)), Emag, np.zeros(len(Emag))])

def calculated_IPT():
    grid_width = int((end-begin-1)/stepsize+1)
    IPT_TOT_HMR = np.zeros((grid_width, grid_width))
    IPT_TOT_INT = np.zeros((grid_width, grid_width))
    IPT_R = np.zeros((grid_width, grid_width))
    ycoords = np.zeros((grid_width, grid_width))
    zcoords = np.zeros((grid_width, grid_width))
    ybegin = begin; zbegin=begin
    yend = end; zend=end
    for valy in range(ybegin,yend,stepsize):
        for valz in range(zbegin,zend,stepsize):
            print(valy, valz)
            base_file = str('raster_data/x0_y')+str(int(valy))+str('_z')+str(int(valz)) + \
                         str('/fml_x0y')+str(int(valy))+str('z')+str(int(valz))+str('_')
            lines_H = file_len(base_file+str('H'))
            lines_R = file_len(base_file+str('R'))
            base = 29; ndiff = lines_H-lines_R
            field_MM(filename=str(base_file+str('H')),skip=base+ndiff)
            E_H, weights, theta, phi = field_MM(filename=str(base_file+str('H')),skip=base+ndiff) 
            E_R, _, _, _ = field_MM(filename=str(base_file+str('R')),skip=base)
            E_p = E_gaussian(filename=str(base_file+str('R')),skip=base)
            hot_minus_room = np.zeros(numtheta*numphi)
            interf_term = np.zeros(numtheta*numphi)
            n = 1.473
            for i in range(0, numtheta*numphi):
                E_H_dTheta_dPhi = E_H[:,i]
                E_R_dTheta_dPhi = E_R[:,i]
                E_p_dTheta_dPhi = E_p[:,i]
                hot_minus_room[i] = 1/4*( np.linalg.norm(E_H[:,i])**2 - np.linalg.norm(E_R[:,i])**2 )*np.sin(theta[i])
                interf_term[i] = 1/2*np.real( np.dot(E_p[:,i] , np.conj(E_H[:,i] - E_R[:,i])) )*np.sin(theta[i])
                if i in weights[0][0]:
                    hot_minus_room[i] = hot_minus_room[i]*0.5
                    interf_term[i] = interf_term[i]*0.5
                if i in weights[1][0]:
                    hot_minus_room[i] = hot_minus_room[i]*0.5
                    interf_term[i] = interf_term[i]*0.5
                if i in weights[2][0]:
                    hot_minus_room[i] = hot_minus_room[i]*0.5
                    interf_term[i] = interf_term[i]*0.5
                if i in weights[3][0]:
                    hot_minus_room[i] = hot_minus_room[i]*0.5
                    interf_term[i] = interf_term[i]*0.5

            yi = int((valy - begin)/stepsize)
            zi = int((valz - begin)/stepsize)
            ycoords[yi, zi] = valy
            zcoords[yi, zi] = valz
            dtheta = np.abs(theta[0]-theta[numtheta+1])
            dphi = np.abs(phi[0]-phi[1])
            IPT_TOT_HMR[yi, zi] = c*n/(8*np.pi)*np.sum(hot_minus_room)*dtheta*dphi
            IPT_TOT_INT[yi, zi] = c*n/(8*np.pi)*np.sum(interf_term)*dtheta*dphi

    return ycoords, zcoords, IPT_TOT_HMR, IPT_TOT_INT

def write_IPT_terms():
    ycoords, zcoords, IPT_HMR, IPT_INT = calculated_IPT()
    ywrite = np.ravel(ycoords); zwrite = np.ravel(zcoords)
    IPT_HMR_write = np.ravel(IPT_HMR)
    IPT_INT_write = np.ravel(IPT_INT)
    file = open(str('pt_signal.txt'),'w')
    file.write(str('y') + '\t' + str('z') + '\t' + str('H-R') + '\t' + str('INT') + '\n')
    for i in range(0, len(ywrite)):
        file.write(str(ywrite[i]) + '\t' + str(zwrite[i]) + '\t' + str(IPT_HMR_write[i]) + '\t' + str(IPT_INT_write[i]) + '\n')
write_IPT_terms()



