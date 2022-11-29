import numpy as np
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from scipy import integrate


class Mie_Theory:

    def __init__(self, radius, nb, wave, drude_or_jc):
        """Defines the different system parameters.
        Keyword arguments:
        radius -- [cm] radius of NP
        n -- [unitless] refractive index of background
        wave -- [cm] wavelength of driving laser
        """
        self.radius = radius
        self.wave = wave
        self.nb = nb
        self.drude_or_jc = drude_or_jc

    def psi(self, n, rho):
        return rho * spherical_jn(n, rho)

    def psi_prime(self, n, rho):
        return spherical_jn(n, rho) +\
            rho * spherical_jn(n, rho, derivative=True)

    def hankel(self, n, rho):
        return spherical_jn(n, rho) + 1j * spherical_yn(n, rho)

    def hankel_prime(self, n, rho):
        return spherical_jn(n, rho, derivative=True) +\
            1j * spherical_yn(n, rho, derivative=True)

    def xi(self, n, rho):
        return rho * self.hankel(n, rho)

    def xi_prime(self, n, rho):
        return self.hankel(n, rho) + rho * self.hankel_prime(n, rho)

    def mie_coefficent(self, n):
        '''Calculates Mie coefficients, a and b.
        n: multipole label (n = 1 is the dipole)
        '''
        if self.drude_or_jc == 'drude':
            data = np.loadtxt('drude_interp.txt', skiprows=3)
        if self.drude_or_jc == 'jc':
            data = np.loadtxt('auJC_interp.tab', skiprows=3)
        wave_raw = data[:, 0] * 1E-4  # cm
        n_re_raw = data[:, 1]
        n_im_raw = data[:, 2]
        idx = np.where(np.in1d(np.round(wave_raw, 7),
                               np.round(self.wave, 7)))[0]
        n_re = n_re_raw[idx]
        n_im = n_im_raw[idx]
        m = (n_re + 1j * n_im) / self.nb
        k = 2 * np.pi * self.nb / self.wave
        x = k * self.radius
        numer_a = m * self.psi(n, m * x) * self.psi_prime(n, x)\
            - self.psi(n, x) * self.psi_prime(n, m * x)
        denom_a = m * self.psi(n, m * x) * self.xi_prime(n, x)\
            - self.xi(n, x) * self.psi_prime(n, m * x)
        numer_b = self.psi(n, m * x) * self.psi_prime(n, x)\
            - m * self.psi(n, x) * self.psi_prime(n, m * x)
        denom_b = self.psi(n, m * x) * self.xi_prime(n, x)\
            - m * self.xi(n, x) * self.psi_prime(n, m * x)
        an = numer_a / denom_a
        bn = numer_b / denom_b
        return an, bn

    def cross_sects(self, nTOT):
        ''' Calculates Mie cross-sections.
        nTOT: total number of multipoles (nTOT = 1 is just dipole)
        '''
        a_n = np.zeros((nTOT, 1), dtype=complex)
        b_n = np.zeros((nTOT, 1), dtype=complex)
        ext_insum = np.zeros((len(np.arange(1, nTOT + 1)), 1))
        sca_insum = np.zeros((len(np.arange(1, nTOT + 1)), 1))
        nTOT_ar = np.arange(1, nTOT + 1)
        for idx, ni in enumerate(nTOT_ar):
            a_n[idx, :], b_n[idx, :] = self.mie_coefficent(n=ni)
            ext_insum[idx, :] = (2 * ni + 1) * \
                np.real(a_n[idx, :] + b_n[idx, :])
            sca_insum[idx, :] = (2 * ni + 1) * \
                (np.abs(a_n[idx, :])**2 + np.abs(b_n[idx, :])**2)
        k = 2 * np.pi * self.nb / self.wave
        C_ext = 2 * np.pi / k**2 * np.sum(ext_insum, axis=0)
        C_sca = 2 * np.pi / k**2 * np.sum(sca_insum, axis=0)
        C_abs = C_ext - C_sca
        return C_abs, C_sca

# class Alpha_Others:
    # def alpha_sph_QS(self, n, r):
    #     nb = self.nb_T0
    #     return r**3 * nb**2 * (n**2 - nb**2) / (n**2 + 2 * nb**2)

    # def alpha_sph_MW(self, n, r, q):
    #     nb = self.nb_T0
    #     return 1 / 3 * r**3 * nb**2 * (n**2 - nb**2) /\
    #         (nb**2 + q * (n**2 - nb**2))

    # def alpha_CS_QS(self, n1, n2, r1, r2):
    #     nb = self.nb_T0
    #     f = r1**3 / r2**3
    #     numerator = r2**3 * nb**2 * ((n2**2 - nb**2) * (n1**2 + 2 * n2**2)
    #                                  + f * (n1**2 - n2**2)
    #                                  * (nb**2 + 2 * n2**2))
    #     denominator = ((n2**2 + 2 * nb**2) * (n1**2 + 2 * n2**2)
    #                    + 2 * f * (n2**2 - nb**2) * (n1**2 - n2**2))
    #     return numerator / denominator

    # d alpha / dn #

    # def d_alpha_sphQS_dn(self, n, r):
    #     nb = self.nb_T0
    #     return 6 * n * nb**4 * r**3 / (n**2 + 2 * nb**2)**2

    # def d_alpha_sphMW_dn(self, n, r, q):
    #     nb = self.nb_T0
    #     return 2 * n * nb**4 * r**3 / (3 * (nb**2 * (-1 + q) - n**2 * q)**2)

    # d alpha / dn1 #

    # def d_alpha_CS_QS_dn1(self, n1, n2, r1, r2):
    #     nb = self.nb_T0
    #     f = r1**3 / r2**3
    #     numerator = 54 * f * n1 * n2**4 * nb**4 * r2**3
    #     denominator = ((-2 * (-1 + f) * n2**4 + 2 * (2 + f) * n2**2 * nb**2
    #                     + n1**2 * ((1 + 2 * f) * n2**2
    #                                - 2 * (-1 + f) * nb**2))**2)
    #     return numerator / denominator
   # d alpha / dn2 at T0 #
    # def d_alpha_CS_QS_dn2(self, n1, n2, r1, r2):
    #     nb = self.nb_T0
    #     f = r1**3 / r2**3
    #     numerator = -(6 * (-1 + f) * n2
    #                   * ((1 + 2 * f) * n1**4 - 4 * (-1 + f) * n1**2 * n2**2
    #                      + 2 * (2 + f) * n2**4) * nb**4 * r2**3)
    #     denominator = ((-2 * (-1 + f) * n2**4 + 2 * (2 + f) * n2**2 * nb**2
    #                     + n1**2 * ((1 + 2 * f) * n2**2
    #                                - 2 * (-1 + f) * nb**2))**2)
    #     return numerator / denominator

    # def alpha_atT0(self):
    #     ng_T0, _ = self.eps_gold_room(selected_waves=self.wave_probe)

    #     if self.kind == 'glyc_sph_QS':
    #         alpha_atT0 = self.alpha_sph_QS(n=self.nb_T0, r=self.shell_radius)

    #     if self.kind == 'gold_sph_QS':
    #         alpha_atT0 = self.alpha_sph_QS(n=ng_T0, r=self.radius)

    #     if self.kind == 'glyc_sph_MW':
    #         alpha_atT0 = self.alpha_sph_MW(n=self.nb_T0,
    #                                        r=self.shell_radius,
    #                                        q=self.qi(self.shell_radius))

    #     if self.kind == 'gold_sph_MW':
    #         alpha_atT0 = self.alpha_sph_MW(n=ng_T0,
    #                                        r=self.radius,
    #                                        q=self.qi(self.radius))

    #     if self.kind == 'coreshell_QS':
    #         alpha_atT0 = self.alpha_CS_QS(n1=ng_T0,
    #                                       n2=self.nb_T0,
    #                                       r1=self.radius,
    #                                       r2=self.shell_radius)

    #     if self.kind == 'coreshell_MW':
    #         alpha_atT0 = self.alpha_CS_MW(n1=ng_T0,
    #                                       n2=self.nb_T0,
    #                                       r1=self.radius,
    #                                       r2=self.shell_radius,
    #                                       q1=self.qi(self.radius),
    #                                       q2=self.qi(self.shell_radius))
    #     return alpha_atT0

    ########################################################
   # def dalphadT_atT0(self, separate=False):
    #     ng_T0, _ = self.eps_gold_room(selected_waves=self.wave_probe)
    #     dn1_dT, dn2_dT = self.find_dnj_dT()

    #     if self.kind == 'glyc_sph_QS':
    #         dgold_dT_atT0 = 0
    #         dglyc_dT_atT0 = self.d_alpha_sphQS_dn(
    #             n=self.nb_T0, r=self.shell_radius)

    #     if self.kind == 'gold_sph_QS':
    #         dgold_dT_atT0 = self.d_alpha_sphQS_dn(n=ng_T0, r=self.radius)
    #         dglyc_dT_atT0 = 0

    #     if self.kind == 'glyc_sph_MW':
    #         dgold_dT_atT0 = 0
    #         dglyc_dT_atT0 = self.d_alpha_sphMW_dn(
    #             n=self.nb_T0,
    #             r=self.shell_radius,
    #             q=self.qi(ri=self.shell_radius))

    #     if self.kind == 'gold_sph_MW':
    #         dgold_dT_atT0 = self.d_alpha_sphMW_dn(
    #             n=ng_T0,
    #             r=self.radius,
    #             q=self.qi(ri=self.radius))
    #         dglyc_dT_atT0 = 0

    #     if self.kind == 'coreshell_QS':
    #         dgold_dT_atT0 = self.d_alpha_CS_QS_dn1(
    #             n1=ng_T0,
    #             n2=self.nb_T0,
    #             r1=self.radius,
    #             r2=self.shell_radius)
    #         dglyc_dT_atT0 = self.d_alpha_CS_QS_dn2(
    #             n1=ng_T0,
    #             n2=self.nb_T0,
    #             r1=self.radius,
    #             r2=self.shell_radius)

    #     if self.kind == 'coreshell_MW':
    #         dgold_dT_atT0 = self.d_alpha_CS_MW_dn1(
    #             n1=ng_T0,
    #             n2=self.nb_T0,
    #             r1=self.radius,
    #             r2=self.shell_radius,
    #             q1=self.qi(self.radius),
    #             q2=self.qi(self.shell_radius))
    #         dglyc_dT_atT0 = self.d_alpha_CS_MW_dn2(
    #             n1=ng_T0,
    #             n2=self.nb_T0,
    #             r1=self.radius,
    #             r2=self.shell_radius,
    #             q1=self.qi(self.radius),
    #             q2=self.qi(self.shell_radius))

    #     if separate is False:
    #         return dgold_dT_atT0, dn1_dT, dglyc_dT_atT0, dn2_dT

    #     if separate is True:
    #         return dgold_dT_atT0, dn1_dT, dglyc_dT_atT0, dn2_dT


class Photothermal_Image:

    def __init__(self,
                 radius,        # [cm] radius of NP
                 wave_pump,     # [cm] wavelength of pump laser
                 abs_cross,     # [cm^2] absorption of NP at wave_pump
                 P0h,           # [erg/s] incident pump power
                 wave_pr,    # [cm] wavelength of probe laser
                 nb_T0,         # [unitless] background refractive index at T0
                 delta_z,       # [cm] z offset between probe focus and pump
                 zprobe_focus,  # [cm] z position of probe focus
                 order,
                 ):
        self.radius = radius
        self.wave_pump = wave_pump
        self.abs_cross = abs_cross
        self.P0h = P0h
        self.wave_pr = wave_pr
        self.nb_T0 = nb_T0
        self.delta_z = delta_z
        self.zprobe_focus = zprobe_focus
        self.order = order
        self.kappa = 0.6 * (1E7 / 100)  # erg/ (s cm K)
        self.C = (1.26 * 2.35 * 1E7)  # erg / (cm^3 K)
        self.Omega = 1E5  # 1/s (100 kHz)
        self.rth = np.sqrt(2 * self.kappa / (self.Omega * self.C))
        self.shell_radius = self.rth

    def waist_at_focus(self, wave):
        NA = 1.25
        return wave * 0.6 / NA

    def waist_at_z(self, wave, z, z_at_focus):  # [cm]
        waist_at_focus = self.waist_at_focus(wave=wave)
        waist_z = waist_at_focus * \
            np.sqrt(1 + ((z - z_at_focus) / self.zR(wave=wave))**2)
        return waist_z

    def convert_k(self, wave):
        return 2 * np.pi * self.nb_T0 / wave

    def zR(self, wave):
        waist_at_focus = self.waist_at_focus(wave=wave)
        return np.pi * waist_at_focus**2 * self.nb_T0 / wave

    def eps_au(self, wave):
        JCdata = np.loadtxt('auJC_interp.tab', skiprows=3)
        waveJC = np.round(JCdata[:, 0] * 1E-4, 7)  # cm
        n_re = JCdata[:, 1]
        n_im = JCdata[:, 2]
        idx = np.where(np.in1d(waveJC, wave))[0]
        n = n_re[idx] + 1j * n_im[idx]
        eps = n_re[idx]**2 - \
            n_im[idx]**2 + 1j * (2 * n_re[idx] * n_im[idx])
        return n, eps

    def find_dnj_dT(self):
        dn1_dT = -1E-4 - 1j * 1E-4
        dn2_dT = -10**(-4)
        # T = int(np.round(self.Pabs()
        #                  / (4 * np.pi * self.kappa * self.radius)))
        # if T > 99:
        #     T = int(99)
        # data = np.loadtxt(str('../temp_dep_gold/au_Conor_')
        #                   + str(T) + str('K.txt'), skiprows=3)
        # wave = np.round(data[:, 0], 3)
        # idx = (np.abs(wave - self.wave_pump * 1E4)).argmin()
        # fa = data[idx, 1] + 1j * data[idx, 2]
        # data_plus_step = np.loadtxt(str('../temp_dep_gold/au_Conor_')
        #                             + str(T + 1) + str('K.txt'),
        #                             skiprows=3)
        # fa_plus_step = data_plus_step[idx, 1] + 1j * data_plus_step[idx, 2]
        # dnj_dT = (fa_plus_step - fa) / 1
        # dn1_dT = dnj_dT
        # dn2_dT = -10**(-4)
        return dn1_dT, dn2_dT

    def qi(self, ri):
        xi = 2 * np.pi * ri / self.wave_pr
        return 1 / 3 * (1 - xi**2 - 1j * 2 / 9 * xi**3)

    def alpha_T0(self, n1, n2, r1, r2):
        e1 = n1**2
        e2 = n2**2
        eb = self.nb_T0**2
        q1 = self.qi(r1)
        q2 = self.qi(r2)
        numerator = (1 / 3 * r2**3 * eb
                     * ((e2 - eb) * (e1 * q1 - e2 * (q1 - 1)) * r2**3
                        - (e1 - e2) * (e2 * (q2 - 1) - eb * q2) * r1**3))
        denominator = ((e1 * q1 - e2 * (q1 - 1)) * (e2 * q2 - eb * (q2 - 1))
                       * r2**3 - (e1 - e2) * (e2 - eb) * q2 * (q2 - 1) * r1**3)
        return numerator / denominator

    def d_alpha_dn1(self, n1, n2, r1, r2):
        nb = self.nb_T0
        q1 = self.qi(r1)
        q2 = self.qi(r2)
        numerator = 2 * n1 * n2**4 * nb**4 * r1**3 * r2**6
        denominator = (3 * ((n1 - n2) * (n1 + n2) * (n2 - nb) * (n2 + nb)
                            * (q2 - 1) * q2 * r1**3
                            + (n2**2 * (q1 - 1) - n1**2 * q1)
                            * (-nb**2 * (q2 - 1) + n2**2 * q2) * r2**3)**2)
        return numerator / denominator

    def d_alpha_dn2(self, n1, n2, r1, r2):
        nb = self.nb_T0
        q1 = self.qi(r1)
        q2 = self.qi(r2)
        numer = (2 * n2 * nb**4 * r2**3
                 * (r1**6 * (n1**2 - n2**2)**2 * (q2 - 1) * q2
                    + r1**3 * (n1**4 * q1 * (1 - 2 * q2)
                               + n2**4 * (-1 + q1 + 2 * q2 - 2 * q1 * q2)
                               + 2 * n1**2 * n2**2 * (-q1 - q2 + 2 * q1 * q2))
                     * r2**3 + (n2**2 * (q1 - 1) - n1**2 * q1)**2 * r2**6))
        denom = (3 * (r1**3 * (n1 - n2) * (n1 + n2) * (n2 - nb) * (n2 + nb)
                      * (q2 - 1) * q2 + (n2**2 * (q1 - 1) - n1**2 * q1)
                      * (-nb**2 * (q2 - 1) + n2**2 * q2) * r2**3)**2)
        return numer / denom

    def d_alpha2_dn12(self, n1, n2, r1, r2):
        nb = self.nb_T0
        q1 = self.qi(r1)
        q2 = self.qi(r2)
        numer = -((2 * r1**3 * n2**4 * nb**4 * r2**6
                   * (r1**3 * (3 * n1**2 + n2**2) * (n2 - nb)
                       * (n2 + nb) * (q2 - 1) * q2
                       - (n2**2 * (q1 - 1) + 3 * n1**2 * q1)
                       * (-nb**2 * (q2 - 1) + n2**2 * q2) * r2**3)))
        denom = (3 * (r1**3 * (n1 - n2) * (n1 + n2) * (n2 - nb)
                      * (n2 + nb) * (q2 - 1) * q2
                      + (n2**2 * (q1 - 1) - n1**2 * q1)
                      * (-nb**2 * (q2 - 1) + n2**2 * q2) * r2**3)**3)
        return numer / denom

    def d_alpha2_dn1dn2(self, n1, n2, r1, r2):
        nb = self.nb_T0
        q1 = self.qi(r1)
        q2 = self.qi(r2)
        numer = -((8 * r1**3 * n1 * n2**3 * nb**4 * r2**6
                   * (-r1**3 * (n2**4 - n1**2 * nb**2) * (q2 - 1)
                      * q2 + (-n1**2 * nb**2 * q1 * (q2 - 1)
                              + n2**4 * (q1 - 1) * q2) * r2**3)))
        denom = (3 * (r1**3 * (n1 - n2) * (n1 + n2) * (n2 - nb)
                      * (n2 + nb) * (q2 - 1) * q2
                      + (n2**2 * (q1 - 1) - n1**2 * q1)
                      * (-nb**2 * (q2 - 1) + n2**2 * q2) * r2**3)**3)
        return numer / denom

    def d_alpha2_dn22(self, n1, n2, r1, r2):
        nb = self.nb_T0
        q1 = self.qi(r1)
        q2 = self.qi(r2)
        numer = (2 * nb**4 * r2**3
                 * (r1**9 * (n1**2 - n2**2)**3 * (3 * n2**2 + nb**2)
                    * (q2 - 1)**2 * q2**2 - r1**6 * (q2 - 1) * q2
                    * (n2**6 * (q1 - 1) * (n2**2 * (3 - 9 * q2) + nb**2
                                           * (2 - 3 * q2)) + n1**6 * q1
                        * (nb**2 * (-2 + 3 * q2) + n2**2 * (-3 + 9 * q2))
                        + 3 * n1**4 * n2**2
                        * (nb**2 * (1 + q1 * (2 - 3 * q2) + q2)
                           + 3 * n2**2 * (q1 + q2 - 3 * q1 * q2))
                        + n1**2 * n2**4
                        * (n2**2 * (-1 - 9 * q1 + 9 * (-2 + 3 * q1) * q2)
                           + 3 * nb**2 * (1 - 2 * q2 + q1 * (-2 + 3 * q2))))
                     * r2**3 + r1**3
                     * (-3 * n2**8 * (q1 - 1)**2 * q2 * (-2 + 3 * q2)
                        + n1**6 * nb**2 * q1**2 * (q2 - 1) * (-1 + 3 * q2)
                        - n2**6 * (q1 - 1)
                        * (nb**2 * (q1 - 1) * (q2 - 1) * (3 * q2 - 1)
                           + n1**2 * q2 * (1 + 9 * q1 * (2 - 3 * q2)
                                           + 9 * q2))
                        + 3 * n1**4 * n2**2 * q1
                        * (n1**2 * q1 * q2
                            * (-2 + 3 * q2) - nb**2 * (q2 - 1)
                            * (-1 - q1 - 2 * q2 + 3 * q1 * q2))
                        + 3 * n1**2 * n2**4
                        * (3 * n1**2 * q1 * q2
                           * (-1 + 2 * q1 + 2 * q2 - 3 * q1 * q2)
                           + nb**2 * (q1 - 1) * (q2 - 1)
                           * (-q1 - q2 + 3 * q1 * q2))) * r2**6
                     + (n2**2 * (q1 - 1) - n1**2 * q1)**3
                     * (nb**2 * (q2 - 1) + 3 * n2**2 * q2) * r2**9))
        denom = (3 * (-r1**3 * (n1 - n2) * (n1 + n2) * (n2 - nb)
                      * (n2 + nb) * (q2 - 1) * q2
                      + (-n2**2 * (q1 - 1)
                         + n1**2 * q1) * (-nb**2 * (q2 - 1) + n2**2 * q2)
                      * r2**3)**3)
        return numer / denom

    def lock_in_integral(self, integrand):
        tau = 2 * np.pi / self.Omega

        def sin_func_re(t):
            return np.real(integrand(t) * np.sin(self.Omega * t))

        def sin_func_im(t):
            return np.imag(integrand(t) * np.sin(self.Omega * t))

        def cos_func_re(t):
            return np.real(integrand(t) * np.cos(self.Omega * t))

        def cos_func_im(t):
            return np.imag(integrand(t) * np.cos(self.Omega * t))

        int_sin_re, _ = integrate.quad(sin_func_re, 0, tau)
        int_sin_im, _ = integrate.quad(sin_func_im, 0, tau)
        int_cos_re, _ = integrate.quad(cos_func_re, 0, tau)
        int_cos_im, _ = integrate.quad(cos_func_im, 0, tau)

        return np.array([1 / tau * (int_sin_re + 1j * int_sin_im),
                         1 / tau * (int_cos_re + 1j * int_cos_im)])

    def I_T1(self):
        def integrand(t):
            return self.T1(t)
        return self.lock_in_integral(integrand)

    def I_T2(self):
        def integrand(t):
            return self.T2(t)
        return self.lock_in_integral(integrand)

    def I_T1_sqrd(self):
        def integrand(t):
            return np.abs(self.T1(t))**2
        return self.lock_in_integral(integrand)

    def I_T2_sqrd(self):
        def integrand(t):
            return np.abs(self.T2(t))**2
        return self.lock_in_integral(integrand)

    def I_T1T2(self):
        def integrand(t):
            return self.T1(t) * self.T2(t)
        return self.lock_in_integral(integrand)

    def I_f_sqrd(self, A1, A2):
        def integrand(t):
            return np.abs(A1 * self.T1(t) + A2 * self.T2(t))**2
        return self.lock_in_integral(integrand)

    def I_s_sqrd(self, A1, A2, A3):
        def integrand(t):
            return 1 / 4 * np.abs(A1 * self.T1(t)**2
                                  + 2 * A2 * self.T1(t) * self.T2(t)
                                  + A3 * self.T2(t)**2)**2
        return self.lock_in_integral(integrand)

    def I_re_zf(self, A1, A2, alpha0):
        def integrand(t):
            return 2 * np.real(alpha0
                               * np.conj(A1 * self.T1(t)
                                         + A2 * self.T2(t)))
        return self.lock_in_integral(integrand)

    def I_re_zs(self, A1, A2, A3, alpha0):
        def integrand(t):
            return 2 * np.real(alpha0
                               * 1 / 2
                               * np.conj(A1 * self.T1(t)**2
                                         + 2 * A2 * self.T1(t)
                                         * self.T2(t)
                                         + A3 * self.T2(t)**2))
        return self.lock_in_integral(integrand)

    def I_re_fs(self, A1, A2, A3, A4, A5):

        def integrand(t):
            return 2 * np.real((A1 * self.T1(t) + A2 * self.T2(t))
                               * 1 / 2
                               * np.conj(A3 * self.T1(t)**2
                                         + 2 * A4 * self.T1(t)
                                         * self.T2(t)
                                         + A5 * self.T2(t)**2))
        return self.lock_in_integral(integrand)


    def I_a_terms(self, n1, n2, r1, r2, dn1_dT, dn2_dT):
        alphaT0 = self.alpha_T0(n1, n2, r1, r2)
        da_dn1 = self.d_alpha_dn1(n1, n2, r1, r2)
        da_dn2 = self.d_alpha_dn2(n1, n2, r1, r2)
        d2a_dn12 = self.d_alpha2_dn12(n1, n2, r1, r2)
        d2a_dn1dn2 = self.d_alpha2_dn1dn2(n1, n2, r1, r2)
        d2a_dn22 = self.d_alpha2_dn22(n1, n2, r1, r2)
        I_T1 = self.I_T1()
        I_T2 = self.I_T2()
        I_T1_sqrd = self.I_T1_sqrd()
        I_T2_sqrd = self.I_T2_sqrd()
        I_T1T2 = self.I_T1T2()
        first_ord = da_dn1 * dn1_dT * I_T1 + da_dn2 * dn2_dT * I_T2
        sec_ord = 1 / 2 * (d2a_dn12 * (dn1_dT)**2 * I_T1_sqrd
                           + d2a_dn22 * (dn2_dT)**2 * I_T2_sqrd
                           + 2 * d2a_dn1dn2 * dn1_dT * dn2_dT * I_T1T2)

        mod_f = self.I_f_sqrd(A1=(da_dn1 * dn1_dT), A2=(da_dn2 * dn2_dT))
        mod_s = self.I_s_sqrd(A1=(d2a_dn12 * (dn1_dT)**2),
                              A2=(da_dn1 * dn1_dT * da_dn2 * dn2_dT),
                              A3=(d2a_dn22 * (dn2_dT)**2))
        re_zf = self.I_re_zf(A1=(da_dn1 * dn1_dT), A2=(da_dn2 * dn2_dT),
                             alpha0=alphaT0)
        re_zs = self.I_re_zs(A1=(d2a_dn12 * (dn1_dT)**2),
                             A2=(da_dn1 * dn1_dT * da_dn2 * dn2_dT),
                             A3=(d2a_dn22 * (dn2_dT)**2),
                             alpha0=alphaT0)
        re_fs = self.I_re_fs(A1=(da_dn1 * dn1_dT), A2=(da_dn2 * dn2_dT),
                             A3=(d2a_dn12 * (dn1_dT)**2),
                             A4=(da_dn1 * dn1_dT * da_dn2 * dn2_dT),
                             A5=(d2a_dn22 * (dn2_dT)**2))
        if self.order == 'first':
            I_conj_alpha = np.conj(first_ord)
            I_alpha_modsqrd = np.real(mod_f + re_zf)

        if self.order == 'second':
            I_conj_alpha = np.conj(first_ord + sec_ord)
            I_alpha_modsqrd = np.real(mod_f + mod_s + re_zf + re_zs + re_fs)

        return I_conj_alpha, I_alpha_modsqrd

    def Pabs(self):
        zpump_focus = self.zprobe_focus - self.delta_z
        waist_at_focus_pump = self.waist_at_focus(wave=self.wave_pump)
        waist_at_NP_pump = self.waist_at_z(
            wave=self.wave_pump,
            z=0,
            z_at_focus=zpump_focus)
        I_at_focus = 2 * self.P0h / (np.pi * waist_at_focus_pump**2)
        Int = I_at_focus * (waist_at_focus_pump / waist_at_NP_pump)**2
        return self.abs_cross * Int

    def T(self, r, t):
        r1 = self.radius
        r2 = self.shell_radius
        const = self.Pabs() / (8 * np.pi * self.kappa * r)
        pref = (np.exp(-(r - r1) / r2)
                / (((r1 + r2) / r2)**2 + (r1 / r2)**2))
        costerm = (r1 + r2) / r2 * np.cos(self.Omega * t - (r - r1) / r2)
        sinterm = r1 / r2 * np.sin(self.Omega * t - (r - r1) / r2)
        return const * (1 + pref * (costerm + sinterm))

    def T1(self, t):
        r1 = self.radius
        return self.T(r1, t)

    def T2(self, t):
        r1 = self.radius
        r2 = self.shell_radius
        V = 4 / 3 * np.pi * r2**3

        def func(r):
            return r**2 * self.T(r, t)

        integral, err = integrate.quad(func, r1, r2)
        return 4 * np.pi / V * integral

    def pt_signal(self):
        k = self.convert_k(self.wave_pr)
        zR = self.zR(wave=self.wave_pr)
        zp = self.zprobe_focus
        z = 1
        r = z
        E0 = 1
        G = k**2 * np.exp(1j * k * r) / r
        dn1_dT, dn2_dT = self.find_dnj_dT()
        n1, _ = self.eps_au(wave=self.wave_pr)
        I_conj_a, I_a_sqrd = self.I_a_terms(n1=n1,
                                            n2=self.nb_T0,
                                            r1=self.radius,
                                            r2=self.shell_radius,
                                            dn1_dT=dn1_dT,
                                            dn2_dT=dn2_dT)
        Epr_np = E0 / np.sqrt(1 + (-zp / zR)**2) * np.exp(-1j * k * zp)\
            * np.exp(-1j * np.arctan2(-zp, zR))

        Epr_d = E0 / (1j * z / zR) * np.exp(1j * k * (z - zp))

        Phiint = (2 * np.real(Epr_d * np.conj(G) * I_conj_a * np.conj(Epr_np))
                  / np.abs(Epr_d)**2)

        Phisca = (np.abs(G)**2 * I_a_sqrd * np.abs(Epr_np)**2
                  / np.abs(Epr_d)**2)


        zpump_focus = self.zprobe_focus - self.delta_z

        waist_at_NP_pump = self.waist_at_z(
            wave=self.wave_pump,
            z=0,
            z_at_focus=zpump_focus)

        return np.real(Phiint), np.real(Phisca), waist_at_NP_pump


