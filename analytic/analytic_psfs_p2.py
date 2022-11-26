import numpy as np
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from scipy import integrate


class Mie_Theory:

    def __init__(self, radius, nb, selected_waves, drude_or_jc):
        """Defines the different system parameters.
        Keyword arguments:
        radius -- [cm] radius of NP
        n -- [unitless] refractive index of background
        wave_pump -- [cm] wavelength of driving laser
        """
        self.radius = radius
        self.selected_waves = selected_waves
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
                               np.round(self.selected_waves, 7)))[0]
        n_re = n_re_raw[idx]
        n_im = n_im_raw[idx]
        m = (n_re + 1j * n_im) / self.nb
        k = 2 * np.pi * self.nb / self.selected_waves
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
        a_n = np.zeros((nTOT, len(self.selected_waves)), dtype=complex)
        b_n = np.zeros((nTOT, len(self.selected_waves)), dtype=complex)
        ext_insum = np.zeros((len(np.arange(1, nTOT + 1)),
                              len(self.selected_waves)))
        sca_insum = np.zeros((len(np.arange(1, nTOT + 1)),
                              len(self.selected_waves)))
        nTOT_ar = np.arange(1, nTOT + 1)
        for idx, ni in enumerate(nTOT_ar):
            a_n[idx, :], b_n[idx, :] = self.mie_coefficent(n=ni)
            ext_insum[idx, :] = (2 * ni + 1) * \
                np.real(a_n[idx, :] + b_n[idx, :])
            sca_insum[idx, :] = (2 * ni + 1) * \
                (np.abs(a_n[idx, :])**2 + np.abs(b_n[idx, :])**2)
        k = 2 * np.pi * self.nb / self.selected_waves
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



class Alpha_CW_MW:
    def qi(self, ri):
        xi = 2 * np.pi * ri / self.wave_probe
        return 1 / 3 * (1 - xi**2 - 1j * 2 / 9 * xi**3)

    def alpha_CS_MW(self, n1, n2, r1, r2, q1, q2):
        e1 = n1**2
        e2 = n2**2
        eb = self.nb_T0**2
        numerator = (1 / 3 * r2**3 * eb
                     * ((e2 - eb) * (e1 * q1 - e2 * (q1 - 1)) * r2**3
                        - (e1 - e2) * (e2 * (q2 - 1) - eb * q2) * r1**3))

        denominator = ((e1 * q1 - e2 * (q1 - 1)) * (e2 * q2 - eb * (q2 - 1))
                       * r2**3 - (e1 - e2) * (e2 - eb) * q2 * (q2 - 1) * r1**3)
        return numerator / denominator

    def d_alpha_CS_MW_dn1(self, n1, n2, r1, r2, q1, q2):
        nb = self.nb_T0
        numerator = 2 * n1 * n2**4 * nb**4 * r1**3 * r2**6
        denominator = (3 * ((n1 - n2) * (n1 + n2) * (n2 - nb) * (n2 + nb)
                            * (q2 - 1) * q2 * r1**3
                            + (n2**2 * (q1 - 1) - n1**2 * q1)
                            * (-nb**2 * (q2 - 1) + n2**2 * q2) * r2**3)**2)
        return numerator / denominator

    def d_alpha_CS_MW_dn2(self, n1, n2, r1, r2, q1, q2):
        nb = self.nb_T0
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

    def d_alpha2_CS_MW_dn12(self, n1, n2, r1, r2, q1, q2):
        nb = self.nb_T0
        numer = -((2 * r1**3 * n2**4 * nb**4 * r2**6
                   * (a**3 * (3 * n1**2 + n2**2) * (n2 - nb)
                       * (n2 + nb) * (q2 - 1) * q2
                       - (n2**2 * (q1 - 1) + 3 * n1**2 * q1)
                       * (-nb**2 * (q2 - 1) + n2**2 * q2) * r2**3)))
        denom = (3 * (r1**3 * (n1 - n2) * (n1 + n2) * (n2 - nb)
                      * (n2 + nb) * (q2 - 1) * q2
                      + (n2**2 * (q1 - 1) - n1**2 * q1)
                      * (-nb**2 * (q2 - 1) + n2**2 * q2) * r2**3)**3)
        return numer / denom

    def d_alpha2_CS_MW_dn1dn2(self, n1, n2, r1, r2, q1, q2):
        nb = self.nb_T0
        numer = -((8 * r1**3 * n1 * n2**3 * nb**4 * r2**6
                   * (-r1**3 * (n2**4 - n1**2 * nb**2) * (q2 - 1)
                      * q2 + (-n1**2 * nb**2 * q1 * (q2 - 1)
                              + n2**4 * (q1 - 1) * q2) * r2**3)))
        denom = (3 * (r1**3 * (n1 - n2) * (n1 + n2) * (n2 - nb)
                      * (n2 + nb) * (q2 - 1) * q2
                      + (n2**2 * (q1 - 1) - n1**2 * q1)
                      * (-nb**2 * (q2 - 1) + n2**2 * q2) * r2**3)**3)

        return numer / denom

    def d_alpha2_CS_MW_dn22(self, n1, n2, r1, r2, q1, q2):
        nb = self.nb_T0
        numer = (2 * nb**4 * r2**3
                 * (a**9 * (n1**2 - n2**2)**3 * (3 * n2**2 + nb**2)
                    * (q2 - 1)**2 * q2**2 - r1**6 * (q2 - 1) * q2
                    * (n2**6 * (q1 - 1) * (n2**2 * (3 - 9 * q2) + nb**2
                                           * (2 - 3 * q2)) + n1**6 * q1
                        * (nb**2 * (-2 + 3 * q2) + n2**2 * (-3 + 9 * q2))
                        + 3 * n1**4 * n2**2
                        * (nb**2 * (1 + q1 * (2 - 3 * q2) + q2)
                           + 3 * n2**2 * (q1 + q2 - 3 * q1 * q2))
                        + n1**2 * n2**4
                        * (n2**2 * (-1 - 9 * q1 + 9 (-2 + 3 * q1) * q2)
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
                           * (-q1 - q2 + 3 * q1 * q2))) * rth**6
                     + (n2**2 * (q1 - 1) - n1**2 * q1)**3
                     * (nb**2 * (q2 - 1) + 3 * n2**2 * q2) * rth**9))
        denom = (3 * (-r1**3 * (n1 - n2) * (n1 + n2) * (n2 - nb)
                      * (n2 + nb) * (q2 - 1) * q2
                      + (-n2**2 * (q1 - 1)
                         + n1**2 * q1) * (-nb**2 * (q2 - 1) + n2**2 * q2)
                      * rth**3)**3)








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
                 wave_probe,    # [cm] wavelength of probe laser
                 nb_T0,         # [unitless] background refractive index at T0
                 delta_z,       # [cm] z offset between probe focus and pump
                 zprobe_focus,  # [cm] z position of probe focus
                 simple,
                 printit,
                 ):
        self.radius = radius
        self.wave_pump = wave_pump
        self.abs_cross = abs_cross
        self.P0h = P0h
        self.wave_probe = wave_probe
        self.nb_T0 = nb_T0
        self.delta_z = delta_z
        self.zprobe_focus = zprobe_focus
        self.simple = simple
        self.printit = printit
        self.kind = 'coreshell_MW'
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

    def eps_gold_room(self, selected_waves, drude=False):
        JCdata = np.loadtxt('auJC_interp.tab', skiprows=3)
        wave = np.round(JCdata[:, 0] * 1E-4, 7)  # cm
        n_re = JCdata[:, 1]
        n_im = JCdata[:, 2]
        idx = np.where(np.in1d(wave, selected_waves))[0]
        n = n_re[idx] + 1j * n_im[idx]
        eps = n_re[idx]**2 - \
            n_im[idx]**2 + 1j * (2 * n_re[idx] * n_im[idx])
        return n, eps

    def find_dnj_dT(self):
        if self.simple is True:
            dn1_dT = -1E-4 - 1j * 1E-4
            dn2_dT = -10**(-4)

        if self.simple is False:
            T = int(np.round(self.Pabs()
                             / (4 * np.pi * self.kappa * self.radius)))
            if T > 99:
                T = int(99)
            data = np.loadtxt(str('../temp_dep_gold/au_Conor_')
                              + str(T) + str('K.txt'), skiprows=3)
            wave = np.round(data[:, 0], 3)
            idx = (np.abs(wave - self.wave_pump[0] * 1E4)).argmin()
            fa = data[idx, 1] + 1j * data[idx, 2]
            data_plus_step = np.loadtxt(str('../temp_dep_gold/au_Conor_')
                                        + str(T + 1) + str('K.txt'),
                                        skiprows=3)
            fa_plus_step = data_plus_step[idx, 1] + 1j * data_plus_step[idx, 2]
            dnj_dT = (fa_plus_step - fa) / 1
            dn1_dT = dnj_dT
            dn2_dT = -10**(-4)
        return dn1_dT, dn2_dT

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

    # def pt_signal_as_written(self, which):
    #     w_focus_pr = self.waist_at_focus(wave=self.wave_probe)
    #     k = self.convert_k(self.wave_probe)
    #     zR = self.zR(wave=self.wave_probe)
    #     zp = self.zprobe_focus
    #     I1 = (np.exp(1) - 2 * np.cos(1) - np.sin(1)) / (4 * np.exp(1))
    #     I2 = (2 * np.sin(1) - np.cos(1)) / (4 * np.exp(1))
    #     if which == 'sin':
    #         Int = I1
    #     if which == 'cos':
    #         Int = I2
    #     if self.kind == 'gold_sph_QS' or self.kind == 'gold_sph_MW':
    #         V = 4 / 3 * np.pi * self.radius**3
    #     else:
    #         V = 4 / 3 * np.pi * self.shell_radius**3
    #     guoy = np.exp(1j * np.arctan2(-zp, zR))
    #     zpump_focus = self.zprobe_focus - self.delta_z
    #     waist_at_NP_pump = self.waist_at_z(
    #         wave=self.wave_pump,
    #         z=0,
    #         z_at_focus=zpump_focus)
    #     phi_int = 4 * k / (w_focus_pr**2 * np.sqrt(1 + (zp / zR)**2)) *\
    #         np.imag(guoy * np.conj(self.dalphadT_atT0())) * self.Pabs() \
    #         * self.rth**2 / (2 * self.kappa * V) * Int
    #     phi_scatt1 = k**4 / (zR**2 + zp**2) * \
    #         2 * np.real(np.conj(self.alpha_atT0())
    #                     * self.dalphadT_atT0()) * self.Pabs() \
    #         * self.rth**2 / (2 * self.kappa * V) * Int
    #     phi_scatt2 = k**4 / (zR**2 + zp**2) \
    #         * np.abs(self.dalphadT_atT0())**2\
    #         * (self.Pabs() * self.rth**2
    #             / (2 * self.kappa * V))**2 * Int
    #     return phi_int, phi_scatt1, phi_scatt2, waist_at_NP_pump

    # def h_minus_r(self, which, which_T):
    #     k = self.convert_k(self.wave_probe)  # cm^-1
    #     zR = self.zR(wave=self.wave_probe)
    #     zp = self.zprobe_focus
    #     z = 1
    #     r = z
    #     E0 = 1
    #     Omega = self.Omega
    #     E_atNP = E0 / np.sqrt(1 + (-zp / zR)**2) * \
    #         np.exp(-1j * k * zp) * np.exp(-1j * np.arctan2(-zp, zR))
    #     E_atd = E0 / (1j * z / zR) * np.exp(1j * k * (z - zp))
    #     E_atNP_modsqrd = np.abs(E_atNP)**2
    #     E_atd_modsqrd = np.abs(E_atd)**2
    #     G = k**2 * np.exp(1j * k * r) / r
    #     Vth = 4 / 3 * np.pi * self.rth**3
    #     A = np.exp(1)
    #     B = -np.sin(1) + np.exp(1) - 2 * np.cos(1)
    #     C = -np.cos(1) + 2 * np.sin(1)
    #     T_constant = self.Pabs() * self.rth**2 / \
    #         (4 * np.exp(1) * self.kappa * Vth)
    #     if which_T == 'Tavg':
    #         t = np.pi / (Omega)
    #         Tavg_t = T_constant * (A + B * np.sin(Omega * t)
    #                                + C * np.cos(Omega * t))
    #     if which_T == 'Tss':
    #         Tavg_t = self.Pabs() / (8 * np.pi * self.kappa * self.radius)
    #     EH = G * (self.alpha_atT0() + self.dalphadT_atT0() * Tavg_t) * E_atNP
    #     ER = G * self.alpha_atT0() * E_atNP
    #     H_minus_R = (1 / E_atd_modsqrd) * (np.abs(EH)**2 - np.abs(ER)**2)
    #     Int = (1 / E_atd_modsqrd) * 2 * np.real((E_atd * np.conj(EH - ER)))

    #     ################################################################
    #     # Photothermal Signal -- derived as we do in paper #

    #     if which == 'sin':
    #         term = B
    #     if which == 'cos':
    #         term = C
    #     Phisca = np.real(np.abs(G)**2 * term / 2 * T_constant * (
    #         2 * A * np.abs(self.dalphadT_atT0())**2 * T_constant
    #         + np.conj(self.dalphadT_atT0()) * self.alpha_atT0()
    #         + self.dalphadT_atT0() * np.conj(self.alpha_atT0()))
    #         * E_atNP_modsqrd / E_atd_modsqrd)
    #     Phiint = np.real(E_atd * np.conj(E_atNP) / E_atd_modsqrd
    #                      * np.conj(G) * np.conj(self.dalphadT_atT0())
    #                      * T_constant * term)
    #     return H_minus_R, Int, Phisca, Phiint

    def T(self, r, t, idx):
        r1 = self.radius
        r2 = self.shell_radius
        const = self.Pabs()[idx] / (8 * np.pi * self.kappa * r)
        pref = (np.exp(-(r - r1) / r2)
                / (((r1 + r2) / r2)**2 + (r1 / r2)**2))
        costerm = (r1 + r2) / r2 * np.cos(self.Omega * t - (r - r1) / r2)
        sinterm = r1 / r2 * np.sin(self.Omega * t - (r - r1) / r2)
        return const * (1 + pref * (costerm + sinterm))

    def T1(self, t, idx):
        r1 = self.radius
        return self.T(r1, t, idx)

    def T2(self, t, idx):
        r1 = self.radius
        r2 = self.shell_radius
        V = 4 / 3 * np.pi * r2**3

        def func(r):
            return r**2 * self.T(r, t, idx)

        integral, err = integrate.quad(func, r1, r2)
        return 4 * np.pi / V * integral

    def alphapt(self, t, idx):
        zeroth = self.alpha_atT0()
        da_dn1, dn1_dT, da_dn2, dn2_dT = dalphadT_atT0()
        da_dT1 = da_dn1 * dn1_dT
        da_dT2 = da_dn2 * dn2_dT

        d2a_dT12 = d2a_dn12 * (dn1_dT)**2
        d2a_dT22 = d2a_dn22 * (dn2_dT)**2
        d2a_dT1dT2 = da_dn1 * dn1_dT * da_dn2 * dn2_dT

        first = (da_dT1 * self.T1(t, idx) + da_dT2 * self.T2(t, idx))
        second = 1 / 2 * (d2a_dT12 * self.T1(t, idx)**2
                          + 2 * d2a_dT1dT2 * self.T1(t, idx) * self.T2(t, idx)
                          + d2a_dT22 * self.T2(t, idx)**2)
        return zeroth, first, second

    ####################################################################
    # Integral of Conj(Alpha) Term #

    def lock_in_integral(self, integrand, idx):

        def sin_func(t):
            return integrand(t) * np.sin(self.Omega * t)

        def cos_func(t):
            return integrand(t) * np.cos(self.Omega * t)

        integral_sin, _ = integrate.quad(sin_func, 0, 2 * np.pi / self.Omega)
        integral_cos, _ = integrate.quad(cos_func, 0, 2 * np.pi / self.Omega)
        return (self.Omega / (2 * np.pi) * integral_sin,
                self.Omega / (2 * np.pi) * integral_cos)

    def I_T1(self, idx):
        def integrand(t):
            return self.T1(t, idx)
        return self.lock_in_integral(integrand, idx)

    def I_T2(self, idx):
        def integrand(t):
            return self.T2(t, idx)
        return self.lock_in_integral(integrand, idx)

    def I_T1_sqrd(self, idx):
        def integrand(t):
            return self.T1(t, idx)**2
        return self.lock_in_integral(integrand, idx)

    def I_T2_sqrd(self, idx):
        def integrand(t):
            return self.T2(t, idx)**2
        return self.lock_in_integral(integrand, idx)

    def I_T1T2(self, idx):
        def integrand(t):
            return self.T1(t, idx) * self.T2(t, idx)
        return self.lock_in_integral(integrand, idx)

    def I_f_sqrd(self, idx):
        A1 = 1
        A2 = 1

        def integrand(t):
            return (A1 * self.T1(t, idx) + A2 * self.T2(t, idx))**2
        return self.lock_in_integral(integrand, idx)


    def I_s_sqrd(self, idx):

    def I_re_zf(self, idx):

    def I_re_zs(self, idx):

    def I_re_fs(self, idx):















    def pt_signal_new(self, which, old_way):
        r1 = self.radius
        r2 = self.rth
        k = self.convert_k(self.wave_probe)  # cm^-1
        zR = self.zR(wave=self.wave_probe)
        zp = self.zprobe_focus
        z = 1
        r = z
        E0 = 1
        G = k**2 * np.exp(1j * k * r) / r
        V = 4 / 3 * np.pi * self.rth**3
        V_tot = V
        Vth = V
        Epr_np = E0 / np.sqrt(1 + (-zp / zR)**2) * np.exp(-1j * k * zp)\
            * np.exp(-1j * np.arctan2(-zp, zR))
        Epr_d = E0 / (1j * z / zR) * np.exp(1j * k * (z - zp))
        dalpha_dn1, dn1_dT, dalpha_dn2, dn2_dT = self.dalphadT_atT0(
            separate=True)
        alphaT0 = self.alpha_atT0()
        B = -np.sin(1) + np.exp(1) - 2 * np.cos(1)
        C = -np.cos(1) + 2 * np.sin(1)

        if which == 'sin':
            term = B
            int_T1 = self.Pabs() * r2 /\
                (16 * np.pi * self.kappa * (2 * r1**2 + 2 * r1 * r2 + r2**2))

            int_T2 = self.Pabs() * r2**2 *\
                (np.exp(1) * (2 * r1**2 + 2 * r1 * r2
                              + r2**2) - np.exp(r1 / r2)
                 * r2 * ((3 * r1 + 2 * r2) * np.cos(1 - r1 / r2)
                         + (-r1 + r2) * np.sin(1 - r1 / r2)))\
                / (8 * np.exp(1) * (r1**2 + (r1 + r2)**2) * V * self.kappa)

            T1T2_postint = (self.Pabs()**2 * r2
                            * (np.exp(1)
                               * (-r1**3 + 2 * r1**2 * r2
                                  + 3 * r1 * r2**2 + r2**3)
                               - np.exp(r1 / r2) * r2**2
                               * ((3 * r1 + 2 * r2) * np.cos(1 - r1 / r2)
                                  + (-r1 + r2) * np.sin(1 - r1 / r2)))) /\
                (32 * np.exp(1) * np.pi * r1
                 * (2 * r1**2 + 2 * r1 * r2 + r2**2)
                 * V * self.kappa**2)

            T1quad_post_int = self.Pabs()**2 * r2 /\
                (64 * np.pi**2 * self.kappa**2 * r1
                 * (2 * r1**2 + 2 * r1 * r2 + r2**2))

            # T2quad_post_int = (((self.Pabs() * r2)**2 * (-r1 + r2)
            #                     * (r1 + r2) * (np.exp(1)
            #                                    * (2 * r1**2 + 2 * r1 * r2
            #                                       + r2**2) - np.exp(r1 / r2)
            #                                    * r2 * ((3 * r1 + 2 * r2)
            #                                            * np.cos(1 - r1 / r2)
            #                                            + (-r1 + r2)
            #                                            * np.sin(1 - r1
            #                                                     / r2)))) /\
            # (16 * np.exp(1) * (2 * r1**2 + 2 * r1 * r2 + r2**2)
            #        * V**2 * self.kappa**2)

        if which == 'cos':
            term = C
            int_T1 = self.Pabs() * r2 * (r1 + r2)\
                / (16 * np.pi * self.kappa * r1
                   * (2 * r1**2 + 2 * r1 * r2 + r2**2)
                   )

            int_T2 = np.exp(-1 + r1 / r2) * self.Pabs() * r2**3\
                * ((r1 - r2) * np.cos(1 - r1 / r2) + (3 * r1 + 2 * r2)
                   * np.sin(1 - r1 / r2)) / (8 * (r1**2 + (r1 + r2)**2)
                                             * V * self.kappa)

            T1T2_postint = (self.Pabs()**2 * r2
                            * (np.exp(1) * (-r1 + r2) * (r1 + r2)**2
                               + np.exp(r1 / r2) * r2**2
                               * ((r1 - r2) * np.cos(1 - r1 / r2)
                                  + (3 * r1 + 2 * r2)
                                  * np.sin(1 - r1 / r2))))\
                / (32 * np.exp(1) * np.pi * r1
                   * (2 * r1**2 + 2 * r1 * r2 + self.rth**2)
                   * V * self.kappa**2)

            T1quad_post_int = self.Pabs()**2 * r2 * (r1 + r2) / \
                (64 * np.pi**2 * self.kappa**2 * r1**2
                 * (2 * r1**2 + 2 * r1 * r2 + r2**2))

            T2quad_post_int = (np.exp(-1 + r1 / r2) * self.Pabs()**2
                               * r2**3 * (-r1 + r2) * (r1 + r2)
                               * ((r1 - r2) * np.cos(1 - r1 / r2)
                                   + (3 * r1 + 2 * r2)
                                  * np.sin(1 - r1 / r2))) /\
                (16 * (2 * r1**2 + 2 * r1 * r2 + r2**2)
                 * V**2 * self.kappa**2)

        ######################################################
        #  Old Way #
        ######################################################
        T_post_int = self.Pabs() * r2**2 /\
            (8 * np.exp(1) * self.kappa * V_tot) * term
        Tquad_post_int = self.Pabs()**2 * r2**4 /\
            (16 * np.exp(1) * self.kappa**2 * Vth**2) * term

        if old_way is True:
            int_alpha = (dalpha_dn1 * dn1_dT
                         + dalpha_dn2 * dn2_dT) * T_post_int
            int_dalphadT_T = (dalpha_dn1 * dn1_dT
                              + dalpha_dn2 * dn2_dT) * T_post_int
            int_dalphadT_T_sqrd = ((np.abs(dalpha_dn1 * dn1_dT)**2
                                    + np.abs(dalpha_dn2 * dn2_dT)**2)
                                   * Tquad_post_int)

        ######################################################
        #  New Way #
        ######################################################
        if old_way is False:
            zeroth = 0  # integrates to zero
            first = dalpha_dn1 * dn1_dT * int_T1
            + dalpha_dn2 * dn2_dT * int_T2
            second = 0  # need to fill this in
            I_alpha = zeroth + first + second
            int_dalphadT_T = (dalpha_dn1 * dn1_dT * int_T1
                              + dalpha_dn2 * dn2_dT * int_T2)

            int_dalphadT_T_sqrd = (np.abs(dalpha_dn1 * dn1_dT)**2
                * T1quad_post_int + np.abs(dalpha_dn2 * dn2_dT)**2
                * T2quad_post_int + 2 * np.real(dalpha_dn1 * dn1_dT * dalpha_dn2 * dn2_dT * T1T2_postint))

        I_Esca = G * I_alpha * Epr_np

# np.real(np.abs(G)**2*np.abs(E_np)**2 * (
#                         2*np.real(alphaT0*np.conj(int_dalphadT_T)) +  int_dalphadT_T_sqrd))

        I_alpha_modsqrd = 0

        Phiint = 2 * np.real(Epr_d * np.conj(I_Esca)) / np.abs(Epr_d)**2

        I_Esca_modsqrd = np.abs(G)**2 * I_alpha_modsqrd * np.abs(Epr_np)**2
        Phisca = I_Esca_modsqrd / np.abs(Epr_d)**2

        return Phiint, Phisca






















