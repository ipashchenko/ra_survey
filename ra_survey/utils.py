import math
import warnings
import numpy as np
try:
    from scipy.special import iv
    from scipy.integrate import quad
    is_scipy = True
except ImportError:
    warnings.warn('Install ``scipy`` python package to use Rice CDF')
    is_scipy = False

def shift(x):
    """
    Shift angles in range [-pi, pi] (like differences of angles from
    [pi/2, pi/2] interval) to range [-pi/2, pi/2] back.

    """
    return (x + np.pi) % np.pi - np.pi / 2


def gauss_2d_anisotropic(p, u, v, lambda_m=None):
    """
    :param p:
        Parameter vector:
        (log of zero-baseline flux [logJy], lg of brightness temperature [lgK],
        minor-to-major axis ratio, Major axis PA in uv-plane).
    :param u:
        Numpy array of u-coordinates.
    :param v:
        Numpy array of v-coordinates.
    :return:
        Numpy array of fluxes [Jy] of elliptical gaussian at point(s) (u, v).

    """
    k = 1.38 * 10 ** (-23)
    # From Jy to W / m ** 2 / Hz
    v0 = np.exp(p[0]) * 10 ** (-26)
    tb = 10. ** p[1]
    return 10. ** 26 * v0 * np.exp(-(np.pi * v0 * lambda_m ** 2. / (2. * k * p[2] * tb)) *
                       (u ** 2. * (np.cos(p[3]) ** 2. + (np.sin(p[3]) / p[2]) ** 2.) +
                       u * v * (1. - p[2] ** 2.) * np.sin(2. * p[3]) / p[2] ** 2.  +
                       (v ** 2. * (np.sin(p[3]) ** 2. + (np.cos(p[3]) / p[2]) ** 2.))))


def hdi_of_mcmc(sample_vec, cred_mass=0.95):
    """
    Highest density interval of sample.
    """

    assert len(sample_vec), 'need points to find HDI'
    sorted_pts = np.sort(sample_vec)

    ci_idx_inc = int(np.floor(cred_mass * len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc
    ci_width = sorted_pts[ci_idx_inc:] - sorted_pts[:n_cis]

    min_idx = np.argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx + ci_idx_inc]

    return hdi_min, hdi_max


def prepare_source_data(source_df):
    def get_with_status(source_df, status):
        flux = source_df[source_df.status == status].flux.values.astype('float')
        u = source_df[source_df.status == status].u.values.astype('float')
        v = source_df[source_df.status == status].v.values.astype('float')
        std = source_df[source_df.status == status].s_thr.values.astype('float')
        indxs_notnan = ~np.isnan(flux)
        flux = flux[indxs_notnan]
        flux_all = np.hstack((flux, flux,))
        u = u[indxs_notnan]
        u_all = np.hstack((u, -u,))
        v = v[indxs_notnan]
        v_all = np.hstack((v, -v,))
        std = std[indxs_notnan]
        std_all = np.hstack((std, std,))
        return (flux_all, np.dstack((u_all, v_all,))[0], std_all)

    detections = get_with_status(source_df, 'y')
    ulimits = get_with_status(source_df, 'n')
    undefined = get_with_status(source_df, 'u')
    undetections = list()
    undetections.append(np.hstack((ulimits[0], undefined[0],)))
    undetections.append(np.vstack((ulimits[1], undefined[1],)))
    undetections.append(np.hstack((ulimits[2], undefined[2],)))
    return detections, tuple(undetections)


def uv_to_ed(u, lambda_m=None):
    return u * lambda_m / (12742. * 1000)
