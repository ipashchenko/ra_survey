import math
import numpy as np
from data_io import get_dframe, get_data_for_source
from models import Model_2d_anisotropic


# TODO: use redshifts. Tb distribution im sources rest frame should be narrow!
if __name__ == '__main__':
    # First, get the number of sources
    df = get_dframe()
    nsources = len(df.source.unique())

    # Fix parameters of group distribution
    # log(V0) ~ N(mu, sigma)
    V0_mu_G = -0.43
    V0_sigma_G = 0.94
    # lg(Tb) ~ N(mu, sigma)
    Tb_mu_G = 12.5
    Tb_sigma_G = 0.5
    # axis ration ~ Beta(alpha, beta)
    e_alpha_G = 33.
    e_beta_G = 55.

    # For each source:
    sources_df = dict()
    sources = df.source.unique()
    for source in sources:
        print "Simulating " + source
        # 1) Create each own df
        source_df = get_data_for_source(source, band='c', df=df)
        # Clear ``flux`` column
        source_df['flux'] = None
        # Clear ``status`` column
        source_df['status'] = None
        # 2) Draw parameters of current source and store them
        logV0 = np.random.normal(V0_mu_G, V0_sigma_G)
        lgTb = np.random.normal(Tb_mu_G, Tb_sigma_G)
        e = np.random.beta(e_alpha_G, e_beta_G)
        # PA of major axis in (u, v)
        pa_jet = np.random.uniform(-math.pi / 2, math.pi / 2)

        # Calculate flux of current source on all baselines and update
        # columns ``snr`` & ``status`` accordingly
        x = np.dstack((source_df.u.values, source_df.v.values))[0]
        model = Model_2d_anisotropic(x, 6.)
        fluxes = model([logV0, lgTb, e, pa_jet])
        for j, i in enumerate(source_df.index):
            source_df.ix[i, 'flux'] = fluxes[j]
            # try:
            try:
                if np.isnan(source_df.ix[i, 's_thr']):
                    print str(j) + " row has no s_thr (NaN)"
                    source_df.ix[i, 'status'] = 'x'
                    continue
            except TypeError:
                print str(j) + " row has no s_thr (None)"
                source_df.ix[i, 'status'] = 'x'
                continue

            if fluxes[j] > 5.7 * source_df.ix[i, 's_thr']:
                print str(j) + " row gives y"
                source_df.ix[i, 'status'] = 'y'
            else:
                print str(j) + " row gives n"
                source_df.ix[i, 'status'] = 'n'

        # Store current source data frame in dictionary with "true" parameters
        sources_df.update({source: [(logV0, lgTb, e, pa_jet), source_df]})
