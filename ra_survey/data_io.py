import math
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


n_q = 0.637
SEFD_dict = {'RADIO-AS': {'K': {'L': 46700., 'R': 36800},
                          'C': {'L': 11600., 'R': None},
                          'L': {'L': 2760., 'R': 2930.}},
             'GBT-VLBA': {'K': {'L': 23., 'R': 23.},
                          'C': {'L': 8., 'R': 8.},
                          'L': {'L': 10., 'R': 10.}},
             'EFLSBERG': {'C': {'L': 20., 'R': 20.},
                          'L': {'L': 19., 'R': 19.}},
             'YEBES40M': {'C': {'L': 160., 'R': 160.},
                          'L': {'L': None, 'R': None}},
             'ZELENCHK': {'C': {'L': 400., 'R': 400.},
                          'L': {'L': 300., 'R': 300.}},
             'EVPTRIYA': {'C': {'L': 44., 'R': 44.},
                          'L': {'L': 44., 'R': 44.}},
             'SVETLOE': {'C': {'L': 250., 'R': 250.},
                         'L': {'L': 360., 'R': 360.}},
             'BADARY': {'C': {'L': 200., 'R': 200.},
                        'L': {'L': 330., 'R': 330.}},
             'TORUN': {'C': {'L': 220., 'R': 220.},
                       'L': {'L': 300., 'R': 300.}},
             'ARECIBO': {'C': {'L': 5., 'R': 5.},
                         'L': {'L': 3., 'R': 3.}},
             'WSTRB-07': {'C': {'L': 120., 'R': 120.},
                          'L': {'L': 40., 'R': 40.}},
             'VLA-N8': {'C': {'L': None, 'R': None},
                        'L': {'L': None, 'R': None}},
             # Default values for KL
             'KALYAZIN': {'C': {'L': 150., 'R': 150.},
                          'L': {'L': 140., 'R': 140.}},
             'MEDICINA': {'C': {'L': 170., 'R': 170.},
                          'L': {'L': 700., 'R': 700.}},
             'NOTO': {'C': {'L': 260., 'R': 260.},
                      'L': {'L': 784., 'R': 784.}},
             'HARTRAO': {'C': {'L': 650., 'R': 650.},
                         'L': {'L': 430., 'R': 430.}},
             'HOBART26': {'C': {'L': 640., 'R': 640.},
                          'L': {'L': 470., 'R': 470.}},
             'MOPRA': {'C': {'L': 350., 'R': 350.},
                       'L': {'L': 340., 'R': 340.},
                       'K': {'L': 900., 'R': 900.}},
             'WARK12M': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}},
             'TIDBIN64': {'C': {'L': None, 'R': None},
                          'L': {'L': None, 'R': None}},
             'DSS63': {'C': {'L': 24., 'R': 24.},
                       'L': {'L': 24., 'R': 24.}},
             'PARKES': {'C': {'L': 110., 'R': 110.},
                        'L': {'L': 40., 'R': 40.},
                        'K': {'L': 810., 'R': 810.}},
             'USUDA64': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}},
             'JODRELL2': {'C': {'L': 320., 'R': 320.},
                          'L': {'L': 320., 'R': 320.}},
             'ATCA104': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}}}

SEFD_panel = pd.Panel(SEFD_dict)


def get_dframe(host='odin.asc.rssi.ru', port='5432', db='ra_results',
               user='guest', password='WyxRep0Dav',
               table_name='pima_observations'):

    connection = psycopg2.connect(host=host, port=port, dbname=db,
                                  password=password, user=user)
    df = pd.io.sql.read_frame("SELECT * FROM %s;" % table_name, connection)
    return df


def add_s_thr_to_df(df, n_q=0.637, dnu=16. * 10 ** 6, n=2):
    # Adding new column
    df['s_thr'] = None

    # Looping through rows
    for i in df.index:
        try:
            sefd1 = SEFD_panel[df.ix[i, 'st1']][df.ix[i, 'band'].upper()][df.ix[i, 'polar'][0]]
        except KeyError:
            sefd1 = None
        try:
            sefd2 = SEFD_panel[df.ix[i, 'st2']][df.ix[i, 'band'].upper()][df.ix[i, 'polar'][1]]
        except KeyError:
            sefd2 = None

        try:
            df.ix[i, 's_thr'] = (1. / n_q) * math.sqrt((sefd1 * sefd2) /
                                                    (n * dnu * df.ix[i, 'solint']))
        except (ValueError, TypeError):
            pass


def get_data_for_source(source, band, df=None):

    if band not in ('k', 'c', 'l', 'p',):
        raise Exception("band must be k, c, l or p!")
    if df is None:
        df = get_dframe()
    source_df = df[df.source == source]
    # Only parallel hands
    source_df = source_df[(source_df.polar == 'RR') |
                          (source_df.polar == 'LL')]
    # Only with RA
    source_df = source_df[(source_df.st1 == 'RADIO-AS') |
                          (source_df.st2 == 'RADIO-AS')]
    # Only band
    source_df = source_df[source_df.band == band]
    # Add baseline PA angles
    source_df['angles'] = np.arctan((source_df.u / source_df.v).values)
    # Add sigma on baseline
    add_s_thr_to_df(source_df)
    # Add flux on detections and upper limits on nondetections
    source_df['flux'] = None
    # Looping through rows
    for i in source_df.index:
        if source_df.ix[i, 'status'] == 'y':
            try:
                source_df.ix[i, 'flux'] = source_df.ix[i, 's_thr'] * source_df.ix[i, 'snr']
            except TypeError:
                source_df.ix[i, 'flux'] = None
        elif source_df.ix[i, 'status'] in ('n', 'u',):
            try:
                source_df.ix[i, 'flux'] = source_df.ix[i, 's_thr'] * 5.7
            except TypeError:
                source_df.ix[i, 'flux'] = None

    return source_df


def get_antab(exper_name):
    pass


def get_fits(exper_name):
    pass


def plot_angles_histo(source, band, df=None, bins=10):
    source_df = get_data_for_source(source, band, df=df)
    plt.hist(source_df.angles.values, bins=bins)
