from directories import *

import os.path as op
import numpy as np
import xarray as xr
from scipy.signal import convolve


def apply_artifact_rejection(epochs, sbj, sn, ev, reject='both'):
    if reject == 'both' or reject == 'channels':
        if op.exists(op.join(prep_dir.format(sbj, sn),
                             'channels_rejection.npz')):
            print('Loading channels rejection')
            d = np.load(op.join(prep_dir.format(sbj, sn),
                                'channels_rejection.npz'), allow_pickle=True)
            ar = d['ar']
            epochs.drop_channels(list(ar))
        else:
            print('Channels rejection not found')
    if reject == 'both' or reject == 'trials':
        if op.exists(op.join(prep_dir.format(sbj, sn),
                             'trials_rejection_{0}.npz'.format(ev))):
            d = np.load(op.join(prep_dir.format(sbj, sn),
                                'trials_rejection_{0}.npz'.format(ev)),
                        allow_pickle=True)
            ar = d['ar']
            epochs.drop(ar)
            epochs.drop_bad()
        else:
            print('Trials rejection not found')
    return epochs


def reject_hga_trials(subject, session):
    xls = pd.read_excel(op.join(db_mne, project, 'bad_hga.xlsx'),
                        index_col=0, engine='openpyxl')
    bad_tr = xls.loc[subject][session]

    if isinstance(bad_tr, float):
        bad_tr = []
    elif isinstance(bad_tr, str):
        bad_tr = bad_tr.split(',')
        bad_tr = [int(t) for t in bad_tr]
    elif isinstance(bad_tr, int):
        bad_tr = [bad_tr]

    # Transform list in array and subtract positional 1
    bad_tr = np.array(bad_tr) - 1
    return bad_tr


def xr_conv(data, kernel):
    kernel = np.expand_dims(kernel, (0, 1))
    func = lambda x: convolve(x, kernel, mode="same", method="fft")
    return xr.apply_ufunc(func, data)


def z_score(data):
    '''
    Perform z-score on the 3rd dimension of an array
    :param data: np.ndarray | xr.DataArray
        Data on which perform the z-score, average e standard deviation
        are computed on the 3rd dimension
    :return: np.ndarray | xr.DataArray
        z-scored data
    '''
    if isinstance(data, xr.DataArray):
        data.data -= data.data.mean(-1, keepdims=True)
        data.data /= data.data.std(-1, keepdims=True)
    elif isinstance(data, np.ndarray):
        data -= data.mean(-1, keepdims=True)
        data /= data.std(-1, keepdims=True)
    return data
