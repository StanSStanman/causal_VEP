from causal_VEP.directories import *

import os.path as op
import numpy as np
import xarray as xr
import pandas as pd
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


def z_score(data, twin=None):
    '''
    Perform z-score on the 3rd dimension of an array
    ( y = (x - mean(x)) / std(x) )
    :param data: np.ndarray | xr.DataArray
        Data on which perform the z-score, average e standard deviation
        are computed on the 3rd dimension
    :return: np.ndarray | xr.DataArray
        z-scored data
    '''
    isinstance(twin, (tuple, type(None)))
    if twin is None:
        if isinstance(data, xr.DataArray):
            data.data = ((data.data - data.data.mean(-1, keepdims=True)) /
                         data.data.std(-1, keepdims=True))
        elif isinstance(data, np.ndarray):
            data = ((data - data.mean(-1, keepdims=True)) /
                    data.std(-1, keepdims=True))
    else:
        if isinstance(data, xr.DataArray):
            bln = data.sel({'times': slice(*twin)})
            data.data = ((data.data - bln.data.mean(-1, keepdims=True)) /
                         bln.data.std(-1, keepdims=True))
        else:
            raise ValueError('If twin is specified, data should be an '
                             'xarray.DataArray with one dim called \'times\'')
    return data


def relchange(data):
    '''
    Perform the relative change normalization on the 3rd dimension of an array
    ( y = (x - mean(x)) / mean(x) )
    :param data: np.ndarray | xr.DataArray
        Data on which compute the relative change, averages
        are computed on the 3rd dimension
    :return: np.ndarray | xr.DataArray
        relative change of data
    '''
    if isinstance(data, xr.DataArray):
        data.data = ((data.data - data.data.mean(-1, keepdims=True)) /
                     data.data.mean(-1, keepdims=True))
    elif isinstance(data, np.ndarray):
        data = ((data - data.mean(-1, keepdims=True)) /
                data.mean(-1, keepdims=True))
    return data

def lognorm(data):
    '''
    Perform the logarithmic normalization on the 3rd dimension of an array
    ( y = log(x) - log(mean(x)) )
    :param data: np.ndarray | xr.DataArray
        Data on which compute the logarithmic normalization, average
        is computed on the 3rd dimension
    :return: np.ndarray | xr.DataArray
        logarithmic normalization of data
    '''
    if isinstance(data, xr.DataArray):
        data.data = np.log(data.data)
        data.data -= data.data.mean(-1, keepdims=True)
    elif isinstance(data, np.ndarray):
        data = np.log(data)
        data -= data.mean(-1, keepdims=True)
    return data


def log_zscore(data):
    '''
      Compute the base 10 logarithm and the z-score normalization on the 3rd
      dimension of an array
      ( y = (x - mean(x)) / mean(x) )
      :param data: np.ndarray | xr.DataArray
          Data on which compute the relative change, averages
          are computed on the 3rd dimension
      :return: np.ndarray | xr.DataArray
          relative change of data
      '''
    if isinstance(data, xr.DataArray):
        data.data = np.log10(data)
        data.data = ((data.data - data.data.mean(-1, keepdims=True)) /
                     data.data.std(-1, keepdims=True))
    elif isinstance(data, np.ndarray):
        data = np.log10(data)
        data = ((data - data.mean(-1, keepdims=True)) /
                data.std(-1, keepdims=True))
    return data


def apply_baseline(data, t_win, mode):
    tmin, tmax = t_win
    bln = data.sel({'times': slice(tmin, tmax)})

    if mode == 'mean':
        data.data = data.data - bln.data.mean(-1, keepdims=True)
    elif mode == 'ratio':
        data.data = data.data / bln.data.mean(-1, keepdims=True)
    elif mode == 'relchange':
        data.data = ((data.data - bln.data.mean(-1, keepdims=True)) /
                     bln.data.mean(-1, keepdims=True))
    elif mode == 'logratio':
        data.data = np.log10(data.data / bln.data.mean(-1, keepdims=True))
    elif mode == 'zscore':
        data.data = ((data.data - bln.data.mean(-1, keepdims=True)) /
                     bln.data.std(-1, keepdims=True))
    elif mode == 'zlogratio':
        data.data = (np.log10(data.data / bln.data.mean(-1, keepdims=True)) /
                     np.log(bln.data).std(-1, keepdims=True))

    return data
