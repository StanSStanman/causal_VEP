import numpy as np
import pandas as pd
import xarray as xr
from itertools import combinations
from scipy.spatial.distance import pdist
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

from causal_VEP.utils import (reject_hga_trials, xr_conv,
                              z_score, relchange, lognorm)
from causal_VEP.directories import *
from causal_VEP.plot_rois import plot_vep


def create_dataset(subjects, sessions, regressor, norm=None,
                   crop=None, smoothing=None, rois=None, pow_bads=False):
    """

    :param subjects: list
        List of considered subjects
    :param sessions: list
        List of considered sessions
    :param regressor: str
        The behavioral regressor
    :param norm: None | str
        The normalization to apply on the data
    :param crop: None | Tuple
        A tuple of length two will describe the starting and ending time of
        the period of analysis (in seconds)
    :param smoothing: None | dict
        Apply a convolutional smoothing on data. If a dict, it should be of
        length one. The key will describe the windowing methos, while the
        argument should be the number of time points to smooth (e.g.
        {'blackman': 40})
    :param rois: None | list | tuple
        Specify a subset of rois to take into consideration. If a list is
        provided, it should contain the names of the rois contained in the
        data to be selected, if a tuple is provided, it shoul be of length two
        and contain the min and the max range of the rois in the data
    :param pow_bads: Boolean
        If True, the trial marked as bad in the postprocessing steps after the
        power estimation will be excluded. Default is False.

    :return: dataset : frites.dataset.DatasetEphy
        Dataset compatible with the frites MI workflow
    """

    # Main loop
    subs_ephy, subs_regs = [], []
    for sbj in subjects:
        _ephy, _regs = [], []
        # Load subject relative .xlsx file with behavioral infos
        xls = pd.read_excel(op.join(database, 'db_behaviour', project,
                                    sbj, 'regressors_2.xlsx'),
                            sheet_name=None, engine='openpyxl')

        for ses in sessions:
            # Collecting elctrophysiological data
            ephy = xr.load_dataarray(op.join(db_mne, project, sbj,
                                             'vep', 'pow', str(ses),
                                             '{0}_lzs60-pow.nc'.format(sbj)))

            # Selecting regions of interest
            if rois is not None:
                if type(rois) == list:
                    ephy = ephy.loc[{'roi': rois}]
                elif type(rois) == tuple:
                    ephy = ephy.loc[{'roi': ephy.roi[rois[0]:rois[1]]}]

            # Normalizing data
            if norm is not None:
                assert isinstance(norm, str), \
                    TypeError('norm shoud be None or a string type')
                if norm == 'zscore':
                    ephy = z_score(ephy, (-.6, -.4))
                elif norm == 'relchange':
                    ephy = relchange(ephy)
                elif norm == 'lognorm':
                    ephy = lognorm(ephy)

            # Cutting data in a time window
            if crop is not None:
                ephy = ephy.loc[{'times': slice(crop[0], crop[1])}]

            # Collecting behavioral data
            df = xls['Team {0}'.format(ses)]
            reg = df.get(regressor).values
            if reg.ndim == 2:
                reg = reg.squeeze()

            # Rejecting trials marked as bad in behavioral data
            if op.exists(op.join(prep_dir.format(sbj, ses),
                                 'trials_rejection_outcome.npz')):
                d = np.load(op.join(prep_dir.format(sbj, ses),
                                    'trials_rejection_outcome.npz'),
                            allow_pickle=True)
                bad_t = d['ar']

                if len(bad_t) != 0:
                    reg = np.delete(reg, bad_t)

            # TODO test it
            # Reject trials marked as bad after power estimation, it must be
            # performed always after the first trial rejection on behavioral
            # data in order to preserve data order
            if pow_bads is True:
                bt_hga = reject_hga_trials(sbj, ses)
                ephy = ephy.drop_sel({'trials': bt_hga})
                if len(bt_hga) != 0:
                    reg = np.delete(reg, bt_hga)

            # Smoothing data
            if smoothing is not None:
                for smk in smoothing.keys():
                    if smk == 'blackman':
                        ephy = xr_conv(ephy, np.blackman(smoothing[smk]))
                    elif smk == 'barlett':
                        ephy = xr_conv(ephy, np.barlett(smoothing[smk]))
                    elif smk == 'hamming':
                        ephy = xr_conv(ephy, np.hamming(smoothing[smk]))
                    elif smk == 'hanning':
                        ephy = xr_conv(ephy, np.hanning(smoothing[smk]))


            # Filling sublists with data
            _ephy.append(ephy)
            _regs.append(reg)
            # # Filling sublists with data [ TAKING LAST TRIAL ONLY ]
            # _ephy.append(ephy.sel({'trials': ephy.trials[-1]}))
            # _regs.append(reg[-1])


        # Filling lists with data
        subs_ephy.append(xr.concat(_ephy, dim='trials'))
        subs_regs.append(np.hstack(tuple(_regs)))

    for nsbj in range(len(subs_ephy)):
        # assert xr.ufuncs.isfinite(subs_ephy[nsbj]).all()
        assert xr.apply_ufunc(np.isfinite, subs_ephy[nsbj]).all()
        assert np.isfinite(subs_regs[nsbj]).all()

    for sbj, se, sr in zip(subjects, subs_ephy, subs_regs):
        se.name = sbj
        se['trials'] = sr
        se = se.groupby('trials').mean()

    dataset = xr.merge(se.groupby('trials').mean() for se in subs_ephy)

    return dataset


def distance_corr(dataset, time_window, step=1, metric='euclidean'):
    trials = dataset.trials.data

    ds_twin = dataset.rolling(times=time_window, center=True).\
        construct('window', stride=step).dropna('times')

    couples = xr.apply_ufunc(pdist, ds_twin,
                             input_core_dims=[['trials', 'window']],
                             output_core_dims=[['couples']],
                             vectorize=True,
                             kwargs={'metric': metric})

    couples['couples'] = list(str(cp) for cp in combinations(trials, 2))

    # for k in dataset.keys():
    #     trials = dataset[k].trials.data
    #
    #     ds_twin = dataset.rolling(times=time_window, center=True). \
    #         construct('window', stride=step).dropna('times')
    #
    #     couples = xr.apply_ufunc(pdist, ds_twin,
    #                              input_core_dims=[['trials', 'window']],
    #                              output_core_dims=[['couples']],
    #                              vectorize=True,
    #                              kwargs={'metric': metric})
    #
    #     couples['couples'] = list(str(cp) for cp in combinations(trials, 2))

    # for tc in combinations(ds_twin.trials.data, 2):
    #     t1, t2 = tc

    return couples


if __name__ == '__main__':
    subjects = ['subject_08', 'subject_13',
                'subject_16', 'subject_18']
    # subjects = ['subject_01', 'subject_05',
    #             'subject_06', 'subject_07', 'subject_08',
    #             'subject_10', 'subject_13',
    #             'subject_16', 'subject_17', 'subject_18']
    subjects = ['subject_01', 'subject_02', 'subject_04', 'subject_05',
                'subject_06', 'subject_07', 'subject_08',
                'subject_10', 'subject_11', 'subject_13', 'subject_14',
                'subject_16', 'subject_17', 'subject_18']
    # subjects = ['subject_09']
    sessions = range(1, 16)
    # regressor = ['Team']
    regressor = ['Win']
    rois = (0, -2)
    norm = None
    crop = None
    smoothing = None

    ds = create_dataset(subjects, sessions, regressor, norm=norm,
                        crop=crop, smoothing=smoothing, rois=rois,
                        pow_bads=True)
    couples = distance_corr(ds, 20, step=1, metric='euclidean')

    # subset = [s for s in couples.couples.data if '15' in s]
    # plt.pcolormesh(couples.times, subset,
    #                couples['subject_18']
    #                .sel({'roi': 'Orbito-frontal-cortex-lh',
    #                      'couples': subset}).T)
    # plt.colorbar()
    #
    # plt.show()
    subset = ['(4, 14)']
    subset = ['(9, 15)']
    subset = ['(11, 15)']
    subset = ['(0, 1)']
    for sbj in subjects:
        plot_vep(couples[sbj].sel({'couples': subset}).squeeze(),
                 time=None, contrast=.01,
                 vlines={0.: dict(color='black'),
                         -.25: dict(color='black', linestyle='--')},
                 title=sbj, brain=False)

    conc = couples.to_array(dim='subjects')
    plot_vep(conc.mean('subjects').sel({'couples': subset}).squeeze(),
             pvals=None, threshold=.05, time=None, contrast=.02,
             cmap='viridis', title='Average',
             vlines={0.: dict(color='black'),
                     -.25: dict(color='black', linestyle='--')},
             brain=False)
