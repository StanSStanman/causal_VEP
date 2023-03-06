import os
import os.path as op
import numpy as np
import pandas as pd
import xarray as xr
import sklearn as sl
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed

from causal_VEP.utils import (reject_hga_trials, xr_conv,
                              z_score, relchange, lognorm)
from causal_VEP.plot_rois import plot_vep
from causal_VEP.directories import *


def create_dataset(subjects, sessions, regressor, norm=None,
                   crop=None, smoothing=None, rois=None, pow_bads=False):
    """

    :param subjects: list
        List of considered subjects
    :param sessions:
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
    subs_ephy, subs_regs, subs_rois = [], [], []
    for sbj in subjects:
        _ephy = []
        # Load subject relative .xlsx file with behavioral infos
        xls = pd.read_excel(op.join(database, 'db_behaviour', project,
                                    sbj, 'regressors_2.xlsx'),
                            sheet_name=None, engine='openpyxl')

        for ses in sessions:
            # Collecting elctrophysiological data
            ephy = xr.load_dataarray(op.join(db_mne, project, sbj,
                                             'vep', 'pow', str(ses),
                                             '{0}-pow.nc'.format(sbj)))

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

            # Add subject_n as coord
            ephy = ephy.assign_coords(sbj=('trials', [sbj]*len(ephy.trials)))

            # Replacing trials with regs values
            ephy['trials'] = reg

            # This is just a passage to be sure about time definition
            ephy['times'] = ephy.times.round(3)

            # Naming dataarray
            ephy = ephy.rename(sbj)

            # Filling sublists with data
            _ephy.append(ephy)
            # _regs.append(reg)

        # Filling lists with data
        subs_ephy.append(xr.concat(_ephy, dim='trials'))

    for nsbj in range(len(subs_ephy)):
        assert xr.ufuncs.isfinite(subs_ephy[nsbj]).all()

    dataset = xr.concat(subs_ephy, 'trials')

    return dataset


def data_modeling(dataset, estimator='QDA', n_jobs=-1):

    rois = dataset.roi.data
    times = dataset.times.data

    if estimator is 'LDA':
        model = LinearDiscriminantAnalysis
    elif estimator is 'QDA':
        model = QuadraticDiscriminantAnalysis
    elif estimator is 'DTR':
        model = DecisionTreeRegressor
    else:
        raise ValueError('Estimator not known')

    pr_vals = []
    for _r in rois:
        print(_r)
        dtst = dataset.sel({'roi': _r})
        predictions = Parallel(n_jobs=n_jobs, verbose=False)(
            delayed(estimation)
            (X=dtst.sel({'times': _t}).data.reshape(-1, 1),
             y=dtst.trials.data, model=model)
            for _t in times)
        pr_vals.append(predictions)
    # for _t in times:
    #     estimation(dataset.sel({'times': _t}).data, dataset.trials.data, model)

    pr_vals = np.stack(pr_vals)
    scores = dataset.copy().mean('trials')
    scores.data = pr_vals

    return scores


def estimation(X, y, model):
    _m = model()
    # X should be [n.trials; subjects], there should be 2 loops: 1 on regions, 1 on times
    # pred = _m.fit(X, y).predict(X)
    score = _m.fit(X, y).score(X, y)
    return score


if __name__ == '__main__':
    # Electrophysiology dataset definition
    subjects = ['subject_02', 'subject_04', 'subject_05',
                'subject_06', 'subject_07', 'subject_08', 'subject_09',
                'subject_10', 'subject_11', 'subject_13', 'subject_14',
                'subject_16', 'subject_17', 'subject_18']

    sessions = range(1, 16)

    norm = 'zscore'

    crop = (-.4, 1.)

    smooth = {'blackman': 30}

    rois = (0, -2)

    cd_reg = ['Team_dp']
    cc_reg = []

    regressors = cd_reg + cc_reg

    for _r in regressors:
        ds = create_dataset(subjects, sessions, _r,
                            norm=norm, crop=crop, smoothing=smooth, rois=rois,
                            pow_bads=True)

        scores = data_modeling(ds, estimator='LDA')

        plot_vep(scores, contrast=0.01, cmap='viridis')
