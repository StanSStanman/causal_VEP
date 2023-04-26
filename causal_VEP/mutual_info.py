import datetime
import os.path as op
import numpy as np
import pandas as pd
import xarray as xr
import frites
from frites.dataset import DatasetEphy
from frites.estimator import (GCMIEstimator, CorrEstimator,
                              BinMIEstimator, DcorrEstimator,
                              CustomEstimator)
from meg_analysis.utils import valid_name
from meg_causal.utils import check_rejected_sessions
from causal_VEP.config.config import read_db_coords
from causal_VEP.utils import (reject_hga_trials, xr_conv,
                              z_score, relchange, lognorm)
from causal_VEP.directories import *

import matplotlib
matplotlib.use('Qt5Agg')

vep_pow_dir = op.join(*read_db_coords(), '{0}/vep/pow/{1}')


def create_dataset(subjects, sessions, regressor, condition=None, norm=None,
                   crop=None, smoothing=None, rois=None, pow_bads=False):
    """

    :param subjects: list
        List of considered subjects
    :param sessions:
        List of considered sessions
    :param regressor: str
        The behavioral regressor
    :param condition: None | str
        The conditional regressor. Default is None
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
    subs_ephy, subs_regs, subs_cond, subs_rois = [], [], [], []
    for sbj in subjects:
        _ephy, _regs, _cond = [], [], []
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
                    ephy = ephy.drop_sel({'roi': rois})
                elif type(rois) == tuple:
                    ephy = ephy.loc[{'roi': ephy.roi[rois[0]:rois[1]]}]

            # Normalizing data
            if norm is not None:
                assert isinstance(norm, str), \
                    TypeError('norm shoud be None or a string type')
                if norm == 'zscore':
                    ephy = z_score(ephy, None)
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
            if condition is not None:
                con = df.get(condition).values

            # Rejecting trials marked as bad in behavioral data
            if op.exists(op.join(prep_dir.format(sbj, ses),
                                 'trials_rejection_outcome.npz')):
                d = np.load(op.join(prep_dir.format(sbj, ses),
                                    'trials_rejection_outcome.npz'),
                            allow_pickle=True)
                bad_t = d['ar']

                if len(bad_t) != 0:
                    reg = np.delete(reg, bad_t)
                    if condition is not None:
                        con = np.delete(con, bad_t)

            # TODO test it
            # Reject trials marked as bad after power estimation, it must be
            # performed always after the first trial rejection on behavioral
            # data in order to preserve data order
            if pow_bads is True:
                bt_hga = reject_hga_trials(sbj, ses)
                ephy = ephy.drop_sel({'trials': bt_hga})
                if len(bt_hga) != 0:
                    reg = np.delete(reg, bt_hga)
                    if condition is not None:
                        con = np.delete(con, bt_hga)

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


            # # Filling sublists with data
            _ephy.append(ephy)
            _regs.append(reg)
            # Filling sublists with data [ TAKING LAST TRIAL ONLY ]
            # _ephy.append(ephy.sel({'trials': ephy.trials[-1]}))
            # _regs.append(reg[-1])
            if condition is not None:
                _cond.append(con)

        # Filling lists with data
        subs_ephy.append(xr.concat(_ephy, dim='trials'))
        subs_regs.append(np.hstack(tuple(_regs)))
        # np.random.shuffle(subs_regs[-1])       ################################################## !!!! ATTENTION, SHUFFLING !!!!! ############################################################
        if condition is not None:
            subs_cond.append(np.hstack(tuple(_cond)))
        subs_rois.append(list(ephy.roi.values))

    if condition is None:
        subs_cond = None

    for nsbj in range(len(subs_ephy)):
        # assert xr.ufuncs.isfinite(subs_ephy[nsbj]).all()
        assert xr.apply_ufunc(np.isfinite, subs_ephy[nsbj]).all()
        assert np.isfinite(subs_regs[nsbj]).all()
        if condition is not None:
            assert np.isfinite(subs_cond[nsbj]).all()

    times = ephy.times.values.round(3)
    dataset = DatasetEphy(x=subs_ephy, y=subs_regs, roi=subs_rois,
                          z=subs_cond, times=times, verbose='WARNING')

    return dataset


def create_dataset_from_dict(ss_dict, regressor, condition=None, norm=None,
                             crop=None, smoothing=None, rois=None,
                             pow_bads=False):
    """

    :param ss_dict: dict
        Dictionary in which each key is subject name and contains a
        list of sessions to consider
    :param regressor: str
        The behavioral regressor
    :param condition: None | str
        The conditional regressor. Default is None
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
    subs_ephy, subs_regs, subs_cond, subs_rois = [], [], [], []
    for sbj in ss_dict:
        _ephy, _regs, _cond = [], [], []
        # Load subject relative .xlsx file with behavioral infos
        xls = pd.read_excel(op.join(database, 'db_behaviour', project,
                                    sbj, 'regressors_2.xlsx'),
                            sheet_name=None, engine='openpyxl')

        for ses in ss_dict[sbj]:
            # Collecting elctrophysiological data
            ephy = xr.load_dataarray(op.join(db_mne, project, sbj,
                                             'vep', 'pow', str(ses),
                                             '{0}-pow.nc'.format(sbj)))
            # Cutting data in a time window
            if crop is not None:
                ephy = ephy.loc[{'times': slice(crop[0], crop[1])}]

            # Selecting regions of interest
            if rois is not None:
                if type(rois) == list:
                    ephy = ephy.loc[{'roi': rois}]
                elif type(rois) == tuple:
                    ephy = ephy.loc[{'roi': ephy.roi[rois[0]:rois[1]]}]

            if norm is not None:
                assert isinstance(norm, str), \
                    TypeError('norm shoud be None or a string type')
                if norm == 'zscore':
                    ephy = z_score(ephy)
                elif norm == 'relchange':
                    ephy = relchange(ephy)
                elif norm == 'lognorm':
                    ephy = lognorm(ephy)

            # Collecting behavioral data
            df = xls['Team {0}'.format(ses)]
            reg = df.get(regressor).values
            if condition is not None:
                con = df.get(condition).values

            # Rejecting trials marked as bad in behavioral data
            if op.exists(op.join(prep_dir.format(sbj, ses),
                                 'trials_rejection_outcome.npz')):
                d = np.load(op.join(prep_dir.format(sbj, ses),
                                    'trials_rejection_outcome.npz'),
                            allow_pickle=True)
                bad_t = d['ar']

                if len(bad_t) != 0:
                    reg = np.delete(reg, bad_t)
                    if condition is not None:
                        con = np.delete(con, bad_t)

            # TODO test it
            # Reject trials marked as bad after power estimation, it must be
            # performed always after the first trial rejection on behavioral
            # data in order to preserve data order
            if pow_bads is True:
                bt_hga = reject_hga_trials(sbj, ses)
                ephy = ephy.drop_sel({'trials': bt_hga})
                scores = np.delete(scores, bt_hga)

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
            if condition is not None:
                _cond.append(con)

        # Filling lists with data
        subs_ephy.append(xr.concat(_ephy, dim='trials'))
        subs_regs.append(np.hstack(tuple(_regs)))
        if condition is not None:
            subs_cond.append(np.hstack(tuple(_cond)))
        subs_rois.append(list(ephy.roi.values))

    if condition is None:
        subs_cond = None

    for nsbj in range(len(subs_ephy)):
        assert xr.ufuncs.isfinite(subs_ephy[nsbj]).all()
        assert np.isfinite(subs_regs[nsbj]).all()
        if condition is not None:
            assert np.isfinite(subs_cond[nsbj]).all()

    times = ephy.times.values.round(3)
    dataset = DatasetEphy(x=subs_ephy, y=subs_regs, roi=subs_rois,
                          z=subs_cond, times=times, verbose='WARNING')

    return dataset


def model_based_analysis(subjects, sessions, regressors, conditions, mi_types,
                         norm=None, crop=None, smoothing=None, rois=None,
                         pow_bads=False, inference='rfx', conjunction=True,
                         fname='default'):

    if inference == 'ffx' and conjunction is True:
        raise UserWarning("Can't perform conjunction analysis when fixed "
                          "effect is computed, conjunction set to False")
        conjunction = False

    for reg, con, mit in zip(regressors, conditions, mi_types):
        # Creating frites dataset
        if isinstance(subjects, list) and sessions is not None:
            dataset = create_dataset(subjects, sessions, reg, condition=con,
                                     norm=norm, crop=crop, smoothing=smoothing,
                                     rois=rois, pow_bads=pow_bads)
        elif isinstance(subjects, dict) and sessions is None:
            dataset = create_dataset_from_dict(subjects, reg, condition=con,
                                               norm=norm, crop=crop,
                                               smoothing=smoothing, rois=rois,
                                               pow_bads=pow_bads)
        else:
            raise ValueError

        # # Define estimator
        # est = CorrEstimator(method='spearman', implementation='tensor')
        # # Defining a frites workflow
        # workflow = frites.workflow.WfMi(mi_type=mit, inference=inference,
        #                                 estimator=est)
        # # Fitting
        # gcmi, pvals = workflow.fit(dataset, n_perm=200, tail=0, n_jobs=-1)
        ######################
        # # Define estimator
        # est = BinMIEstimator(mi_type=mit, n_bins=4)
        # # Defining a frites workflow
        # workflow = frites.workflow.WfMi(mi_type=mit, inference=inference,
        #                                 estimator=est)
        # # Fitting
        # gcmi, pvals = workflow.fit(dataset, n_perm=200, n_jobs=-1)
        ######################
        # # Define estimator
        # est = DcorrEstimator(implementation='dcor', verbose=None)
        # # Defining a frites workflow
        # workflow = frites.workflow.WfMi(mi_type=mit, inference=inference,
        #                                 estimator=est)
        # # Fitting
        # gcmi, pvals = workflow.fit(dataset, n_perm=100, n_jobs=-1)
        ######################
        # Define estimator
        est = GCMIEstimator(mi_type=mit, copnorm=True, gpu=False)
        # Defining a frites workflow
        workflow = frites.workflow.WfMi(mi_type=mit, inference=inference,
                                        estimator=est)
        # Fitting
        # gcmi, pvals = workflow.fit(dataset, n_perm=200, cluster_alpha=0.01,
        #                            n_jobs=-1)
        gcmi, pvals = workflow.fit(dataset, n_perm=1000, n_jobs=-1)
        # gcmi, pvals = workflow.fit(dataset, n_perm=200,
        #                            mcp='fdr', n_jobs=-1)
        ######################
        # est = CustomEstimator('logreg', mi_type=mit, core_fun=logreg,
        #                       multivariate=False)
        # # Defining a frites workflow
        # workflow = frites.workflow.WfMi(mi_type=mit, inference=inference,
        #                                 estimator=est)
        # # Fitting
        # gcmi, pvals = workflow.fit(dataset, n_perm=200, tail=0, n_jobs=-1)
        ######################
        # # Defining a frites workflow
        # workflow = frites.workflow.WfMi(mi_type=mit, inference=inference)
        # # Fitting
        # gcmi, pvals = workflow.fit(dataset, n_perm=100, n_jobs=-1)
        ######################
        #workflow._copnorm = False
        # Fitting
        # gcmi, pvals = workflow.fit(dataset, n_perm=100, tail=0, n_jobs=-1)
        # gcmi, pvals = workflow.fit(dataset, n_perm=100, rfx_center=False,
        #                            cluster_alpha=0.005, rfx_sigma=0.001,
        #                            n_jobs=-1)
        # gcmi, pvals = workflow.fit(dataset, n_perm=100, rfx_center=True,
        #                            n_jobs=-1)

        # Getting t-values
        if inference == 'rfx':
            tvals = workflow.get_params("tvalues")

        # Perform conjunction analysis (rfx)
        if conjunction is True:
            cjan = workflow.conjunction_analysis()
            ds = xr.Dataset({"mi": gcmi, "pv": pvals, "tv": tvals,
                             'conj_ss': cjan[0], 'conj': cjan[1]})

        elif conjunction is False and inference == 'ffx':
            ds = xr.Dataset({"mi": gcmi, "pv": pvals})

        else:
            ds = xr.Dataset({"mi": gcmi, "pv": pvals, "tv": tvals})

        # Automaticlly asigning and correct file name (get rid of symbols)
        if fname is not False and fname == 'default':
            today = datetime.date.today().strftime('%d%m%Y')
            _f = valid_name('MI_{0}.nc'.format(reg))
            _fname = op.join(database, 'stats', project, today , _f)
        elif fname is not False and isinstance(fname, str):
            _fname = fname

        if _fname is not False and isinstance(_fname, str):
            if op.exists(op.dirname(_fname)):
                ds.to_netcdf(_fname)
                print('...Saved to ', _fname)
            else:
                UserWarning("Directory in 'fname' does not exist, "
                            "creating directory...")
                os.makedirs(op.dirname(_fname))
                ds.to_netcdf(_fname)
                print('...Saved to ', _fname)

        # Save analysis info
        fname_info = _fname.replace('.nc', '_info.xlsx')
        info = {'subjects': subjects,
                'sessions': sessions,
                'regressors': [reg],
                'conditions': [con],
                'norm': [norm],
                'crop': [crop],
                'smoothing': [smoothing],
                'mi_types': [mit],
                'regions': gcmi.roi.values,
                'inference': [inference]}
        info = pd.DataFrame.from_dict(info, orient='index')
        info = info.transpose()
        info.to_excel(fname_info, sheet_name='info', index=False)
        print('...Saving info')

    return


def logreg(x, y):
    """Regression between x and y variables.

    x.shape = (n_var, 1, n_samples)
    y.shape = (n_var, 1, n_samples)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # define the regression to use
    regressor = LogisticRegression()

    # classify each point
    n_var = x.shape[0]
    regr = np.zeros((n_var,))
    for n in range(n_var):
        d = cross_val_score(regressor, x[n, :, :].T, y[n, 0, :], cv=10,
                            n_jobs=1, scoring='r2')
        regr[n] = d.mean()
    return regr


if __name__ == '__main__':
    # Electrophysiology dataset definition

    # Subjects I would like to consider
    # subjects = ['subject_02', 'subject_04', 'subject_05',
    #             'subject_06', 'subject_07', 'subject_08', 'subject_09',
    #             'subject_10', 'subject_11', 'subject_13', 'subject_14',
    #             'subject_16', 'subject_17', 'subject_18']
    # All subs
    # subjects = ['subject_01', 'subject_02', 'subject_04', 'subject_05',
    #             'subject_06', 'subject_07', 'subject_08', 'subject_09',
    #             'subject_10', 'subject_11', 'subject_13', 'subject_14',
    #             'subject_15', 'subject_16', 'subject_17', 'subject_18']
    # Subjects with R2 regression with model >= .95
    # subjects = ['subject_08', 'subject_13',
    #             'subject_16', 'subject_18']
    # Subjects with R2 regression with model >= .9
    # subjects = ['subject_01', 'subject_05',
    #             'subject_06', 'subject_07', 'subject_08',
    #             'subject_10', 'subject_13',
    #             'subject_16', 'subject_17', 'subject_18']
    subjects = ['subject_01', 'subject_02', 'subject_04', 'subject_05',
                'subject_06', 'subject_07', 'subject_08',
                'subject_10', 'subject_11', 'subject_13',
                'subject_16', 'subject_17', 'subject_18']
    # subjects = ['subject_14']


    # Positive only sessions
    # sessions = [1, 3, 5, 7, 8, 9, 11, 14, 15]
    sessions = range(1, 16)

    ss_dict = check_rejected_sessions(subjects, sessions)

    # norm = 'zscore'
    norm = None

    crop = (-.7, 1.2)

    smooth = {'blackman': 10}

    # rois = (0, -4)
    rois = ['Subcallosal-area-lh', 'Subcallosal-area-rh', 
            'Unknown-lh', 'Unknown-rh']

    # cd_reg = ['Team', 'Play', 'Win', 'Team_dp']
    # cc_reg = ['Ideal_dp',
    #           'P(W|P)_pre', 'P(W|nP)_pre', 'P(W|P)_post', 'P(W|nP)_post',
    #           'P(W|C)_pre', 'P(W|C)_post',
    #           'KL_pre', 'KL_post',
    #           'JS_pre', 'JS_post',
    #           'dP_pre', 'log_dP_pre', 'dP_post', 'log_dP_post',
    #           'S', 'BS',
    #           'dp_meta_post', 'conf_meta_post', 'info_bias_post', 'marg_surp',
    #           'empowerment']

    # cd_reg = ['Team', 'Team_dp']
    # cc_reg = ['Ideal_dp', 'dP_post', 'log_dP_post', 'S', 'BS', 'KL_post',
    #           'P(W|P)_post', 'P(W|nP)_post', 'P(W|C)_post', 'dp_meta_post',
    #           'conf_meta_post', 'info_bias_post', 'marg_surp',
    #           'empowerment']

    cd_reg = ['Team', 'Team_dp']
    cc_reg = ['Ideal_dp', 'dP_post', 'log_dP_post', 'P(W|C)_post', 'S', 'BS', 
              'KL_post', 'JS_post']

    # cd_reg = ['Team_dp']
    # cc_reg = ['Ideal_dp', 'dP_post', 'BS']
    
    cd_reg = []
    cc_reg = ['JS_post', 'dp_meta_post', 'conf_meta_post', 'info_bias_post', 
              'marg_surp', 'empowerment']
    
    cd_reg = []
    cc_reg = ['info_bias_pre']

    # cd_reg = ['Team', 'Team_dp', 'Win']
    # cc_reg = []

    # cd_reg = []
    # cc_reg = ['Ideal_dp', 'dP_post', 'BS']

    regressors = cd_reg + cc_reg

    conditions = [None] * len(regressors)
    # conditions = ['Team'] * len(regressors)

    mi_types = ['cd' for mt in range(len(cd_reg))] + \
               ['cc' for mt in range(len(cc_reg))]
    # mi_types = ['cc', 'cc'] ##################################################   REMEMBER ME!!!!     ###############################################################

    ### Toy dataset
    # subjects = ['subject_02', 'subject_11']
    # sessions = [5, 7, 11]
    # norm = 'zscore'
    # crop = (0., .25)
    # smooth = {'blackman': 40}
    # rois = (74, 82)
    # cd_reg = ['Team', 'Team_dp']
    # cc_reg = ['Ideal_dp']
    # regressors = cd_reg + cc_reg
    # conditions = [None] * len(regressors)
    # mi_types = ['cd' for mt in range(len(cd_reg))] + \
    #            ['cc' for mt in range(len(cc_reg))]

    model_based_analysis(subjects, sessions, regressors, conditions, mi_types,
                         norm=norm, crop=crop, smoothing=smooth, rois=rois,
                         pow_bads=True, inference='rfx', conjunction=True,
                         fname='default')

    # model_based_analysis(ss_dict, None, regressors, conditions, mi_types,
    #                      norm=norm, crop=crop, smoothing=smooth, rois=rois,
    #                      pow_bads=False, inference='rfx', conjunction=True,
    #                      fname='default')

    # today = datetime.date.today().strftime('%d%m%Y')
    # for sbj in subjects:
    #     for r, mit in zip(regressors, mi_types):
    #         save_fname = op.join('/media/jerry/data_drive/data/'
    #                              'stats/meg_causal/', today, sbj,
    #                              'MI_{0}.nc'.format(valid_name(r)))
    #         model_based_analysis([sbj], sessions, [r], conditions, [mit],
    #                              norm=norm, crop=crop, smoothing=smooth,
    #                              rois=rois, pow_bads=True, inference='ffx',
    #                              conjunction=False, fname=save_fname)

