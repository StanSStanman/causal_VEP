import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path as op
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as ss
from ggseg import ggplot_vep

def plot_vep(data, pvals=None, threshold=.05, time=None, contrast=.05,
             cmap='hot_r', title=None, vlines=None, brain=False):
    '''
    Plot electrophysiological and MI data in the VEP atlas space

    :param data: xaray.DataArray
        The DataArray containing the data, it has to be 2-dimensional
        ([roi, times]).
    :param pvalues: xarray.DataArray | None
        The DataArray containing the p-values, it should be 2-dimensional
        ([roi, times]), default value is None.
    :param threshold: float
        Only clusters with p-values passing this threshold will be shown.
    :param time: tuple | None
        If time is a tuple, it should contain (tmin, tmax) to cut the data in
        the desired time window, default value is None.
    :param contrast: float | None
        Contrast to use for the plot. A contrast of .05 means that vmin is set
        to 5% of the data and vmax 95% of the data. If None, vmin and vmax are
        set to the min and max of the data. Alternatively, you can also provide
        a tuple to manually define it. Default value is None.
    :param cmap: str | None
        The matplotlib colormap to use. Default value is 'hot_r'.
    :param title: str | None
        Title of the plot, default value is None.
    :param vlines: dict | None
        Add and control vertical lines. For controlling vertical lines, use a
        dict like :

            * vlines={0.: dict(color='k'), 1.: dict(color='g', linestyle='--')}
              This draw two lines respectively at 0 and 1 secondes.

    :return: fig : plt.Figure
        A VEP atlas based representation of the data.
    '''

    # check that data is a 2D DataArray with the correct name of dims
    if isinstance(data, xr.DataArray):
        data_dims = data.coords._names
        assert ('roi' in data_dims and 'times' in data_dims), AssertionError(
            "DataArray must contain two dimensions with dims names "
            "'roi' and 'times'.")
    else:
        ValueError('data should be in xarray.DataArray format.')

    if data.dims == ('times', 'roi'):
        data = data.transpose('roi', 'times')

    # check if pvalues is None or a 2D DataArray with the correct dims
    if isinstance(pvals, xr.DataArray):
        pval_dims = pvals.coords._names
        assert ('roi' in pval_dims and 'times' in pval_dims), AssertionError(
            "DataArray must contain two dimensions with dims names "
            "'roi' and 'times'.")
        if pvals.dims == ('times', 'roi'):
            pvals = pvals.transpose('roi', 'times')

    else:
        assert pvals is None, ValueError('pvalues can be of type None or '
                                         'xarray.DataArray')

    # play with rois
    # standardizing names
    rois = []
    for _r in data.roi.values:
        if _r.startswith('Left-'):
            _r.replace('Left-', '')
            _r += '-lh'
        elif _r.startswith('Right-'):
            _r.replace('Right-', '')
            _r += '-rh'
        rois.append(_r)
    data['roi'] = rois

    # check if one or both hemispheres are considered
    lh_r, rh_r = [], []
    for _r in rois:
        if _r.endswith('-lh'):
            lh_r.append(_r)
        elif _r.endswith('-rh'):
            rh_r.append(_r)
        else:
            lh_r.append(_r)

    mode = 'single'
    if len(lh_r) != 0 and len(rh_r) != 0:
        mode = 'double'
        _lh = [_r.replace('-lh', '') for _r in lh_r]
        _rh = [_r.replace('-rh', '') for _r in rh_r]
        if _lh != _rh:
            mode = 'bordel'
            # list of rois in lh but not in rh
            lh_uniq = list(set(_lh) - set(_rh))
            # list of rois in rh but not in lh
            rh_uniq = list(set(_rh) - set(_lh))
            # add missing right regions
            for u in lh_uniq:
                _d = xr.DataArray(np.full((1, len(data.times)), np.nan),
                                  coords=[[u.replace('-lh', '-rh')],
                                          data.times],
                                  dims=['roi', 'times'])
                data = xr.concat([data, _d], 'roi')
                if pvals is not None:
                    pvals = xr.concat([pvals, _d])
            # add missing left regions
            for u in rh_uniq:
                _d = xr.DataArray(np.full((1, len(data.times)), np.nan),
                                  coords=[[u.replace('-rh', '-lh')],
                                          data.times],
                                  dims=['roi', 'times'])
                data = xr.concat([data, _d], 'roi')
                if pvals is not None:
                    pvals = xr.concat([pvals, _d])
            # sort DataArrays by rois name
            data.sortby('roi')
            if pvals is not None:
                pvals.sortby('roi')
            # reinitialize rois lists
            _lh = [_r.replace('-lh', '') for _r in data.roi
                   if _r.endswith('-lh')]
            _rh = [_r.replace('-rh', '') for _r in data.roi
                   if _r.endswith('-rh')]

    #
    ordered_labels = order_vep_labels(_lh)

    # crop time window
    if time is not None:
        data = data.sel({'times': slice(time[0], time[1])})
        if pvals is not None:
            pvals = pvals.sel({'times': slice(time[0], time[1])})

    # picking data on p-values threshold
    if pvals is not None:
        pvals = pvals.fillna(1.)
        data = xr.where(pvals >= threshold, np.nan, data)

    # get colorbar limits
    if isinstance(contrast, float):
        vmin = data.quantile(contrast, skipna=True)
        vmax = data.quantile(1 - contrast, skipna=True)
    elif isinstance(contrast, (tuple, list)) and (len(contrast) == 2):
        vmin, vmax = contrast
    else:
        vmin, vmax = data.min(skipna=True), data.max(skipna=True)
    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

    # plot specs
    if vlines is None:
        vlines = {0.: dict(color='k', linewidth=1)}
    title = '' if not isinstance(title, str) else title

    times = data.times.values
    tp = np.hstack((np.flip(np.arange(0, times.min(), -.2)),
                    np.arange(0, times.max(), .2)))
    tp = np.unique(tp.round(3))
    time_ticks = np.where(np.isin(times, tp))[0]

    # design plots
    if mode == 'single':
        h, w = len(data.roi), 9
        if brain == True:
            fig, [lbr, lh] = plt.subplots(2, 1, figsize=(w, scaling(h)),
                                          gridspec_kw={'height_ratios':
                                                       [scaling(h), h]})
            # TODO vep plot of right hemisphere
            # TODO put a small colorbar aside
            ma_brain = plot_vep_brain(data, ax=lbr)

            lh.pcolormesh(data.times, data.roi, data)
        else:
            fig, lh = plt.subplots(1, 1, figsize=(w, scaling(h)))

    elif mode == 'double' or mode == 'bordel':
        h, w = len(data.roi), 14
        if brain == True:
            fig, [lbr, lh, rbr, rh] = \
                plt.subplots(2, 2, figsize=(w, scaling(h)), gridspec_kw={
                    'height_ratios': [scaling(h), h, scaling(h), h]},
                             sharey=True)

            # TODO vep plot of right hemisphere
            # TODO put a small colorbar aside
            ma_brain = plot_vep_brain(data, ax=rbr)

        else:
            fig, [lh, rh] = plt.subplots(1, 2, figsize=(w, scaling(h)))
            #fig, [lh, rh] = plt.subplots(1, 2, figsize=(14, 20))

        _data = data.sel({'roi': lh_r})
        _data['roi'] = _lh
        _data = _data.sel({'roi': ordered_labels['label']})

        if mode is 'single':
            ss.heatmap(_data.to_pandas(), yticklabels=True, xticklabels=False,
                       vmin=vmin.values, vmax=vmax.values, cmap=cmap, ax=lh,
                       zorder=0)

            for k, kw in vlines.items():
                _k = np.where(data.times.values == k)[0][0]
                lh.axvline(_k, **kw)

            lh.set_xticks(time_ticks)
            lh.set_xticklabels(tp, rotation='horizontal')
            lh.tick_params(axis='y', which='major', labelsize=10)
            lh.tick_params(axis='y', which='minor', labelsize=10)
            lh.yaxis.set_label_text('')
            plt.tight_layout()

        elif mode == 'double' or mode == 'bordel':
            ss.heatmap(_data.to_pandas(), yticklabels=True, xticklabels=False,
                       vmin=vmin.values, vmax=vmax.values, cmap=cmap, ax=lh,
                       cbar=False, zorder=0)

            for k, kw in vlines.items():
                _k = np.where(data.times.values == k)[0][0]
                lh.axvline(_k, **kw)

            lh.set_xticks(time_ticks)
            lh.set_xticklabels(tp, rotation='horizontal')

            ylabs = [item.get_text() for item in lh.get_yticklabels()]
            lh.set_yticklabels(['' for yl in ylabs])
            lh.tick_params(axis='y', bottom=True, top=False, left=False,
                           right=True, direction="out", length=3, width=1.5)
            lh.yaxis.set_label_text('')

            _data = data.sel({'roi': rh_r})
            _data['roi'] = _rh
            _data = _data.sel({'roi': ordered_labels['label']})

            ss.heatmap(_data.to_pandas(), yticklabels=True, xticklabels=False,
                       vmin=vmin.values, vmax=vmax.values, cmap=cmap, ax=rh,
                       cbar=False, zorder=0)

            for k, kw in vlines.items():
                _k = np.where(data.times.values == k)[0][0]
                rh.axvline(_k, **kw)

            rh.set_xticks(time_ticks)
            rh.set_xticklabels(tp, rotation='horizontal')
            rh.set_yticklabels(_data.roi.values, ha='center',
                               position=(-.27, 0))
            rh.tick_params(axis='y', which='major', labelsize=9)
            rh.tick_params(axis='y', which='minor', labelsize=9)
            rh.tick_params(axis='y', bottom=True, top=False, left=True,
                           right=False, direction="out", length=3, width=1.5)
            rh.yaxis.set_label_text('')

            for ytl, col in zip(rh.get_yticklabels(), ordered_labels['color']):
                ytl.set_color(col)

            cbar = fig.add_axes([.3, .05, .4, .015])
            norm = mpl.colors.Normalize(vmin=kwargs['vmin'],
                                        vmax=kwargs['vmax'])
            cb_cmap = mpl.cm.get_cmap(kwargs['cmap'])
            mpl.colorbar.ColorbarBase(cbar, cmap=cb_cmap, norm=norm,
                                      orientation='horizontal')
            cbar.tick_params(labelsize=10)

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.1)

    plt.show()

    return


    # TODO list
    # add single/double plotting functions


def scaling(x):
    a = 13 / 71
    b = 3 - a
    y = (a * x) + b
    return y


def plot_vep_brain(data, thld=.05, ax=None):
    midd = {x: data.loc[{'roi': x}].mean('times', skipna=True).values
            for x in data.roi.values}
    midd = {x: midd[x] for x in midd if not np.isnan(midd[x])}

    vep_plt = ggplot_vep(midd, ax=ax)
    return vep_plt


def order_vep_labels(labels):
    vep_fname = op.join(op.dirname(__file__), 'vep.xlsx')
    df = pd.read_excel(vep_fname, index_col=None, usecols=range(4),
                       engine='openpyxl')
    ord_labels = dict(label=[], lobe=[], color=[])
    for _, l in df.iterrows():
        if l['roi'] in labels:
            ord_labels['label'].append(l['roi'])
            ord_labels['lobe'].append(l['lobes'])
            ord_labels['color'].append(l['color'])

    return ord_labels


if __name__ == '__main__':
    # from utils import z_score, relchange, lognorm, xr_conv
    # subjects = ['subject_02', 'subject_04', 'subject_05',
    #             'subject_06', 'subject_07', 'subject_08', 'subject_09',
    #             'subject_10', 'subject_11', 'subject_13', 'subject_14',
    #             'subject_16', 'subject_17', 'subject_18']
    #
    # # subjects = ['subject_01', 'subject_02', 'subject_04', 'subject_05',
    # #             'subject_06', 'subject_07', 'subject_08', 'subject_09',
    # #             'subject_10', 'subject_11', 'subject_13', 'subject_14',
    # #             'subject_15', 'subject_16', 'subject_17', 'subject_18']
    #
    # sessions = range(1, 16)
    #
    # fname = '/media/jerry/data_drive/data/db_mne/meg_causal/' \
    #         '{0}/vep/pow/{1}/{0}-pow.nc'
    #
    # datas, n = [], 0
    # for sbj in subjects:
    #     for ses in sessions:
    #         data = xr.load_dataarray(fname.format(sbj, ses))
    #         data = data.drop_sel({'roi': ['Unknown-lh', 'Unknown-rh']})
    #         data = z_score(data)
    #         data = xr_conv(data, np.blackman(30))
    #         data = data.mean('trials')
    #         if n == 0:
    #             datas = data.data
    #         else:
    #             datas += data.data
    #         n += 1
    #
    # datas /= n
    # data.data = datas
    #
    # plot_vep(data, pvals=None, threshold=.05, time=None, contrast=.02,
    #          cmap='Spectral_r', title='HGA',
    #          vlines={0.: dict(color='black'),
    #                  -.3: dict(color='black', linestyle='--')},
    #          brain=False)

    from meg_analysis.utils import valid_name
    cd_reg = ['Team', 'Play', 'Win', 'Team_dp']
    cc_reg = ['Ideal_dp',
              'P(W|P)_pre', 'P(W|nP)_pre', 'P(W|P)_post', 'P(W|nP)_post',
              'P(W|C)_pre', 'P(W|C)_post',
              'KL_pre', 'KL_post',
              'JS_pre', 'JS_post',
              'dP_pre', 'log_dP_pre', 'dP_post', 'log_dP_post',
              'S', 'BS',
              'dp_meta_post', 'conf_meta_post', 'info_bias_post', 'marg_surp',
              'empowerment']

    cd_reg = ['Team', 'Team_dp']
    cc_reg = ['Ideal_dp', 'KL_post', 'dP_post', 'log_dP_post', 'S', 'BS']

    reg = cd_reg + cc_reg

    for r in reg:
        _r = valid_name(r)
        fname = '/media/jerry/data_drive/data/stats/meg_causal/' \
                '23022023/MI_{0}.nc'.format(_r)
        dataar = xr.load_dataset(fname)
        data = dataar.mi
        pvals = dataar.pv

        print(_r)
        plot_vep(data, pvals=None, threshold=.05, time=None, contrast=.02,
                 cmap='viridis', title='HGA',
                 vlines={0.: dict(color='black'),
                         -.3: dict(color='black', linestyle='--')},
                 brain=False)

        plot_vep(pvals, pvals=None, threshold=.05, time=None, contrast=.02,
                 cmap='gist_stern', title='HGA',
                 vlines={0.: dict(color='black'),
                         -.3: dict(color='black', linestyle='--')},
                 brain=False)

        plot_vep(data, pvals=pvals, threshold=.05, time=None, contrast=.02,
                 cmap='viridis', title='HGA',
                 vlines={0.: dict(color='black'),
                         -.3: dict(color='black', linestyle='--')},
                 brain=False)
