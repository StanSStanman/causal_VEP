from causal_VEP.directories import *

import numpy as np
import xarray as xr
import mne
from mne.time_frequency import tfr_morlet, csd_tfr
from mne.beamformer import (make_dics, make_lcmv, apply_dics_tfr_epochs,
                            apply_lcmv_epochs)
from causal_VEP.plot_rois import plot_vep
from causal_VEP.utils import apply_artifact_rejection, z_score, log_zscore

# epochs = mne.read_epochs('/media/jerry/data_drive/data/db_mne/meg_causal/subject_02/prep/1/subject_02_outcome-epo.fif')
# # epochs = epochs[:20]
# freqs = np.geomspace(50, 150, num=10)
# n_cycles = freqs / 3.
# # freqs = np.geomspace(10, 20, num=5)
# # n_cycles = freqs / 3.
# freqs = np.array([100.])
# n_cycles = 10.
#
# # epochs_tfr = tfr_morlet(epochs, freqs, n_cycles=n_cycles, return_itc=False,
# #                         output='complex', average=False, n_jobs=-1, decim=2)
# #
# # baseline_csd = csd_tfr(epochs_tfr, tmin=-.7, tmax=-.5)
# #
# # epochs_tfr.crop(tmin=-.5, tmax=1.3)
# # csd = csd_tfr(epochs_tfr)
#
# fwd = mne.read_forward_solution('/media/jerry/data_drive/data/db_mne/meg_causal/subject_02/vep/subject_02-fwd.fif')
# fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True,
#                                            force_fixed=False,
#                                            use_cps=True)
# # epochs = epochs.filter(1., 10.)
# # epochs = epochs.crop(-.5, 1.3)
#
# noise_cov = mne.make_ad_hoc_cov(epochs.info)
# covariance = mne.compute_covariance(epochs, keep_sample_mean=False,
#                                     method='empirical', cv=10, n_jobs=-1)
# filters = make_lcmv(epochs.info, fwd, data_cov=covariance, noise_cov=noise_cov,
#                     reg=0.05, pick_ori='max-power', reduce_rank=False)
# epochs_stcs = apply_lcmv_epochs(epochs, filters, return_generator=True)
#
# # filters = make_dics(epochs.info, fwd, csd, noise_csd=baseline_csd, reg=0.5,
# #                     pick_ori='normal', reduce_rank=False, real_filter=False)
# # filters = make_dics(epochs.info, fwd, csd, noise_csd=None, reg=0.5,
# #                     pick_ori='normal', reduce_rank=False, real_filter=False)
#
# # epochs_stcs = apply_dics_tfr_epochs(
# #     epochs_tfr, filters, return_generator=True)
#
# # epo_stcs = next(epochs_stcs)
#
# labels = mne.read_labels_from_annot(subject='subject_02', parc='aparc.vep',
#                                         hemi='both', surf_name='white',
#                                         subjects_dir='/media/jerry/data_drive/data/db_freesurfer/meg_causal')
# sources = mne.read_source_spaces('/media/jerry/data_drive/data/db_mne/meg_causal/subject_02/vep/subject_02-src.fif')
#
# # tc = mne.extract_label_time_course(epo_stcs, labels, sources)
#
# # tc = []
# # for stc in epochs_stcs:
# #     _tc = []
# #     for _s in stc:
# #         # _s.data = abs(_s.data)
# #         # _s.resample(sfreq=200, n_jobs=-1)
# #         _s.data = (_s.data * _s.data.conj()).real
# #         _s.resample(200, n_jobs=-1)
# #     _tc.append(mne.extract_label_time_course(_s, labels, sources, mode='mean'))
# #     _tc = np.stack(_tc, axis=-1).mean(-1)
# #     tc.append(_tc)
#
# ##################### METHOD 1 #####################
# # extract labels and then compute power
# # tc = []
# # for stc in epochs_stcs:
# #     stc = stc.resample(500., n_jobs=-1)
# #     tc.append(mne.extract_label_time_course(stc, labels, sources, mode='mean'))
# # # tc = np.stack(tc, axis=-1).mean(-1)
# # tc = np.stack(tc, axis=0)
# #
# # info = mne.create_info(ch_names=[l.name for l in labels], sfreq=500., ch_types='mag')
# # src_epo = mne.EpochsArray(data=tc, info=info, events=epochs.events,
# #                           tmin=stc.times[0], raw_sfreq=epochs.info['sfreq'])
# # # src_tfr = mne.time_frequency.tfr_morlet(src_epo, freqs, n_cycles,
# # #                                         return_itc=False, n_jobs=-1,
# # #                                         average=False, output='power')
# #
# # src_tfr = mne.time_frequency.tfr_multitaper(src_epo, freqs=[100.],
# #                                             n_cycles=[15.], time_bandwidth=15.,
# #                                             return_itc=False, n_jobs=-1,
# #                                             average=False, verbose=True)
# #
# # # src_tfr = mne.time_frequency.tfr_multitaper(src_epo, freqs=[15.],
# # #                                             n_cycles=[3.], time_bandwidth=2.,
# # #                                             return_itc=False, n_jobs=-1,
# # #                                             average=False)
# #
# # src_tfr_xr = xr.DataArray(src_tfr.data.mean(2),
# #                   coords=[range(len(src_tfr)),
# #                          src_tfr.ch_names,
# #                          src_tfr.times],
# #                   dims=['trials', 'roi', 'times']
# #                   )
# #
# # plot_vep(z_score(src_tfr_xr.mean('trials')).sel({'times': slice(-.5, 1.2)}))
# #
# #
# # tc = np.stack(tc, axis=0)
# #
# # # times = np.arange(-.5, 1.3, 0.005).round(3)
# # times = stc.times.round(3)
# #
# # tc = xr.DataArray(tc,
# #                   coords=[range(len(tc)),
# #                          [l.name for l in labels],
# #                          times],
# #                   dims=['trials', 'roi', 'times']
# #                   )
#
#
# ####################### METHOD 2 ############################
# # compute power and then extract labels
# tc = []
# for i, stc in enumerate(epochs_stcs):
#     stc = stc.resample(500., n_jobs=-1)
#
#     ch_names = list(map(str, range(stc.shape[0])))
#     info = mne.create_info(ch_names=ch_names, sfreq=500.,
#                            ch_types='mag')
#
#     stc_epo = mne.EpochsArray(data=np.expand_dims(stc.data, 0), info=info,
#                               events=np.expand_dims(epochs.events[i], 0),
#                               tmin=stc.times[0],
#                               raw_sfreq=epochs.info['sfreq'])
#
#     stc_tfr = mne.time_frequency.tfr_multitaper(stc_epo, freqs=[100.],
#                                                 n_cycles=[15.],
#                                                 time_bandwidth=15.,
#                                                 return_itc=False, n_jobs=-1,
#                                                 average=False, verbose=True)
#
#     stc.data = stc_tfr.data.squeeze()
#     # stc.times = stc_tfr.times
#
#     tc.append(mne.extract_label_time_course(stc, labels, sources, mode='mean'))
# # tc = np.stack(tc, axis=-1).mean(-1)
# tc = np.stack(tc, axis=0)
#
# info = mne.create_info(ch_names=[l.name for l in labels],
#                        sfreq=500., ch_types='mag')
# pow_epo = mne.EpochsArray(data=tc, info=info, events=epochs.events,
#                           tmin=stc_tfr.times[0],
#                           raw_sfreq=epochs.info['sfreq'])
# pow_epo = pow_epo.resample(200.)
#
# # src_tfr = mne.time_frequency.tfr_morlet(src_epo, freqs, n_cycles,
# #                                         return_itc=False, n_jobs=-1,
# #                                         average=False, output='power')
#
# # src_tfr = mne.time_frequency.tfr_multitaper(src_epo, freqs=[100.],
# #                                             n_cycles=[15.], time_bandwidth=15.,
# #                                             return_itc=False, n_jobs=-1,
# #                                             average=False, verbose=True)
#
# # src_tfr = mne.time_frequency.tfr_multitaper(src_epo, freqs=[15.],
# #                                             n_cycles=[3.], time_bandwidth=2.,
# #                                             return_itc=False, n_jobs=-1,
# #                                             average=False)
#
# src_tfr_xr = xr.DataArray(pow_epo._data,
#                   coords=[range(len(pow_epo)),
#                          pow_epo.ch_names,
#                          pow_epo.times.round(3)],
#                   dims=['trials', 'roi', 'times']
#                   )
# src_tfr_xr = src_tfr_xr.sel({'times': slice(-1.4, 1.4)})
#
# plot_vep(z_score(src_tfr_xr.mean('trials')).sel({'times': slice(-.5, 1.2)}))
#
#
# tc = np.stack(tc, axis=0)
#
# # times = np.arange(-.5, 1.3, 0.005).round(3)
# times = stc.times.round(3)
#
# tc = xr.DataArray(tc,
#                   coords=[range(len(tc)),
#                          [l.name for l in labels],
#                          times],
#                   dims=['trials', 'roi', 'times']
#                   )
#
# ##############################################################################
#
# tc = z_score(tc)
# plot_vep(tc.sel({'trials': 2}))
# plot_vep(tc.mean('trials'))
#
# # brain = epo_stcs[0].plot(
# #     subjects_dir='/media/jerry/data_drive/data/db_freesurfer/meg_causal',
# #     hemi='both',
# #     views='dorsal',
# #     brain_kwargs=dict(show=False),
# #     add_data_kwargs=dict(scale_factor=0.0001,
# #                          colorbar_kwargs=dict(label_font_size=10))
# # )

def compute_lcmv_sourcepower(subject, session, event):
    epo_fname = op.join(prep_dir.format(subject, session),
                        '{0}_{1}-epo.fif'.format(subject, event))
    epochs = mne.read_epochs(epo_fname, preload=True)
    epochs_ev = epochs.copy()
    epochs_ev.pick_types(meg=True)
    epochs_ev = apply_artifact_rejection(epochs_ev, subject, session,
                                         event, reject='trials')

    fwd_fname = op.join('/media/jerry/data_drive/data/db_mne/meg_causal',
                        '{0}/vep/{0}-fwd.fif'.format(subject))
    fwd = mne.read_forward_solution(fwd_fname)
    fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True,
                                               force_fixed=False,
                                               use_cps=True)

    noise_cov = mne.make_ad_hoc_cov(epochs_ev.info)
    covariance = mne.compute_covariance(epochs_ev, keep_sample_mean=False,
                                        method='empirical', cv=10, n_jobs=-1)
    filters = make_lcmv(epochs_ev.info, fwd, data_cov=covariance,
                        noise_cov=noise_cov, reg=0.05, pick_ori='max-power',
                        reduce_rank=False)

    epochs_stcs = apply_lcmv_epochs(epochs_ev, filters, return_generator=True)

    fs_sbj_dir = '/media/jerry/data_drive/data/db_freesurfer/meg_causal'
    labels = mne.read_labels_from_annot(subject=subject, parc='aparc.vep',
                                        hemi='both', surf_name='white',
                                        subjects_dir=fs_sbj_dir)
    sources = mne.read_source_spaces(
        op.join('/media/jerry/data_drive/data/db_mne/meg_causal',
                '{0}/vep/{0}-src.fif'.format(subject)))

    # compute power and then extract labels
    estc = []
    for i, stc in enumerate(epochs_stcs):
        # Down sampling, if not needed use: epochs_ev.info['sfreq']
        down_sfreq = 600.
        # Reduce number on timepoints
        stc = stc.resample(down_sfreq, n_jobs=-1)
        # Take sources' number as channels' name to create info
        ch_names = list(map(str, range(stc.shape[0])))
        info = mne.create_info(ch_names=ch_names, sfreq=down_sfreq,
                               ch_types='mag')

        stc_epo = mne.EpochsArray(data=np.expand_dims(stc.data, 0), info=info,
                                  events=np.expand_dims(epochs.events[i], 0),
                                  tmin=stc.times[0],
                                  raw_sfreq=epochs_ev.info['sfreq'])
        # Compute multitaper power at the single source level
        # Parameters are chosen to coincide with SEEG experience
        stc_tfr = mne.time_frequency.tfr_multitaper(stc_epo, freqs=[100.],
                                                    n_cycles=[15.],
                                                    time_bandwidth=15.,
                                                    return_itc=False,
                                                    n_jobs=-1,
                                                    average=False,
                                                    verbose=True)
        # Fill source estimate with power values
        stc.data = stc_tfr.data.squeeze()

        # Extract single epoch label timecourse
        estc.append(mne.extract_label_time_course(stc, labels, sources,
                                                  mode='mean'))
    # Stack all the epochs
    estc = np.stack(estc, axis=0)

    # Transform powers into epochs to perform operations
    info = mne.create_info(ch_names=[l.name for l in labels],
                           sfreq=down_sfreq, ch_types='mag')
    pow_epo = mne.EpochsArray(data=estc, info=info, events=epochs_ev.events,
                              tmin=stc_tfr.times[0],
                              raw_sfreq=epochs.info['sfreq'])
    # Cropping (100 ms to cut border effect) and resampling (200 Hz)
    pow_epo = pow_epo.crop(pow_epo.times[0] + .1, pow_epo.times[-1] - .1)
    pow_epo = pow_epo.resample(200.)

    # Transform data into DataArray
    pow_xr = xr.DataArray(pow_epo._data,
                          coords=[range(len(pow_epo)),
                          pow_epo.ch_names,
                          pow_epo.times.round(3)],
                          dims=['trials', 'roi', 'times'])

    return pow_xr

if __name__ == '__main__':
    # subjects = ['subject_01', 'subject_02', 'subject_04', 'subject_05',
    #             'subject_06', 'subject_07', 'subject_08', 'subject_09',
    #             'subject_10', 'subject_11', 'subject_13', 'subject_14',
    #             'subject_15', 'subject_16', 'subject_17', 'subject_18']
    subjects = ['subject_11', 'subject_13', 'subject_14',
                'subject_15', 'subject_16', 'subject_17', 'subject_18']
    sessions = range(1, 16)
    event = 'outcome'

    for sbj in subjects:
        for ses in sessions:
            pow_fname = op.join('/media/jerry/data_drive/data/db_mne',
                                'meg_causal/{0}/vep/pow'.format(sbj),
                                '{1}/{0}_lcmv-pow.nc'.format(sbj, ses))
            src_tfr_xr = compute_lcmv_sourcepower(sbj, str(ses), event)
            src_tfr_xr.to_netcdf(pow_fname)
    # plot_vep(z_score(src_tfr_xr.mean('trials')).sel({'times': slice(-.5, 1.2)}))
