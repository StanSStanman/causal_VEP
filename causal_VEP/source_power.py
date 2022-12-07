import os
import os.path as op
import mne
import numpy as np
import xarray as xr

from causal_VEP.config.config import read_db_coords
from causal_VEP.utils import apply_artifact_rejection


def compute_source_power(subject, sbj_dir, epo_fname, fwd_fname, src_fname,
                         session, event, atlas='aparc' ,fmin=0., fmax=250.,
                         tmin=None, tmax=None, bandwidth=None,
                         win_length=0.2, tstep=0.005, norm='mean',
                         return_unlabeled=False, n_jobs=1):

    # To solve a bug in MNE,
    # if n_jobs=-1 csd_multitaper function doesn't work properly
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()

    epochs = mne.read_epochs(epo_fname, preload=True)
    epochs_ev = epochs.copy()
    epochs_ev.pick_types(meg=True)
    epochs_ev = apply_artifact_rejection(epochs_ev, subject, session,
                                         event, reject='trials')

    if tmin is None:
        tmin = epochs_ev.tmin
    if tmax is None:
        tmax = epochs_ev.tmax

    fwd = mne.read_forward_solution(fwd_fname)
    fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True,
                                               force_fixed=False,
                                               use_cps=True)
    pick_ori = 'normal'

    powers, times = get_epochs_dics(epochs_ev, fwd,
                                    fmin=fmin, fmax=fmax,
                                    tmin=tmin,
                                    tmax=tmax, tstep=tstep,
                                    win_lenghts=win_length,
                                    mt_bandwidth=bandwidth,
                                    pick_ori=pick_ori,
                                    n_jobs=n_jobs)

    # power_array = np.hstack(tuple(pow_arrays))

    # if not op.exists(
    #         op.join(db_mne, project, '{0}', 'hga').format(subject)):
    #     os.mkdir(op.join(db_mne, project, '{0}', 'hga').format(subject))
    # if not op.exists(op.join(hga_dir.format(subject, session))):
    #     os.mkdir(op.join(hga_dir.format(subject, session)))
    # np.savez_compressed(op.join(hga_dir.format(subject, session),
    #                             '{0}_power.npz'.format(event)),
    #                     power_array)
    # np.savez_compressed(op.join(hga_dir.format(subject, session),
    #                             '{0}_times.npz'.format(event)), time_array)

    power_epochs = source_2_atlas(subject=subject,
                                  sbj_dir=sbj_dir,
                                  parc=atlas,
                                  powers=powers,
                                  times=times,
                                  sources=src_fname,
                                  mode=norm)

    if return_unlabeled:
        return power_epochs, (powers, times)
    else:
        return power_epochs


def get_epochs_dics(epochs_event, fwd, fmin=0, fmax=np.inf,
                    tmin=None, tmax=None, tstep=0.005,
                    win_lenghts=0.2, mt_bandwidth=None,
                    pick_ori=None, n_jobs=1):

    if tmin is None:
        tmin = epochs_event.times[0]
    else:
        tmin = tmin

    if tmax is None:
        tmax = epochs_event.times[-1] - win_lenghts
    else:
        tmax = tmax - win_lenghts

    n_tsteps = int(
        np.round(((tmax * 1e3) - (tmin * 1e3)) / (tstep * 1e3))) + 1

    power = []
    time = np.zeros(n_tsteps)

    print('\nComputing sources power from {0}s to {1}s:'.format(tmin, tmax))
    for it in range(n_tsteps):

        win_tmin = ((tmin * 1e3) + (it * (tstep * 1e3))) / 1e3
        win_tmax = ((win_tmin * 1e3) + (win_lenghts * 1e3)) / 1e3
        time[it] = (win_tmin * 1e3 + ((win_lenghts * 1e3) / 2.)) / 1e3

        print('\nFrom {0}s to {1}s.....'.format(win_tmin, win_tmax))

        print('\tComputing average csd matrix')

        avg_csds = mne.time_frequency.csd_multitaper(epochs_event,
                                                     fmin=fmin, fmax=fmax,
                                                     tmin=win_tmin,
                                                     tmax=win_tmax,
                                                     bandwidth=mt_bandwidth,
                                                     adaptive=False,
                                                     low_bias=True,
                                                     n_jobs=n_jobs,
                                                     verbose=False)

        beamformer = mne.beamformer.make_dics(epochs_event.info, fwd,
                                              avg_csds, reg=0.05,
                                              pick_ori=pick_ori,
                                              rank=None,
                                              inversion='single',
                                              weight_norm=None,
                                              real_filter=False,
                                              reduce_rank=False,
                                              verbose=False)

        epo_power = []
        for e in range(epochs_event.__len__()):
            print('\tApplying filters for epoch {0}'.format(e))
            csds = mne.time_frequency.csd_multitaper(epochs_event[e],
                                                     fmin=fmin, fmax=fmax,
                                                     tmin=win_tmin,
                                                     tmax=win_tmax,
                                                     bandwidth=mt_bandwidth,
                                                     adaptive=False,
                                                     low_bias=True,
                                                     n_jobs=n_jobs,
                                                     verbose=False)

            power_time, _ = mne.beamformer.apply_dics_csd(csds,
                                                          beamformer,
                                                          verbose=False)

            epo_power.append(power_time)

        power.append(epo_power)
        print('...[done]')

    return power, time


def source_2_atlas(subject, sbj_dir, parc, powers, times, sources,
                   mode='mean'):

    assert(len(times) == len(powers), 'The number of time points does not '
                                      'corresponds to the lenght '
                                      'of time vector')

    labels = mne.read_labels_from_annot(subject=subject, parc=parc,
                                        hemi='both', surf_name='white',
                                        subjects_dir=sbj_dir)
    rois_names = [l.name for l in labels]

    sources = mne.read_source_spaces(sources)

    rois_timecourse = []
    print('\nComputing labels power time course (mode=', mode, '):')
    for _i, tp in enumerate(powers):
        print('\t...at time ', str(times[_i]))
        rois_single_time = mne.extract_label_time_course(tp, labels,
                                                         sources, mode=mode,
                                                         verbose=False)

        rois_timecourse.append(np.dstack(tuple(rois_single_time)))
    print('[done]')

    rois_timecourse = np.hstack(tuple(rois_timecourse))
    rois_timecourse = np.transpose(rois_timecourse, (2, 0, 1))

    rois_timecourse = xr.DataArray(rois_timecourse,
                                   coords=[range(rois_timecourse.shape[0]),
                                           rois_names, times],
                                   dims=['trials', 'roi', 'times'])

    return rois_timecourse


if __name__ == '__main__':
    db, prj = read_db_coords()
    sbj_dir = op.join(db, 'db_freesurfer/meg_causal')
    epo_dir = op.join(db, 'db_mne/meg_causal/{0}/prep/{1}')
    vep_dir = op.join(db, 'db_mne/meg_causal/{0}/vep')

    subjects = ['subject_01']
    sessions = [1]

    for sbj in subjects:
        for ses in sessions:
            if not op.exists(vep_dir.format(sbj)):
                os.makedirs(vep_dir.format(sbj))
            epo_fname = op.join(epo_dir.format(sbj, ses),
                                '{0}_outcome-epo.fif'.format(sbj))
            src_fname = op.join(vep_dir.format(sbj), '{0}-src.fif'.format(sbj))
            fwd_fname = op.join(vep_dir.format(sbj), '{0}-fwd.fif'.format(sbj))

            src_pow = compute_source_power(sbj, sbj_dir, epo_fname, fwd_fname,
                                           src_fname,
                                           ses, 'outcome', atlas='aparc.vep',
                                           fmin=88., fmax=92.,
                                           tmin=-.8, tmax=1.5,
                                           bandwidth=None, win_length=0.2,
                                           tstep=0.005, norm='mean',
                                           return_unlabeled=False, n_jobs=-1)

            pow_dir = op.join(vep_dir.format(sbj), 'pow', '{0}'.format(ses))
            if not op.exists(pow_dir):
                os.makedirs(pow_dir)
            pow_fname = op.join(pow_dir, '{0}-pow.nc'.format(sbj))

            src_pow.to_netcdf(pow_fname)
