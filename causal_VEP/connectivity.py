
import datetime
import os
import os.path as op
# from bv2mne.directories import read_directories, read_databases
from frites.workflow import WfStats
from xfrites.conn import conn_pid
import numpy as np
# import pandas as pd
from brainets.behavior.beh_te import load_beh
# import mne
from brainets.io import load_marsatlas
import xarray as xr
from frites.utils import parallel_func
from frites.config import CONFIG

from meg_analysis.utils import valid_name
from causal_VEP.mutual_info import create_dataset
from causal_VEP.config.config import read_db_coords
database, project = read_db_coords()


# # ----------------------------------------------------------------------------------------------------------------------
# # GCMI settings
# # ----------------------------------------------------------------------------------------------------------------------
# event = 'outcome'  # {'action', 'outcome'}
# tmin, tmax = -.25, 1.5 # outcome
# # tmin, tmax = -1., 1.3  # action
# # mi settings
# mi_method = 'gc'  # {'gc', 'bin'}
# # stat settings
# inference = 'rfx'
# mcp = 'cluster'
# n_perm = 500
# cluster_alpha = 0.05  # Not used
# # regressor type
# gcmi_type = 'qlearning' # 'learning_steps' # 'conditions' #
# mi_regs =  ['bayes_surprise']  #['Pcor', 'dP' ] # ['bayes_surprise', 'rpe']  ## ['Pcor', 'dP', 'dPn', 'H'] # ['learning_step_action'] # ['absrpe', 'rpe', 'surprise', 'bayes_surprise'] # # ['learning_step_outcome'] # ['Pcor', 'dP', 'dPn', 'H'] #  ['absrpe', 'rpe', 'surprise', 'bayes_surprise', 'Ho']
# mi_types = ['cc'] #   ['cc', 'ccd']
# cnd_reg = 'outcome'  # type of conditioning for 'ccd'
# trial_types = ['all']


def conn_fsync(x, y, roi=None, times=None, mi_type='cc', gcrn=True, dt=1,
               verbose=None):
    syn = conn_pid(x, y, roi=roi, times=times, mi_type=mi_type, gcrn=gcrn, dt=dt,
                   verbose=verbose)[-2]
    return syn.data


def fsync_visuomotor(subjects, sessions, reg, mi_type, rois=None, norm=None, 
                     crop=None, smoothing=None, pow_bads=True, inference='rfx',
                     mcp='maxstat', n_perm=1000, fname='default', n_jobs=-1):

    # # ------------------------------------------------------------------------------------------------------------------
    # # Load data and directories
    # # ------------------------------------------------------------------------------------------------------------------
    # # Read json file and project directories and databases
    # if json_fname == 'default':
    #     read_dir = op.join(op.abspath(__package__), 'config')
    #     json_fname = op.join(read_dir, 'db_info.json')
    # database, project, db_mne, db_bv, db_fs, db_beh = read_databases(json_fname)
    # raw_dir, prep_dir, trans_dir, mri_dir, src_dir, bem_dir, fwd_dir, hga_dir, fc_dir, gc_dir = read_directories(json_fname)

    # # ------------------------------------------------------------------------------------------------------------------
    # # MarsAtlas ROIs
    # # ------------------------------------------------------------------------------------------------------------------
    # # Load ROIs
    # _roi = list(load_marsatlas()['LR_Name'])
    # ind_roi = np.array(range(len(_roi)))
    # roi = [_roi[val] for val in ind_roi]
    # if rm_roi:
    #     new_roi = []
    #     new_ind = []
    #     for i in roi:
    #         k = 1
    #         for j in rm_roi:
    #             if j == i: k = 0
    #         if k:
    #             new_roi += [i]
    #             new_ind += [roi.index(i)]
    #     ind_roi = ind_roi[new_ind]
    #     roi = new_roi

    dataset = create_dataset(subjects, sessions, reg, condition=None,
                             norm=norm, crop=crop, smoothing=smoothing,
                             rois=rois, pow_bads=pow_bads)

    # # Save the settings
    # cfg = pd.DataFrame(dict(regressor=mi_reg, n_perm=n_perm, roi=roi, alpha=cluster_alpha))

    # ------------------------------------------------------------------------------------------------------------------
    # Load and append behavioral from different subjects and sessions
    # ------------------------------------------------------------------------------------------------------------------
    fsync = []
    fsync_p = []
    for i, sbj in enumerate(subjects):

        # # Init
        # hga_subj = []
        # beh_subj = []
        # cnd_subj = []
        # for session in sessions:

        #     # Load good trials
        #     fname_trials = op.join(prep_dir.format(subject, session), '{0}_good_trials.xlsx'.format(subject))
        #     good_trials = load_beh(fname_trials)
        #     i = good_trials['good_trials']

        #     # Define main regressor
        #     if gcmi_type == 'qlearning':
        #         # Q-learning regressors
        #         fname_reg = op.join(prep_dir.format(subject, session), '{0}_qlearning.xlsx'.format(subject))
        #         beh_session = load_beh(fname_reg)
        #         beh_subj += [beh_session[mi_reg][i]]

        #     elif gcmi_type == 'learning_steps':
        #          # S-A-R behavioral regressors
        #          fname_reg = op.join(prep_dir.format(subject, session), '{0}_task_events.xlsx'.format(subject))
        #          beh_session = load_beh(fname_reg)
        #          beh_tmp = beh_session[mi_reg][i]
        #          beh_tmp[beh_tmp <= 7.] = 0
        #          beh_tmp[beh_tmp >= 8.] = 1
        #          beh_subj += [beh_tmp]

        #     # S-A-R behavioral regressors for conditioning
        #     fname_cnd = op.join(prep_dir.format(subject, session), '{0}_task_events.xlsx'.format(subject))
        #     cnd_session = load_beh(fname_cnd)

        #     # Append conditioning discrete regressors (only good trials not rejected from MEG preprocessing)
        #     if cnd_reg == 'learning_step_outcome':
        #         # Learning steps
        #         beh_tmp = [cnd_session[cnd_reg][i]]
        #         beh_tmp[0][beh_tmp[0] <= 6.] = 0
        #         beh_tmp[0][beh_tmp[0] >= 7.] = 1
        #         cnd_subj += [beh_tmp[0]]
        #     elif cnd_reg == 'outcome':
        #         # Behavioral events
        #         cnd_subj += [cnd_session[cnd_reg][i]]

        #     # Load HGA data
        #     fname_hga = op.join(hga_dir.format(subject, session), '{0}_hga-epo.fif'.format(event))
        #     hga_session = mne.read_epochs(fname_hga)
        #     hga_subj.append(hga_session)
        #     del hga_session

        # # HGA concatenate sessions for the same subject
        # hga_subj = mne.epochs.concatenate_epochs(hga_subj)

        # # HGA low-pass filter to reduce noise
        # hga_subj.savgol_filter(15.)

        # # Select a time window
        # hga_subj.crop(tmin=tmin, tmax=tmax)

        # # Get times
        # times = hga_subj.times

        # # Pick ROIs
        # rois = hga_subj.info["ch_names"]
        # roi_hga = [rois[val] for val in ind_roi]
        # hga_subj.pick_channels(roi_hga)

        # # Beh append subjects
        # beh_subj = np.concatenate(beh_subj)
        # cnd_subj = np.concatenate(cnd_subj)

        # ------------------------------------------------------------------------------------------------------------------
        # Compute true and permuted FSynC for each subject
        # ------------------------------------------------------------------------------------------------------------------
        # CONFIG["KW_GCMI"] = dict(shape_checking=False, biascorrect=False,
        #                          demeaned=False, mvaxis=-2, traxis=-1)

        # Init
        x = dataset.x[i]
        y = dataset.x[i].y.values

        # put optional args into a dict
        kw_conn = dict(roi=dataset.x[i].roi.values, 
                       times=dataset.x[0].times.values, 
                       mi_type=mi_type, 
                       gcrn=True, dt=1, verbose=None)

        # True FSynC
        print("Compute FSynC ")
        # It seems to me that the [-2] result of conn_pid is actually redudancy ## TODO: ask
        synergy = conn_pid(x, y, **kw_conn)[-2]

        # Remove baseline
        # synergy.data -= synergy.sel(times=slice(-0.4, -0.1)).mean('times', keepdims=True).data

        # Store
        fsync += [synergy]

        # Permuted FSynC parallel fn
        parallel, p_fun = parallel_func(conn_fsync, n_jobs=n_jobs, total=n_perm)

        # Compute permutations in parallel
        perm = parallel(p_fun(x, np.random.permutation(y), **kw_conn) 
                        for n_p in range(n_perm))
        perm = np.stack(perm, axis=0)

        # Remove baseline
        # ind = np.where((times >= -0.4) & (times<=-0.1))
        # bline = np.mean(perm[:,:,ind], axis=-1)
        # perm = perm - bline

        # Store
        fsync_p += [perm]

        # Links names (roi)
        links = synergy.roi.data

        # Number of links
        n_links = len(links)

        # del dataset, synergy, x, y

    print("Reformat true and permuted fsync")
    fsync1 = np.array_split(np.stack(fsync), n_links, axis=1)
    fsync1_p = np.array_split(np.stack(fsync_p), n_links, axis=2)
    # Squeeze
    fsync_s = [np.squeeze(val) for val in fsync1]
    fsync_p_s = [np.squeeze(val) for val in fsync1_p]
    # Reshape
    fsync_p_s = [np.swapaxes(val, 0, 1) for val in fsync_p_s]
    # print("    Compute statistics")
    wf = WfStats()
    pv, tv = wf.fit(fsync_s, fsync_p_s, inference=inference, mcp=mcp)
    # Mean FSynC across subjects
    fsync = np.mean(np.stack(fsync_s, axis=0), axis=1).T

    # ------------------------------------------------------------------------------------------------------------------
    # Save results the file to DataArray
    # ------------------------------------------------------------------------------------------------------------------

    kw_m = dict(dims=('times', 'links'), coords=(kw_conn['times'], links))
    fredc = xr.DataArray(fsync, **kw_m)
    pv = xr.DataArray(pv, **kw_m)
    tv = xr.DataArray(tv, **kw_m)
    data = xr.Dataset({'fredc': fredc, 'pv': pv, 'tv': tv})

    # Automaticlly asigning and correct file name (get rid of symbols)
    if fname is not False and fname == 'default':
        today = datetime.date.today().strftime('%d%m%Y')
        _f = valid_name('MI_{0}.nc'.format(reg))
        _fname = op.join(database, 'stats', project, 'conn', today, _f)
    elif fname is not False and isinstance(fname, str):
        _fname = fname

    if _fname is not False and isinstance(_fname, str):
        if op.exists(op.dirname(_fname)):
            data.to_netcdf(_fname)
            print('...Saved to ', _fname)
        else:
            UserWarning("Directory in 'fname' does not exist, "
                        "creating directory...")
            os.makedirs(op.dirname(_fname))
            data.to_netcdf(_fname)
            print('...Saved to ', _fname)

    return


if __name__ == '__main__':

    subjects = ['subject_01', 'subject_02', 'subject_04', 'subject_05',
                'subject_06', 'subject_07', 'subject_08',
                'subject_10', 'subject_11', 'subject_13',
                'subject_16', 'subject_17', 'subject_18']
    
    sessions = range(1, 16)

    norm = None

    crop = (-.7, 1.2)

    smooth = {'blackman': 10}

    rois = ['Subcallosal-area-lh', 'Subcallosal-area-rh', 
            'Unknown-lh', 'Unknown-rh']
    
    cd_reg = []
    cc_reg = ['Ideal_dp', 'log_dP_post', 'KL_post']

    regressors = cd_reg + cc_reg

    conditions = [None] * len(regressors)

    mi_types = ['cd' for mt in range(len(cd_reg))] + \
               ['cc' for mt in range(len(cc_reg))]
    
    pow_bads = True

    # GCMI analysises
    for reg, mit, in zip(regressors, mi_types):
        fsync_visuomotor(subjects, sessions, reg, mit, rois=rois, 
                            norm=norm, crop=crop, smoothing=smooth, 
                            pow_bads=True, inference='rfx', mcp='maxstat', 
                            n_perm=1000, fname='default', n_jobs=-1)
