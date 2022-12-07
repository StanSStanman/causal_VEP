import os
import os.path as op

from causal_VEP.config.config import read_db_coords
from causal_VEP.bem import compute_bem
from causal_VEP.source_space import compute_source_space
from causal_VEP.forward_model import compute_forward_model
from causal_VEP.source_power import compute_source_power

if __name__ == '__main__':
    db, prj = read_db_coords()
    sbj_dir = op.join(db, 'db_freesurfer/meg_causal')
    vep_dir = op.join(db, 'db_mne/meg_causal/{0}/vep')
    raw_dir = op.join(db, 'db_mne/meg_causal/{0}/raw/1')
    trans_dir = op.join(db, 'db_mne/meg_causal/{0}/trans')
    epo_dir = op.join(db, 'db_mne/meg_causal/{0}/prep/{1}')

    subjects = ['subject_01', 'subject_02', 'subject_04', 'subject_05',
                'subject_06', 'subject_07', 'subject_08', 'subject_09',
                'subject_10', 'subject_11', 'subject_13', 'subject_14',
                'subject_15', 'subject_16', 'subject_17', 'subject_18']
    sessions = range(1, 16)

    for sbj in subjects:
        if not op.exists(vep_dir.format(sbj)):
            os.makedirs(vep_dir.format(sbj))

        # Compute BEM
        bem_fname = op.join(vep_dir.format(sbj), '{0}-bem-sol.fif'.format(sbj))
        compute_bem(sbj, sbj_dir, bem_fname)

        # Compute source space
        src_fname = op.join(vep_dir.format(sbj), '{0}-src.fif'.format(sbj))
        compute_source_space(sbj, sbj_dir, src_fname, spacing='oct6')

        # Compute forward model
        raw_fname = op.join(raw_dir.format(sbj), '{0}_raw.fif'.format(sbj))
        trans_fname = op.join(trans_dir.format(sbj),
                              '{0}-trans.fif'.format(sbj))
        src_fname = op.join(vep_dir.format(sbj), '{0}-src.fif'.format(sbj))
        bem_fname = op.join(vep_dir.format(sbj), '{0}-bem-sol.fif'.format(sbj))
        fwd_fname = op.join(vep_dir.format(sbj), '{0}-fwd.fif'.format(sbj))

        compute_forward_model(raw_fname, trans_fname, src_fname,
                              bem_fname, fwd_fname)

        # Compute source power
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
