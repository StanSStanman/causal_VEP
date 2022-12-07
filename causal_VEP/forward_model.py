import os
import os.path as op
import mne

from causal_VEP.config.config import read_db_coords


def compute_forward_model(raw_fname, trans_fname, src_fname,
                          bem_fname, fwd_fname):

    raw = mne.io.read_raw_fif(raw_fname)
    trans = mne.read_trans(trans_fname)
    src = mne.read_source_spaces(src_fname)
    bem = mne.read_bem_solution(bem_fname)

    fwd = mne.make_forward_solution(info=raw.info, trans=trans,
                                    src=src, bem=bem,
                                    meg=True, eeg=False,
                                    mindist=5.0, n_jobs=-1, verbose=False)

    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

    return


if __name__ == '__main__':
    db, prj = read_db_coords()
    raw_dir = op.join(db, 'db_mne/meg_causal/{0}/raw/1')
    trans_dir = op.join(db, 'db_mne/meg_causal/{0}/trans')
    vep_dir = op.join(db, 'db_mne/meg_causal/{0}/vep')

    subjects = ['subject_01']

    for sbj in subjects:
        if not op.exists(vep_dir.format(sbj)):
            os.makedirs(vep_dir.format(sbj))
        raw_fname = op.join(raw_dir.format(sbj), '{0}_raw.fif'.format(sbj))
        trans_fname = op.join(trans_dir.format(sbj),
                              '{0}-trans.fif'.format(sbj))
        src_fname = op.join(vep_dir.format(sbj), '{0}-src.fif'.format(sbj))
        bem_fname = op.join(vep_dir.format(sbj), '{0}-bem-sol.fif'.format(sbj))
        fwd_fname = op.join(vep_dir.format(sbj), '{0}-fwd.fif'.format(sbj))

        compute_forward_model(raw_fname, trans_fname, src_fname,
                              bem_fname, fwd_fname)
