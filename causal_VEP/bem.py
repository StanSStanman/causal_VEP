import os
import os.path as op
import mne

from causal_VEP.config.config import read_db_coords


def compute_bem(subject, sbj_dir, bem_fname):
    bem_model = mne.make_bem_model(subject=subject, subjects_dir=sbj_dir)
    bem_solution = mne.make_bem_solution(bem_model)
    mne.write_bem_solution(bem_fname, bem_solution, overwrite=True)

    return


if __name__ == '__main__':
    db, prj = read_db_coords()
    sbj_dir = op.join(db, 'db_freesurfer/meg_causal')
    vep_dir = op.join(db, 'db_mne/meg_causal/{0}/vep')

    subjects = ['subject_01']

    for sbj in subjects:
        if not op.exists(vep_dir.format(sbj)):
            os.makedirs(vep_dir.format(sbj))
        bem_fname = op.join(vep_dir.format(sbj), '{0}-bem-sol.fif'.format(sbj))
        compute_bem(sbj, sbj_dir, bem_fname)
