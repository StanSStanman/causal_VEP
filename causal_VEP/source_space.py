import os
import os.path as op
import mne


def compute_source_space(subject, sbj_dir, src_fname, spacing='oct6'):
    src = mne.setup_source_space(subject, spacing=spacing, add_dist=True,
                                 subjects_dir=sbj_dir, n_jobs=-1)

    src.save(src_fname, overwrite=True)

    return


if __name__ == '__main__':
    subjects = ['subject_01']
    sbj_dir = '/media/jerry/TOSHIBA_EXT/data/db_freesurfer/meg_causal'
    vep_dir = '/media/jerry/TOSHIBA_EXT/data/db_mne/meg_causal/{0}/vep'

    for sbj in subjects:
        if not op.exists(vep_dir.format(sbj)):
            os.makedirs(vep_dir.format(sbj))
        src_fname = op.join(vep_dir.format(sbj), '{0}-src.fif'.format(sbj))
        compute_source_space(sbj, sbj_dir, src_fname, spacing='oct6')
