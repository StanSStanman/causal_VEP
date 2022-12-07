from directories import *

import os.path as op
import numpy as np


def apply_artifact_rejection(epochs, sbj, sn, ev, reject='both'):
    if reject == 'both' or reject == 'channels':
        if op.exists(op.join(prep_dir.format(sbj, sn),
                             'channels_rejection.npz')):
            print('Loading channels rejection')
            d = np.load(op.join(prep_dir.format(sbj, sn),
                                'channels_rejection.npz'), allow_pickle=True)
            ar = d['ar']
            epochs.drop_channels(list(ar))
        else:
            print('Channels rejection not found')
    if reject == 'both' or reject == 'trials':
        if op.exists(op.join(prep_dir.format(sbj, sn),
                             'trials_rejection_{0}.npz'.format(ev))):
            d = np.load(op.join(prep_dir.format(sbj, sn),
                                'trials_rejection_{0}.npz'.format(ev)),
                        allow_pickle=True)
            ar = d['ar']
            epochs.drop(ar)
            epochs.drop_bad()
        else:
            print('Trials rejection not found')
    return epochs
