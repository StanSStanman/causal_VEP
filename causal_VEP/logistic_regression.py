import torch
# from torch.utils.data import TensorDataset, DataLoader
# import tensorflow as tf
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# import seaborn as sns
# import sklearn.metrics as sk_metrics

from causal_VEP.utils import (reject_hga_trials, xr_conv,
                              z_score, relchange, lognorm)
from causal_VEP.directories import *
from causal_VEP.plot_rois import plot_vep

matplotlib.use('Qt5Agg')


def create_dataset(subjects, sessions, regressor, norm=None,
                   crop=None, smoothing=None, rois=None, pow_bads=False):
    """

    :param subjects: list
        List of considered subjects
    :param sessions: list
        List of considered sessions
    :param regressor: str
        The behavioral regressor
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
    subs_ephy, subs_regs = [], []
    for sbj in subjects:
        _ephy, _regs = [], []
        # Load subject relative .xlsx file with behavioral infos
        xls = pd.read_excel(op.join(database, 'db_behaviour', project,
                                    sbj, 'regressors_2.xlsx'),
                            sheet_name=None, engine='openpyxl')

        for ses in sessions:
            # Collecting elctrophysiological data
            ephy = xr.load_dataarray(op.join(db_mne, project, sbj,
                                             'vep', 'pow', str(ses),
                                             '{0}_lcmv-pow.nc'.format(sbj)))

            # Selecting regions of interest
            if rois is not None:
                if type(rois) == list:
                    ephy = ephy.loc[{'roi': rois}]
                elif type(rois) == tuple:
                    ephy = ephy.loc[{'roi': ephy.roi[rois[0]:rois[1]]}]

            # Normalizing data
            if norm is not None:
                assert isinstance(norm, str), \
                    TypeError('norm shoud be None or a string type')
                if norm == 'zscore':
                    ephy = z_score(ephy, (-.6, -.4))
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
            if reg.ndim == 2:
                reg = reg.squeeze()

            # Rejecting trials marked as bad in behavioral data
            if op.exists(op.join(prep_dir.format(sbj, ses),
                                 'trials_rejection_outcome.npz')):
                d = np.load(op.join(prep_dir.format(sbj, ses),
                                    'trials_rejection_outcome.npz'),
                            allow_pickle=True)
                bad_t = d['ar']

                if len(bad_t) != 0:
                    reg = np.delete(reg, bad_t)

            # TODO test it
            # Reject trials marked as bad after power estimation, it must be
            # performed always after the first trial rejection on behavioral
            # data in order to preserve data order
            if pow_bads is True:
                bt_hga = reject_hga_trials(sbj, ses)
                ephy = ephy.drop_sel({'trials': bt_hga})
                if len(bt_hga) != 0:
                    reg = np.delete(reg, bt_hga)

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
            # # Filling sublists with data [ TAKING LAST TRIAL ONLY ]
            # _ephy.append(ephy.sel({'trials': ephy.trials[-1]}))
            # _regs.append(reg[-1])


        # Filling lists with data
        subs_ephy.append(xr.concat(_ephy, dim='trials'))
        subs_regs.append(np.hstack(tuple(_regs)))

    for nsbj in range(len(subs_ephy)):
        # assert xr.ufuncs.isfinite(subs_ephy[nsbj]).all()
        assert xr.apply_ufunc(np.isfinite, subs_ephy[nsbj]).all()
        assert np.isfinite(subs_regs[nsbj]).all()

    for sbj, se, sr in zip(subjects, subs_ephy, subs_regs):
        se.name = sbj
        se['trials'] = sr
        se = se.groupby('trials').mean()

    dataset = xr.concat(subs_ephy, dim='trials')

    return dataset


def sample_idx(data, perc=50):
    return np.random.choice(np.arange(len(data)),
                            int(np.round(len(data) * (perc / 100.))),
                            replace=False)


class LogisticRegression(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.built = False

    def forward(self, x, train=True):
        # Initialize the model parameters on the first call
        if not self.built:
          # Randomly generate the weights and the bias term
          rand_w = torch.rand([x.shape[-1], 1], requires_grad=True).to(device)
          rand_b = torch.rand([], requires_grad=True).to(device)
          self.w = torch.nn.Parameter(rand_w)
          self.b = torch.nn.Parameter(rand_b)
          self.built = True
        # Compute the model output
        z = torch.add(torch.matmul(x, self.w), self.b)
        if z.dim() > 1:
            z = torch.squeeze(z, dim=1)
        if train:
            return z
        return torch.sigmoid(z)


def log_loss(y_pred, y):
    # Compute the log loss function
    ce = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y)
    return ce.mean()


def predict_class(y_pred, thresh=0.5):
    # Return a tensor with  `1` if `y_pred` > `0.5`, and `0` otherwise
    return torch.where(y_pred > thresh,
                       torch.tensor([1.0]).to(device),
                       torch.tensor([0.0]).to(device))


def accuracy(y_pred, y):
    # Return the proportion of matches between `y_pred` and `y`
    y_pred = torch.sigmoid(y_pred)
    y_pred_class = predict_class(y_pred)
    check_equal = torch.eq(y_pred_class, y)
    acc_val = torch.mean(check_equal.float())
    return acc_val


def kf_logreg(chunk):
    # print(chunk.roi.data, chunk.times.data)
    # from joblib import parallel_backend
    # from threadpoolctl import threadpool_limits
    # with parallel_backend('loky', n_jobs=1): # and threadpool_limits(limits=1, user_api='blas'):
    x = chunk.data.reshape(-1, 1)
    y = chunk.trials.data
    logreg = LogisticRegression(max_iter=2000, n_jobs=1)
    # logreg = LogisticRegression(penalty='l2',
    #                            random_state=42,
    #                            C=0.2,
    #                            n_jobs=-1,
    #                            solver='sag',
    #                            multi_class='ovr',
    #                            max_iter=200,
    #                            verbose=False
    #                            )
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
    cv_score = cross_val_score(logreg, X=x, y=y, cv=cv, n_jobs=1)
    return cv_score


def kf_svc(chunk):
    # Maybe it is better to preprocess data using a scaler as described in:
    # https://scikit-learn.org/stable/modules/preprocessing.html

    from sklearn.svm import SVC
    from sklearn import preprocessing
    # print(chunk.roi.data, chunk.times.data)
    # from joblib import parallel_backend
    # from threadpoolctl import threadpool_limits
    # with parallel_backend('loky', n_jobs=1): # and threadpool_limits(limits=1, user_api='blas'):
    x = chunk.data.reshape(-1, 1)
    y = chunk.trials.data

    # preprocessing (standard scaler gaussian distribution mean = 0 & std =1)
    scaler = preprocessing.StandardScaler().fit(x)
    prep_x = scaler.transform(x)

    svc = SVC(n_jobs=1)
    # logreg = LogisticRegression(penalty='l2',
    #                            random_state=42,
    #                            C=0.2,
    #                            n_jobs=-1,
    #                            solver='sag',
    #                            multi_class='ovr',
    #                            max_iter=200,
    #                            verbose=False
    #                            )
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
    cv_score = cross_val_score(svc, X=prep_x, y=y, cv=cv, n_jobs=1)
    return cv_score


if __name__ == '__main__':
    db, prj = read_db_coords()
    vep_dir = op.join(db, 'db_mne/meg_causal/{0}/vep')

    subjects = ['subject_01', 'subject_02', 'subject_04', 'subject_05',
                'subject_06', 'subject_07', 'subject_08',
                'subject_10', 'subject_11', 'subject_13',
                'subject_16', 'subject_17', 'subject_18']
    sessions = range(1, 16)

    # subjects = ['subject_01']
    # sessions = [1]

    regressor = ['Win']
    norm = None
    crop = (-.5, 1.2)
    smoothing = {'blackman': 15}
    rois = (0, -2)

    device = 'cuda'

    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))


    # for sbj in subjects:
    #     dataset = create_dataset([sbj], sessions, regressor, norm=norm,
    #                              crop=crop, smoothing=smoothing, rois=rois,
    #                              pow_bads=True)
    #
    #     accuracies, losses = [], []
    #     for r in dataset.roi:
    #         roi_accuracy, roi_loss = [], []
    #         for tp in dataset.times:
    #             chunk = dataset.copy().sel({'roi': r, 'times': tp})
    #             idx = sample_idx(chunk, 70)
    #             x_train = chunk.copy()[idx].data
    #             x_train = np.expand_dims(x_train, -1)
    #             y_train = chunk.copy().trials[idx].data
    #             x_test = np.delete(chunk.copy().data, idx)
    #             x_test = np.expand_dims(x_test, -1)
    #             y_test = np.delete(chunk.copy().trials.data, idx)
    #
    #             batch_size = 64
    #             train_dataset = TensorDataset(torch.Tensor(x_train),
    #                                           torch.Tensor(y_train))
    #             train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                                       shuffle=True)
    #             test_dataset = TensorDataset(torch.Tensor(x_test),
    #                                          torch.Tensor(y_test))
    #             test_loader = DataLoader(test_dataset, batch_size=batch_size,
    #                                      shuffle=False)
    #
    #             # Set training parameters
    #             epochs = 201
    #             learning_rate = 0.01
    #             train_losses, test_losses = [], []
    #             train_accs, test_accs = [], []
    #
    #             # Define the model
    #             log_reg = LogisticRegression()
    #
    #             # Set up the training loop and begin training
    #             for epoch in range(epochs):
    #                 batch_losses_train, batch_accs_train = [], []
    #                 batch_losses_test, batch_accs_test = [], []
    #
    #                 # Iterate over the training data
    #                 for x_batch, y_batch in train_loader:
    #                     # Forward pass
    #                     y_pred_batch = log_reg(x_batch.to(device))
    #                     batch_loss = log_loss(y_pred_batch, y_batch.to(device))
    #                     batch_acc = accuracy(y_pred_batch, y_batch.to(device))
    #                     # Backward pass and update parameters
    #                     log_reg.zero_grad()
    #                     batch_loss.backward()
    #                     with torch.no_grad():
    #                         for p in log_reg.parameters():
    #                             p -= learning_rate * p.grad
    #                     # Keep track of batch-level training performance
    #                     batch_losses_train.append(batch_loss.item())
    #                     batch_accs_train.append(batch_acc.item())
    #
    #                 # Iterate over the testing data
    #                 for x_batch, y_batch in test_loader:
    #                     # Forward pass
    #                     y_pred_batch = log_reg(x_batch.to(device), train=True)
    #                     batch_loss = log_loss(y_pred_batch, y_batch.to(device))
    #                     batch_acc = accuracy(y_pred_batch, y_batch.to(device))
    #                     # Keep track of batch-level testing performance
    #                     batch_losses_test.append(batch_loss.item())
    #                     batch_accs_test.append(batch_acc.item())
    #
    #                 # Keep track of epoch-level model performance
    #                 train_loss, train_acc = torch.mean(
    #                     torch.tensor(batch_losses_train)), torch.mean(
    #                     torch.tensor(batch_accs_train))
    #                 test_loss, test_acc = torch.mean(
    #                     torch.tensor(batch_losses_test)), torch.mean(
    #                     torch.tensor(batch_accs_test))
    #                 train_losses.append(train_loss)
    #                 train_accs.append(train_acc)
    #                 test_losses.append(test_loss)
    #                 test_accs.append(test_acc)
    #                 if epoch % 200 == 0 and epoch != 0:
    #                     # print(
    #                     #     f"Epoch: {epoch}, Training log loss: "
    #                     #     f"{train_loss:.3f}, "
    #                     #     f"Accuracy: {train_acc}")
    #                     print(
    #                         f"Epoch: {epoch}, Test log loss: {test_loss:.3f}, "
    #                         f"Test accuracy: {test_acc}")
    #
    #             roi_accuracy.append(test_acc)
    #             roi_loss.append(test_loss)
    #
    #         accuracies.append(np.hstack(roi_accuracy))
    #         plt.plot(accuracies[-1])
    #         plt.show()
    #         losses.append(np.hstack(roi_loss))


    from joblib import Parallel, delayed
    # from joblib import parallel_backend
    from itertools import product
    from sklearn.linear_model import LogisticRegression
    # from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
    # from sklearn.metrics import balanced_accuracy_score

    # # Single session
    # for sbj in subjects:
    #     for ses in sessions:
    #         dataset = create_dataset([sbj], [ses], regressor, norm=norm,
    #                                  crop=crop, smoothing=smoothing, rois=rois,
    #                                  pow_bads=True)

    #         rtp = product(dataset.roi.values, dataset.times.values)

    list_accuracies = []
    # Jointed sessions
    for sbj in subjects:
        dataset = create_dataset([sbj], sessions, regressor, norm=norm,
                                    crop=crop, smoothing=smoothing, rois=rois,
                                    pow_bads=True)

        rtp = product(dataset.roi.values, dataset.times.values)
        #
        # accuracies, losses = [], []
        # # for r in dataset.roi:
        #
        # for _rtp in rtp:
        #     logreg = LogisticRegression(max_iter=2000, n_jobs=-1)
        #
        #     _d = dataset.copy().sel({'roi': _rtp[0], 'times': _rtp[1]})
        #     # xtrain, xtest, ytrain, ytest = train_test_split(_d,
        #     #                                                 _d.trials,
        #     #                                                 test_size=.3,
        #     #                                                 random_state=42)
        #     cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
        #     cv_score = cross_val_score(logreg,
        #                                X=_d.data.reshape(-1, 1),
        #                                y=_d.trials.data,
        #                                cv=cv,
        #                                n_jobs=-1)
        #     # logreg.fit(xtrain, ytrain)
        #     # accuracies.append(balanced_accuracy_score(ytest,
        #     #                                           logreg.predict(xtest)))
        #     accuracies.append(cv_score)
        #     print(_rtp)
        #
        # print('done')
        # import joblib
        # from dask.distributed import Client
        #
        # client = Client()
        # with parallel_backend(n_jobs=-1, backend="loky"):

        # Move the accuracy function...
        # ...here if jointed sessions
            # ...here if single session
        # accuracies = Parallel(n_jobs=15, backend='loky', verbose=1)\
        #     (delayed(kf_logreg)
        #         (dataset.copy().sel({'roi': _rtp[0], 'times': _rtp[1]}))
        #         for _rtp in rtp)
        
        # accuracies = np.stack((accuracies))
        # accuracies = np.reshape(accuracies, (144, 341, 10))
        # accuracies = xr.DataArray(accuracies,
        #                             coords=(dataset.roi,
        #                                     dataset.times,
        #                                     range(10)),
        #                             dims=['roi', 'times', 'folds'])

        dec_dir = op.join(vep_dir.format(sbj), 'dec')
        # if not op.exists(dec_dir):
        #     os.makedirs(dec_dir)
        dec_fname = op.join(dec_dir, '{0}_lcmv-dec.nc'.format(sbj))

        # accuracies.to_netcdf(dec_fname)

        accuracies = xr.load_dataarray(dec_fname)
        # list_accuracies.append(accuracies)
        # accuracies = xr_conv(accuracies, np.blackman(10))
    # accuracies = xr.concat(list_accuracies, 'folds')
        plot_vep(accuracies.mean('folds'), title=sbj,
                    contrast=.01, cmap='inferno',
                    vlines={0.: dict(color='black'),
                            -.25: dict(color='black', linestyle='--')})
