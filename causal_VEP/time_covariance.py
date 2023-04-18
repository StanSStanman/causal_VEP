from causal_VEP.config.config import read_db_coords
import os
import os.path as op
import xarray as xr
from joblib import Parallel, delayed
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


def compute_time_cov(data, measure='cov', n_jobs=1):

    if measure == 'cov':
        mfunc = xr.cov
    elif measure == 'corr':
        mfunc = xr.corr
    else:
        raise ValueError('Estimation method not implemented')

    # Fastest reliable method, but makes RAM explode
    # cov = mfunc(data, data.rename({'times': 'times2'}), dim='trials')

    # 'Slower' method, reliable, prevents RAM overload
    cov = Parallel(n_jobs=n_jobs, backend='loky', verbose=1)\
        (delayed(mfunc)
         (data.sel({'roi': r}),
          data.sel({'roi': r}).rename({'times': 'times2'}),
          dim='trials')
         for r in data.roi)

    # cov = []
    # for r in a.roi:
    #     b = a.copy().sel({'roi': r})
    #     cov.append(xr.cov(b, b.rename({'times': 'times2'}), dim='trials'))
    cov = xr.concat(cov, dim='roi')
    return cov


def plot_cov(data, roi=None, trans='avg'):
    assert trans in ['avg', 'max', 'min', 'var', 'std', None]

    if roi is not None:
        assert isinstance(roi, list)
        data = data.sel({'roi': roi})

    if trans == 'avg':
        data = data.mean('roi', keepdims=True)
    elif trans == 'max':
        data = data.max('roi', keepdims=True)
    elif trans == 'min':
        data = data.min('roi', keepdims=True)
    elif trans == 'var':
        data = data.var('roi', keepdims=True)
    elif trans == 'std':
        data = data.std('roi', keepdims=True)
    else:
        data = data


    for r in data.roi:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        im = ax.pcolormesh(data.times, data.times2, data.sel({'roi': r}).data,
                           cmap='Spectral_r')
        ax.axvline(0., color='w', lw=.8)
        ax.axhline(0., color='w', lw=.8)
        plt.colorbar(im)
        plt.show()

    return


@jax.jit
def compute_cov_gpu(x, y):
    # Compute the covariance of two inputs
    return jnp.dot(x - jnp.mean(x), y - jnp.mean(y)) / (len(x) - 1)


@jax.jit
def compute_time_cov_gpu(data):

    # cov = []
    # for r in data.roi:
    #     cov.append(compute_cov_gpu(data.sel({'roi': r}).data,
    #                                data.sel({'roi': r}).data.T))
    cov = jax.vmap(lambda r: compute_cov_gpu(data.sel({'roi': r}).data,
                                             data.sel({'roi': r}).data.T))\
            (data.roi)


    # Convert to JAX array and concatenate along 'roi' dimension
    # cov = jnp.asarray(cov.values)
    cov = jnp.concatenate(jnp.split(cov, cov.shape[0], axis=0), axis=1)

    return xr.DataArray(cov, dims=('roi', 'times', 'times2'))


if __name__ == '__main__':
    db, prj = read_db_coords()
    vep_dir = op.join(db, 'db_mne/meg_causal/{0}/vep')

    subjects = ['subject_01', 'subject_02', 'subject_04', 'subject_05',
                'subject_06', 'subject_07', 'subject_08', 'subject_09',
                'subject_10', 'subject_11', 'subject_13', 'subject_14',
                'subject_15', 'subject_16', 'subject_17', 'subject_18']
    sessions = range(1, 16)

    subjects = ['subject_04']
    sessions = [10, 11]

    for sbj in subjects:
        data = []
        for ses in sessions:
            pow_dir = op.join(vep_dir.format(sbj), 'pow', '{0}'.format(ses))
            pow_fname = op.join(pow_dir, '{0}_lzs60-pow.nc'.format(sbj))
            data = xr.load_dataarray(pow_fname)
            # data.append(xr.load_dataarray(pow_fname).sel({'roi': ['Precentral-gyrus-upper-limb-lh',
            #                'Precentral-gyrus-upper-limb-rh']}))
            data = xr.concat(data, dim='trials')

            covariance = compute_time_cov(data, 'corr', n_jobs=-1)
            # covariance = compute_time_cov_gpu(data)

            cov_dir = op.join(vep_dir.format(sbj), 'cov', '{0}'.format(ses))
            # if not op.exists(cov_dir):
            #     os.makedirs(cov_dir)
            cov_fname = op.join(cov_dir, '{0}_lzs60-cov.nc'.format(sbj))
            #
            # covariance.to_netcdf(cov_fname)

            # covariance = xr.load_dataarray(cov_fname)
            print(sbj, ',', ses)
            trans = 'max'
            plot_cov(covariance,
                     roi=None, trans=trans)
            # plot_cov(covariance,
            #          roi=['Frontal-pole-lh', 'Frontal-pole-rh'], trans=trans)
            # plot_cov(covariance,
            #          roi=['Orbito-frontal-cortex-lh',
            #               'Orbito-frontal-cortex-rh'], trans=trans)
            # plot_cov(covariance,
            #          roi=['Precentral-gyrus-upper-limb-lh',
            #               'Precentral-gyrus-upper-limb-rh'], trans=trans)
            # plot_cov(covariance,
            #          roi=['O1-lh', 'O1-rh'], trans=trans)
            # plot_cov(covariance,
            #          roi=['O1-lh', 'O1-rh',
            #               'O2-lh', 'O2-rh',
            #               'Occipital-pole-lh', 'Occipital-pole-rh',
            #               'Lingual-gyrus-lh', 'Lingual-gyrus-rh',
            #               'Calcarine-sulcus-lh', 'Calcarine-sulcus-rh',
            #               'Cuneus-lh', 'Cuneus-rh'], trans=trans)

