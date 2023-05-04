'''
xrsignal - implementations of scipy.signal using xarray and dask to compute distributed
'''
from scipy import signal
import numpy as np
import xarray as xr
import dask


def __welch_chunk(da, dim, **kwargs):
    '''
    estimate PSD for single chunk of dataarray
    
    Parameters
    ----------
    da : xr.DataArray
        single chunk data
    dim : str
        dimension that psd should be estimated in
    **kwargs
        passed to scipy.signal.welch
    '''

    # Create new dimensions of PSD object
    original_dims = list(da.dims)
    psd_dim_idx = original_dims.index(dim)

    new_dims = original_dims.copy()
    new_dims[original_dims.index(dim)] = f'{dim}_frequency'
    new_dims.append(dim)

    # Estimate PSD and convert to xarray.DataArray
    f, P = signal.welch(da.values, axis=psd_dim_idx, **kwargs)
    P = np.expand_dims(P, -1)

    Px = xr.DataArray(P, dims=new_dims)

    return Px

def welch(da, dim, dB=False, **kwargs):
    '''
    Estimate power spectral density using welch method
    
    For now, an integer number of chunks in PSD dimension is required
    
    Parameters
    ----------
    da : xr.DataArray
        data array to estimate power spectral density
    dim : str
        dimension to calculate PSD over
    dB : bool
        if True, return PSD in dB
    '''

    # Get length of PSD
    if 'nperseg' in kwargs:
        nperseg = kwargs['nperseg']
        psd_len = int(nperseg/2 + 1)
    else:
        psd_len = 129  # default value for signal.welch
    if 'nfft' in kwargs:
        nfft = kwargs['nfft']
        psd_len = int(nfft/2 + 1)

    # Create new dimensions of PSD object
    original_dims = list(da.dims)
    psd_dim_idx = original_dims.index(dim)

    new_dims = original_dims.copy()
    new_dims[original_dims.index(dim)] = f'{dim}_frequency'
    new_dims.append(dim)

    # Get number of chunks in each dimension
    original_chunksize = dict(zip(original_dims, da.data.chunksize))
    nchunks = []

    for k, single_dim in enumerate(original_dims):
        nchunks.append(da.shape[k]/original_chunksize[single_dim])
    nchunks = dict(zip(original_dims, nchunks))

    # raise exception if number of chunks is not integer
    if nchunks[dim] % 1 != 0:
        raise Exception(
            f'number of chunks in dimension "{dim}" is required to be integer. there are currently {nchunks[dim]} chunks')
    # convert new dimension to integer
    nchunks[dim] = int(nchunks[dim])

    # Get size of every dimension
    original_sizes = dict(da.sizes)
    original_sizes[f'{dim}_frequency'] = psd_len

    # reorder sizes
    new_sizes = {}
    for single_dim in new_dims:
        new_sizes[single_dim] = original_sizes[single_dim]

    #new_sizes = original_sizes.copy()
    new_sizes[dim] = nchunks[dim]

    # define new chunk sizes
    new_chunk_sizes = {}
    for k, item in enumerate(new_sizes):
        if item == f'{dim}_frequency':
            new_chunk_sizes[item] = new_sizes[item]
        elif item == dim:
            new_chunk_sizes[item] = 1
        else:
            new_chunk_sizes[item] = original_chunksize[item]

    template = xr.DataArray(
        dask.array.random.random(
            list(new_sizes.values()), chunks=list(new_chunk_sizes.values())),
        dims=new_dims,
        name=f'psd across {dim} dimension')

    kwargs['dim'] = dim
    Pxx = xr.map_blocks(__welch_chunk, da, template=template,  kwargs=kwargs)
    
    if dB:
        return 10*np.log10(Pxx)
    else:
        return Pxx

def filtfilt(da, dim, b,a, **kwargs):
    '''
    distributed version for filtfilt
    
    For know, I'm going to just build this up for a dataarray and see where we get.
    filter only considers a single chunk, therfore causing errors at chunk boundaries

    TODO make this work for dataset or dataarray

    Parameters
    ----------
    da : xr.DataArray
        dataset or dataarray containing data to be filtered.
    dim : str
        dimension to filter accross
    b : array like
        numerator design coefficients for filter
    a : array like
        denominator design coefficients for filter
    kwargs : hashable
        passed to filtfitl

    Return
    ------
    ds_filt : {xr.Dataset, xr.DataArray}
        dask future array that lays out task graph for filtering data
    '''
    dim_idx = list(da.dims).index(dim)

    kwargs['b']=b
    kwargs['a']=a
    kwargs['axis']=dim_idx

    da_filt = da.map_blocks(__filtfilt_chunk, kwargs=kwargs, template=da)
    
    return da_filt

def __filtfilt_chunk(da, **kwargs):
    '''
    single chunk implementation of filt filt
    kwargs must be past and must contain, b, a, and axis
    '''
    b=kwargs['b']
    del kwargs['b']
    a=kwargs['a']
    del kwargs['a']
    axis=kwargs['axis']
    del kwargs['axis']
    da_filt = xr.DataArray(signal.filtfilt(b,a,da.values, axis=axis, **kwargs), dims=da.dims, coords=da.coords)

    return da_filt


def __hilbert_chunk(da, **kwargs):

    xc = np.abs(signal.hilbert(da.values, **kwargs))
    xcx = xr.DataArray(xc, dims=da.dims, coords=da.coords)

    return xcx


def hilbert_mag(da, dim, **kwargs):
    '''
    calculate the hilbert magnitude of da

    Parameters
    ----------
    da : xr.DataArray
    dim : str
        dimension over which to calculate hilbert transform
    '''

    dim_idx = list(da.dims).index(dim)
    kwargs['axis'] = dim_idx
    dac = da.map_blocks(__hilbert_chunk, kwargs=kwargs, template=da)

    return dac
