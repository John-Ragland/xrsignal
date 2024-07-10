from scipy import signal
import numpy as np
import xarray as xr
import dask
import scipy

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


def hilbert_mag(da, dim, **kwargs):
    '''
    calculate the hilbert magnitude of da

    Parameters
    ----------
    da : xr.DataArray, or xr.Dataset
    dim : str
        dimension over which to calculate hilbert transform
    '''

    dim_idx = list(da.dims).index(dim)
    kwargs['axis'] = dim_idx

    if isinstance(da, xr.DataArray):
        dac = __hilbert_mag_array(da, dim, **kwargs)
    elif isinstance(da, xr.Dataset):
        dac = da.map(__hilbert_mag_array, dim=dim, **kwargs)

    return dac

def __hilbert_mag_array(da, dim, **kwargs):
    '''
    calculate hilbert transform of da

    Parameters
    ----------
    da : xr.DataArray, or xr.Dataset
    dim : str
        dimension over which to calculate hilbert transform
    '''
    dim_idx = list(da.dims).index(dim)
    kwargs['axis'] = dim_idx
    dac = da.map_blocks(__hilbert_mag_chunk, kwargs=kwargs, template=da)
    return dac

def __hilbert_mag_chunk(da, **kwargs):
    xc = np.abs(signal.hilbert(da.values, **kwargs))
    xcx = xr.DataArray(xc, dims=da.dims, coords=da.coords)
    return xcx


def hilbert(da, dim, **kwargs):
    '''
    calculate the hilbert magnitude of da

    Parameters
    ----------
    da : xr.DataArray, or xr.Dataset
    dim : str
        dimension over which to calculate hilbert transform
    '''

    dim_idx = list(da.dims).index(dim)
    kwargs['axis'] = dim_idx

    if isinstance(da, xr.DataArray):
        dac = __hilbert_array(da, dim, **kwargs)
    elif isinstance(da, xr.Dataset):
        dac = da.map(__hilbert_array, dim=dim, **kwargs)

    return dac

def __hilbert_array(da, dim, **kwargs):
    '''
    calculate hilbert transform of da

    Parameters
    ----------
    da : xr.DataArray, or xr.Dataset
    dim : str
        dimension over which to calculate hilbert transform
    '''
    dim_idx = list(da.dims).index(dim)
    kwargs['axis'] = dim_idx
    template = da.astype('complex')
    dac = da.map_blocks(__hilbert_chunk, kwargs=kwargs, template=template)
    return dac

def __hilbert_chunk(da, **kwargs):
    xc = signal.hilbert(da.values, **kwargs)
    xcx = xr.DataArray(xc, dims=da.dims, coords=da.coords)
    return xcx
