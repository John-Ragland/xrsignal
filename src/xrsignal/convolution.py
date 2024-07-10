from scipy import signal
import xarray as xr
import dask
import numpy as np
from typing import Union


def correlate(in1: Union[xr.DataArray, xr.Dataset], in2: xr.DataArray, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
    '''
    correlate two xarray objects (in1, in2)
    - unlike the scipy.signal.correlate, in1 can be ndimensional, but in2 must be 1d
    - the dimension of in2 must be contained in in1 and this is the dimension that the correlation is computed along

    Parameters
    ----------
    in1 : xr.DataArray
        data to be correlated. This can be ndimensional data
    in2 : xr.DataArray
        data to correlate with in1. This must be a 1D xarray.DataArray.
    **kwargs : dict
        keyword arguments to pass to scipy.signal.correlate

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        correlated data. type will be determined by type of in1
    '''

    if isinstance(in1, xr.DataArray):
        return correlate_da(in1, in2, **kwargs)
    elif isinstance(in1, xr.Dataset):
        kwargs = kwargs | {'in2': in2}
        return in1.map(correlate_da, **kwargs)


def correlate_da(in1: xr.DataArray, in2: xr.DataArray, **kwargs) -> xr.DataArray:
    '''
    correlate two xarray objects (in1, in2)
    - unlike the scipy.signal.correlate, in1 can be ndimensional, but in2 must be 1d
    - correlation dimension is specified by the dimension of in2, and this dimension must also be contained in in1

    Parameters
    ----------
    in1 : xr.DataArray
        data to be correlated. This can be ndimensional data
    in2 : xr.DataArray
        data to correlate with in1. This must be a 1D xarray.DataArray.
    **kwargs : dict
        keyword arguments to pass to scipy.signal.correlate

    Returns
    -------
    xr.DataArray
        correlated data. type will be determined by type of in1
    '''
    # check that in2 is 1d
    if len(in2.dims) != 1:
        raise ValueError('in2 must be 1d')

    in1_dims = in1.dims
    in2_dim = in2.dims[0]

    # chekc if in2_dim is in in1_dims
    if in2_dim not in in1_dims:
        raise ValueError(f'in2 dimension, {in2_dim} must be contained in in1 dimensions, {in1_dims}')

    # check if in1 only has single chunk in in2_dim
    if in1.chunks is not None:
        if len(in1.chunks[in1.get_axis_num(in2_dim)]) > 1:
            raise ValueError(
                f'Dask array only supports correlation along an axis that has a single chunk. You can change xarray chunks with .chunk()'
            )

    # move correlation dimension to front (for in1)
    in1 = in1.transpose(in2.dims[0], ...)

    # compute shape of output chunks
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'full'

    out_chunks = list(in1.data.chunksize)
    if mode == 'full':
        out_chunks[0] = in1.shape[0] + in2.sizes[in2_dim] - 1
    elif mode == 'same':
        out_chunks[0] = in1.shape[0]
    elif mode == 'valid':
        out_chunks[0] = in1.shape[0] - in2.sizes[in2_dim] + 1
    else:
        raise ValueError(f'invalid mode: {mode}')

    #convert back to tuple
    out_chunks = tuple(out_chunks)

    # move correlation dimension to front and convert to dask / numpy
    in1_array = in1.transpose(in2.dims[0], ...).data
    in2_array = in2.data

    # compute coordinates (from in1 and correlation cooridinate is dropped)
    out_coords = dict(in1.coords)
    _ = out_coords.pop(in2.dims[0])

    Rxy_dask = in1_array.map_blocks(correlate_chunk, in2=in2_array, chunks=out_chunks, **kwargs)

    Rxy = xr.DataArray(Rxy_dask, dims=in1.dims, coords=out_coords, name='correlated data')

    return Rxy


def correlate_chunk(in1, in2, **kwargs):
    '''
    correlate_chunk_dask, cross-correlate single block of dask arrays in1 and in2

    Parameters
    ----------
    in1 : dask.array.core.Array
        data to be correlated. in1 mush have shape (N, ... ) where N is correlation dimension
    in2 : dask.array.core.Array
        data to correlate with in1. in2 must have shape (M, ) where M is correlation dimension
    **kwargs : dict
        keyword arguments to pass to scipy.signal.correlate
    '''
    Rxy = signal.correlate(in1, np.expand_dims(in2, 1), **kwargs)
    return Rxy
