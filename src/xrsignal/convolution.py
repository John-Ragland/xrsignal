'''
convolution functions - a couple of notes about implementing convolution

- there are several different use cases that require different implications
    - the conventional signal.convolution takes
'''
from scipy import signal
import xarray as xr
import dask
import numpy as np
from typing import Union

def correlate(
        in1: Union[xr.DataArray, xr.Dataset],
        in2: xr.DataArray,
        **kwargs) -> Union[xr.DataArray, xr.Dataset]:
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
        kwargs = kwargs | {'in2':in2}
        return in1.map(correlate_da, **kwargs)

def correlate_da(
        in1: xr.DataArray,
        in2: xr.DataArray,
        **kwargs) -> Union[xr.DataArray, xr.Dataset]:
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
            raise ValueError(f'Dask array only supports correlation along an axis that has a single chunk. You can change xarray chunks with .chunk()')
    
    # compute shape of output
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'full'
    
    out_sizes = dict(in1.sizes)
    if mode == 'full':
        out_sizes[in2_dim] = in1.sizes[in2_dim] + in2.sizes[in2_dim] - 1
    elif mode == 'same':
        out_sizes[in2_dim] = in1.sizes[in2_dim]
    elif mode == 'valid':
        out_sizes[in2_dim] = in1.sizes[in2_dim] - in2.sizes[in2_dim] + 1
    else:
        raise ValueError(f'invalid mode: {mode}')
    
    # output coordinates for now I'm just dropping correlation cooridinate
    #TODO add correlation coordinate
    out_coords = dict(in1.coords)
    _ = out_coords.pop(in2_dim)

    # create output template
    out_template = xr.DataArray(
        dask.array.random.random(list(out_sizes.values())),
        dims=list(out_sizes.keys()),
        coords = out_coords,
    )
    out_template = out_template.drop_indexes(out_template.coords.keys())
    # chunk output template the match input template
    outChunkSizes = dict(in1.chunksizes)
    _ = outChunkSizes.pop(in2_dim)
    out_template = out_template.chunk(outChunkSizes)

    # map blocks if dask array
    if isinstance(in1.data, dask.array.core.Array):
        Rxy = xr.map_blocks(
            correlate_chunk,
            in1,
            template=out_template,
            kwargs=kwargs | {'in2':in2}
        )

    else:
        Rxy = correlate_chunk(in1, in2, **kwargs)

    return Rxy

def correlate_chunk(in1, in2, **kwargs):
    '''
    helper function to correlate two xarray objects (in1, in2) for single chunk
    this method is hidden from user and is called by correlate. So all checks on
    dimensions have already been done and in1/in2 are assumed to have correct structure

    Parameters
    ----------
    in1 : Union[xr.DataArray, xr.Dataset]
        data to be correlated. This can be ndimensional data
    in2 : xr.DataArray
        data to correlate with in1. This must be a 1D xarray.DataArray
    **kwargs : dict
        keyword arguments to pass to scipy.signal.correlate
    
    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        correlated data. type will be determined by type of in1
    '''
    Rxy = np.apply_along_axis(
        signal.correlate,
        in1.get_axis_num(in2.dims[0]),
        in1,
        in2,
        **kwargs)
    out_coords = dict(in1.coords)
    _ = out_coords.pop(in2.dims[0])
    Rxy_x = xr.DataArray(Rxy, dims=in1.dims, coords=out_coords)

    Rxy_x = Rxy_x.drop_indexes(Rxy_x.coords.keys())
    return Rxy_x

def convolve(
        in1: Union[xr.DataArray, xr.Dataset],
        in2: xr.DataArray,
        **kwargs) -> Union[xr.DataArray, xr.Dataset]:
    '''
    convolve two xarray objects (in1, in2)
    - unlike the scipy.signal.convolve, in1 can be ndimensional, but in2 must be 1d
    - the dimension of in2 must be contained in in1 and this is the dimension that the correlation is computed along

    Parameters
    ----------
    in1 : xr.DataArray
        data to be convolved. This can be ndimensional data
    in2 : xr.DataArray
        data to convolve with in1. This must be a 1D xarray.DataArray.
    **kwargs : dict
        keyword arguments to pass to scipy.signal.convolve
    
    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        convolved data. type will be determined by type of in1
    '''

    if isinstance(in1, xr.DataArray):
        return convolve_da(in1, in2, **kwargs)
    elif isinstance(in1, xr.Dataset):
        kwargs = kwargs | {'in2':in2}
        return in1.map(convolve_da, **kwargs)

def convolve_da(
        in1: xr.DataArray,
        in2: xr.DataArray,
        **kwargs) -> Union[xr.DataArray, xr.Dataset]:
    '''
    convolve two xarray objects (in1, in2)
    - unlike the scipy.signal.convolve, in1 can be ndimensional, but in2 must be 1d
    - the dimension of in2 must be contained in in1 and this is the dimension that the correlation is computed along

    Parameters
    ----------
    in1 : xr.DataArray
        data to be convolved. This can be ndimensional data
    in2 : xr.DataArray
        data to convolve with in1. This must be a 1D xarray.DataArray.
    **kwargs : dict
        keyword arguments to pass to scipy.signal.convolve
    
    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        convolved data. type will be determined by type of in1
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
            raise ValueError(f'Dask array only supports correlation along an axis that has a single chunk. You can change xarray chunks with .chunk()')
    
    # compute shape of output
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'full'
    
    out_sizes = dict(in1.sizes)
    if mode == 'full':
        out_sizes[in2_dim] = in1.sizes[in2_dim] + in2.sizes[in2_dim] - 1
    elif mode == 'same':
        out_sizes[in2_dim] = in1.sizes[in2_dim]
    elif mode == 'valid':
        out_sizes[in2_dim] = in1.sizes[in2_dim] - in2.sizes[in2_dim] + 1
    else:
        raise ValueError(f'invalid mode: {mode}')
    
    # output coordinates for now I'm just dropping correlation cooridinate
    #TODO add correlation coordinate
    out_coords = dict(in1.coords)
    _ = out_coords.pop(in2_dim)

    # create output template
    out_template = xr.DataArray(
        dask.array.random.random(list(out_sizes.values())),
        dims=list(out_sizes.keys()),
        coords = out_coords,
    )
    out_template = out_template.drop_indexes(out_template.coords.keys())
    # chunk output template the match input template
    outChunkSizes = dict(in1.chunksizes)
    _ = outChunkSizes.pop(in2_dim)
    out_template = out_template.chunk(outChunkSizes)

    # map blocks if dask array
    if isinstance(in1.data, dask.array.core.Array):
        Rxy = xr.map_blocks(
            convolve_chunk,
            in1,
            template=out_template,
            kwargs=kwargs | {'in2':in2}
        )

    else:
        Rxy = convolve_chunk(in1, in2, **kwargs)

    return Rxy

def convolve_chunk(in1, in2, **kwargs):
    '''
    helper function to convolve two xarray objects (in1, in2) for single chunk
    this method is hidden from user and is called by convolve. So all checks on
    dimensions have already been done and in1/in2 are assumed to have correct structure

    Parameters
    ----------
    in1 : Union[xr.DataArray, xr.Dataset]
        data to be convolved. This can be ndimensional data
    in2 : xr.DataArray
        data to convolve with in1. This must be a 1D xarray.DataArray
    **kwargs : dict
        keyword arguments to pass to scipy.signal.convolve
    
    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        convolved data. type will be determined by type of in1
    '''
    Rxy = np.apply_along_axis(
        signal.convolve,
        in1.get_axis_num(in2.dims[0]),
        in1,
        in2,
        **kwargs)
    out_coords = dict(in1.coords)
    _ = out_coords.pop(in2.dims[0])
    Rxy_x = xr.DataArray(Rxy, dims=in1.dims, coords=out_coords)

    Rxy_x = Rxy_x.drop_indexes(Rxy_x.coords.keys())
    return Rxy_x
