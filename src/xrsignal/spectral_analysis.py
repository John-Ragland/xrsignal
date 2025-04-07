from scipy import signal
import numpy as np
import xarray as xr
import dask
import scipy

def csd(data, dim, dB=False, **kwargs):
    '''
    Estimate the cross power spectral density, Pxy, using Welch’s method.

    Parameters
    ----------
    data : xr.Dataset
        dataset containing data to estimate cross power spectral density.
        must only contain two data variables
    dim : str
        dimension to calculate PSD over
    dB : bool
        if True, return PSD in dB
    kwargs : hashable
        passed to scipy.signal.csd

    Returns
    -------
    csd : xr.DataArray
        cross power spectral density
    '''

    # check type and number of data variables
    if not isinstance(data, xr.Dataset):
        raise Exception('data must be xr.Dataset')
    
    if len(data.data_vars) != 2:
        raise Exception('data must contain only two data variables')

    ## Parse Kwargs
    # fs or set to 1
    if 'fs' in kwargs:
        fs = kwargs['fs']
    else:
        fs = 1

    if 'return_onesided' in kwargs:
        return_onesided = kwargs['return_onesided']
    else:
        return_onesided = True

    # Get length of PSD
    if 'nperseg' in kwargs:
        nperseg = kwargs['nperseg']
        psd_len = int(nperseg/2 + 1)
    else:
        psd_len = 129  # default value for signal.welch
        nperseg = 256

    if 'nfft' in kwargs:
        nfft = kwargs['nfft']
        if return_onesided:
            psd_len = int(nfft/2 + 1)
        else:
            psd_len = nfft
    else: #nperseg is nfft by default
        if return_onesided:
            psd_len = int(nperseg/2 + 1)
        else:
            psd_len = nperseg

    # Create new dimensions of PSD object
    original_dims = list(data.dims)

    new_dims = original_dims.copy()
    new_dims[original_dims.index(dim)] = f'{dim}_frequency'
    new_dims.append(dim)

    variables = list(data.data_vars)

    # Get number of chunks in each dimension
    # Assumes that all data variables have the same chunking
    original_chunksize = dict(zip(original_dims, data[variables[0]].data.chunksize))
    nchunks = []

    for k, single_dim in enumerate(original_dims):
        nchunks.append(data[variables[0]].shape[k] /
                       original_chunksize[single_dim])
    nchunks = dict(zip(original_dims, nchunks))

    # raise exception if number of chunks is not integer
    if nchunks[dim] % 1 != 0:
        raise Exception(
            f'number of chunks in dimension "{dim}" is required to be integer. there are currently {nchunks[dim]} chunks')
    
    # convert new dimension to integer
    nchunks[dim] = int(nchunks[dim])

    # Get size of every dimension
    original_sizes = dict(data.sizes)
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

    if return_onesided:
        freq_coords = scipy.fft.rfftfreq(nperseg, 1/fs)
    else:
        freq_coords = scipy.fft.fftfreq(nperseg, 1/fs)

    template = xr.DataArray(
        dask.array.random.random(
            list(new_sizes.values()), chunks=list(new_chunk_sizes.values())),
        dims=new_dims,
        coords={f'{dim}_frequency': freq_coords},
        name=f'psd across {dim} dimension')

    kwargs['dim'] = dim
    Pxy = xr.map_blocks(__csd_chunk, data, template=template,  kwargs=kwargs)

    if dB:
        return (10*np.log10(Pxy)).sortby(f'{dim}_frequency')
    else:
        return Pxy.sortby(f'{dim}_frequency')

def __csd_chunk(data, dim, **kwargs):
    '''
    estimate CSD for single chunk of dataarray

    Parameters
    ----------
    data : xr.Dataset
        dataset containing data to estimate cross power spectral density.
        must contain only two data variables
    dim : str
        dimension to calculate CSD over
    **kwargs : hashable
        passed to scipy.signal.csd
    '''
    # Get x and y from data
    variables = list(data.data_vars)
    x = data[variables[0]].values
    y = data[variables[1]].values

    # Create new dimensions of PSD object
    original_dims = list(data.dims)
    psd_dim_idx = original_dims.index(dim)

    new_dims = original_dims.copy()
    new_dims[original_dims.index(dim)] = f'{dim}_frequency'
    new_dims.append(dim)

    # Estimate PSD and convert to xarray.DataArray
    f, Pxy = signal.csd(x, y, axis=psd_dim_idx, **kwargs)
    Pxy = np.expand_dims(Pxy, -1)

    Pxy_x = xr.DataArray(Pxy, dims=new_dims, coords={f'{dim}_frequency': f})
    
    return np.abs(Pxy_x)


def welch(data, dim, dB=False, nan=False, **kwargs):
    '''
    Estimate power spectral density using welch method
    For now, an integer number of chunks in PSD dimension is required
    
    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        data array to estimate power spectral density
    dim : str
        dimension to calculate PSD over
    dB : bool
        if True, return PSD in dB
    nan : bool
        if True, use welch_nan instead of welch. This throws out segments with NAN and still calculates the PSD
    '''

    if isinstance(data, xr.DataArray):
        Sxx = __welch_da(data, dim, dB=dB, nan=nan, **kwargs)
    elif isinstance(data, xr.Dataset):
        Sxx =  data.map(__welch_da, dim=dim, dB=dB, nan=nan, **kwargs)
    else:
        raise Exception('data must be xr.DataArray or xr.Dataset')

    return Sxx.sortby(f'{dim}_frequency')

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
    # unpack nan kwarg
    nan = kwargs.pop('nan', False)

    # Create new dimensions of PSD object
    original_dims = list(da.dims)
    psd_dim_idx = original_dims.index(dim)

    new_dims = original_dims.copy()
    new_dims[original_dims.index(dim)] = f'{dim}_frequency'
    new_dims.append(dim)

    # Estimate PSD and convert to xarray.DataArray
    if nan:
        f, P, _ = welch_nan(da.values, axis=psd_dim_idx, **kwargs)
    else:
        f, P = signal.welch(da.values, axis=psd_dim_idx, **kwargs)

    P = np.expand_dims(P, -1)

    Px = xr.DataArray(P, dims=new_dims, coords={f'{dim}_frequency': f})

    return Px

def __welch_da(da, dim, dB=False, nan=False, **kwargs):
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
    nan : bool
        if True, use welch_nan instead of welch
    '''

    ## Parse Kwargs
    # fs or set to 1
    if 'fs' in kwargs:
        fs = kwargs['fs']
    else:
        fs = 1
    
    if 'return_onesided' in kwargs:
        return_onesided = kwargs['return_onesided']
    else:
        return_onesided = True

    # Get length of PSD
    if 'nperseg' in kwargs:
        nperseg = kwargs['nperseg']
    else:
        nperseg = 256
    
    if 'nfft' in kwargs:
        nfft = kwargs['nfft']
        if return_onesided:
            psd_len = int(nfft/2 + 1)
        else:
            psd_len = nfft
    else: #nperseg is nfft by default
        nfft = nperseg
        if return_onesided:
            psd_len = int(nperseg/2 + 1)
        else:
            psd_len = nperseg

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

    if return_onesided:
        freq_coords = scipy.fft.rfftfreq(nfft, 1/fs)
    else:
        freq_coords = scipy.fft.fftfreq(nfft, 1/fs)  
    
    template = xr.DataArray(
        dask.array.random.random(
            list(new_sizes.values()), chunks=list(new_chunk_sizes.values())),
        dims=new_dims,
        coords={f'{dim}_frequency': freq_coords},
        name=f'psd across {dim} dimension')

    kwargs['dim'] = dim
    kwargs['nan'] = nan

    Pxx = xr.map_blocks(__welch_chunk, da, template=template,  kwargs=kwargs)
    
    if dB:
        return 10*np.log10(Pxx)
    else:
        return Pxx


def welch_nan(x, fs=1.0, window='hann', nperseg=256, noverlap=None, 
                           nfft=None, detrend='constant', return_onesided=True, 
                           scaling='density', axis=-1, average='mean'):
    """
    Compute Welch's PSD estimate with NaN handling.
    
    This function divides data into segments, removes segments containing NaN values,
    and then computes the PSD using only valid segments.
    
    Parameters are the same as scipy.signal.welch
    
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of x.
    n_valid_segments : int
        Number of valid segments (without NaN) used in the computation.
    """
    # Handle default parameters similar to scipy.signal.welch
    if noverlap is None:
        noverlap = nperseg // 2
        
    # Calculate number of segments and their starting indices
    step = nperseg - noverlap
    indices = np.arange(0, len(x) - nperseg + 1, step)
    
    # Create segments and check which ones contain NaN values
    segments = np.array([x[i:i+nperseg] for i in indices])
    valid_segments = ~np.isnan(segments).any(axis=1)
    
    # Count the number of valid segments
    n_valid_segments = np.sum(valid_segments)
    valid_percent = n_valid_segments / len(segments)

    # If no valid segments, return NaN
    if n_valid_segments == 0:
        f = np.fft.rfftfreq(nperseg, d=1.0/fs) if return_onesided else np.fft.fftfreq(nperseg, d=1.0/fs)
        return f, np.full(len(f), np.nan), 0
    
    # Keep only valid segments
    valid_data = segments[valid_segments]
    
    # Flatten into a 1D array with all valid segments concatenated
    flattened_valid_data = valid_data.reshape(-1)
    
    # Call scipy.signal.welch with the valid data
    # We set nperseg to the segment length and noverlap to 0 since we've already segmented the data
    f, Pxx = signal.welch(flattened_valid_data, fs=fs, window=window, nperseg=nperseg, 
                         noverlap=0, nfft=nfft, detrend=detrend, 
                         return_onesided=return_onesided, scaling=scaling)
    
    return f, Pxx, valid_percent