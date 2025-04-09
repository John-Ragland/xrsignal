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
        f, P = welch_nan(da.values, axis=psd_dim_idx, **kwargs)
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

def welch_nan(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
              detrend='constant', return_onesided=True, scaling='density',
              axis=-1, average='mean'):
    """
    Estimate power spectral density using Welch's method, ignoring NaN values.
    
    This function behaves exactly like scipy.signal.welch but handles NaN values 
    by ignoring segments containing NaN when computing the average.
    
    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. Defaults to 'hann'.
    nperseg : int, optional
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. Defaults to `nperseg // 2`.
    nfft : int, optional
        Length of the FFT used. If `None`, the default is `nperseg`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a string, it is
        passed as the `type` argument to the `detrend` function. If it is a
        function, it takes a segment and returns a detrended segment. If `False`,
        no detrending is done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If `False` return
        a two-sided spectrum. If `x` is complex, the default is `False`.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density') where
        `Pxx` has units of V**2/Hz and computing the power spectrum ('spectrum')
        where `Pxx` has units of V**2, if `x` is measured in V and fs is measured
        in Hz. Defaults to 'density'.
    axis : int, optional
        Axis along which the periodogram is computed; the default is over the
        last axis (i.e., axis=-1).
    average : { 'mean', 'median' }, optional
        Method to use when averaging periodograms. Defaults to 'mean'.
    
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of x.
    """
    # Convert x to numpy array
    x = np.asarray(x)
    
    # Check if there are any NaN values
    if not np.any(np.isnan(x)):
        # If no NaN values, use the original welch function
        return signal.welch(x, fs=fs, window=window, nperseg=nperseg,
                            noverlap=noverlap, nfft=nfft, detrend=detrend,
                            return_onesided=return_onesided, scaling=scaling,
                            axis=axis, average=average)
    
    # Handle negative axis
    if axis < 0:
        axis = x.ndim + axis
    
    # Set default parameters
    if nperseg is None:
        nperseg = min(256, x.shape[axis])
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    if nfft is None:
        nfft = nperseg
    
    # Get the window
    if isinstance(window, str) or isinstance(window, tuple):
        win = signal.get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win) != nperseg:
            raise ValueError('window must have length of nperseg')
    
    # Determine if input is complex
    is_complex = np.iscomplexobj(x)
    
    # Calculate frequencies for return array
    if return_onesided and not is_complex:
        # Real input, one-sided frequency range
        freqs = np.fft.rfftfreq(nfft, 1.0/fs)
    else:
        # Complex input or two-sided frequency range
        if return_onesided and is_complex:
            # For complex input with return_onesided=True, scipy.signal.welch
            # would issue a warning and compute the full spectrum
            import warnings
            warnings.warn('return_onesided=True is ignored for complex input. '
                         'Computing two-sided spectrum.')
        freqs = np.fft.fftfreq(nfft, 1.0/fs)
    
    # Init periodogram array
    segment_psds = []
    
    # Calculate step size between segments
    step = nperseg - noverlap
    
    # For each slice of the input data, calculate a periodogram
    ind = 0
    while ind + nperseg <= x.shape[axis]:
        # Extract segment
        segment_slice = [slice(None)] * x.ndim
        segment_slice[axis] = slice(ind, ind + nperseg)
        segment = x[tuple(segment_slice)].copy()
        
        # Check if segment contains NaN
        if not np.any(np.isnan(segment)):
            # Detrend if needed
            if detrend != False:
                segment = signal.detrend(segment, type=detrend, axis=axis)
            
            # Apply window (broadcasting to the correct shape)
            win_shape = [1] * segment.ndim
            win_shape[axis] = len(win)
            segment = segment * win.reshape(win_shape)
            
            # Compute FFT
            if return_onesided and not is_complex:
                # Real input, one-sided FFT
                fft_data = np.fft.rfft(segment, n=nfft, axis=axis)
            else:
                # Complex input or two-sided FFT
                fft_data = np.fft.fft(segment, n=nfft, axis=axis)
            
            # Compute PSD based on scaling
            if scaling == 'density':
                # Power spectral density
                segment_psd = abs(fft_data)**2 / (fs * (win**2).sum())
            else:  # scaling == 'spectrum'
                # Power spectrum
                segment_psd = abs(fft_data)**2 / win.sum()**2
            
            # Apply one-sided scaling for real data
            if return_onesided and not is_complex:
                # Multiply by 2 for one-sided (except at DC and Nyquist)
                if nfft % 2 == 0:  # Even nfft, Nyquist present
                    segment_psd[..., 1:-1] *= 2
                else:  # Odd nfft, Nyquist not present
                    segment_psd[..., 1:] *= 2
            
            segment_psds.append(segment_psd)
        
        # Move to next segment
        ind += step
    
    # If no valid segments found, return NaN array
    if not segment_psds:
        if return_onesided and not is_complex:
            Pxx = np.full(freqs.shape, np.nan)
        else:
            Pxx = np.full(freqs.shape, np.nan)
        return freqs, Pxx
    
    # Stack periodograms for averaging
    segment_psds = np.stack(segment_psds, axis=0)
    
    # Average the periodograms
    if average == 'mean':
        Pxx = np.mean(segment_psds, axis=0)
    elif average == 'median':
        Pxx = np.median(segment_psds, axis=0)
    else:
        raise ValueError(f"Unknown average: {average}")
    
    return freqs, Pxx