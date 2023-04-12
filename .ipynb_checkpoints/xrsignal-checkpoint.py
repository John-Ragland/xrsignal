'''
xrsignal - implementations of scipy.signal using xarray and dask to compute distributed
'''
from scipy import signal
import numpy as np
import xarray as xr
import dask

def welch(ds, dim, **kwargs):
    '''
    TODO if ds is a dataset, parse to multiple computations for dataarrays
    Parameters
    ----------
    ds : xarray.Dataset, xr.DataArray
        Dataset containing variables to calculate PSD for across given dimension
    dim : str
        Dimension along which the periodogram is computed
    kwargs : kwargs
        passed to signal.welch
            
        fs : float, optional
            Sampling frequency of the x time series. Defaults to 1.0.
        window : str or tuple or array_like, optional
            Desired window to use. If window is a string or tuple, it is passed to get_window to generate the window values, which are DFT-even by default. See get_window for a list of windows and required parameters. If window is array_like it will be used directly as the window and its length must be nperseg. Defaults to a Hann window.
        nperseg : int, optional
            Length of each segment. Defaults to None, but if window is str or tuple, is set to 256, and if window is array_like, is set to the length of the window.
        noverlap : int, optional
            Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
        nfft : int, optional
            Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg. Defaults to None.
        detrend : str or function or False, optional
            Specifies how to detrend each segment. If detrend is a string, it is passed as the type argument to the detrend function. If it is a function, it takes a segment and returns a detrended segment. If detrend is False, no detrending is done. Defaults to ‘constant’.
        return_onesided : bool, optional
            If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Defaults to True, but for complex data, a two-sided spectrum is always returned.
        scaling : { ‘density’, ‘spectrum’ }, optional
            Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz and computing the power spectrum (‘spectrum’) where Pxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density’
        average : { ‘mean’, ‘median’ }, optional
            Method to use when averaging periodograms. Defaults to ‘mean’.
    ''' 

    dim_idx = list(ds.dims).index(dim)
    kwargs['axis'] = dim_idx

    # parse kwargs and construct args
    # This section is outdated, but it handles nperseg
    vars = ['fs', 'window','nperseg','noverlap','nfft','detrend','return_onesided','scaling','axis','average']
    # save nperseg or default
    if 'nperseg' in kwargs:
        nperseg = kwargs['nperseg']
    else:
        nperseg = 256
    defaults = [1.0, 'Hann',nperseg,int(nperseg*2), nperseg, False, True, 'density', -1, 'mean']

    values = []
    for k, item in enumerate(vars):
        if item == 'axis':
            values.append(dim_idx)
        elif item in kwargs:
            values.append(kwargs[item])
        else:
            values.append(defaults[k])

    # create frequency coordinate
    psd_len = int(values[2]/2 + 1)
    f = np.linspace(0,values[0], psd_len)
    
    if isinstance(ds, xr.Dataset):
        template = ds.isel({dim:slice(0,psd_len)}).rename_dims({dim:f'{dim}_frequency'}).assign_coords({f'{dim}_frequency':f})
    elif isinstance(ds, xr.DataArray):
        template = ds.isel({dim:slice(0,psd_len)}).rename({dim:f'{dim}_frequency'}).assign_coords({f'{dim}_frequency':f})
    else:
        raise Exception(f'invalidid datatype {type(ds)} for ds')

    ## Create Template
    # get dimension of template
    # get number of chunks in specified dimension
    dim_list = list(ds.dims)
    dim_idx = dim_list.index(dim)
    _ = dim_list.pop(dim_idx)

    dim_slices = dict(zip(dim_list, [0]*len(dim_list)))
    n_chunks = len(ds.isel(dim_slices).chunks[0])
    chunk_size = ds.isel(dim_slices).chunks[0][0]

    dim_sizes = dict(ds.sizes)
    del dim_sizes[dim]

    template_dims = [dim, f'{dim}_frequency'] + list(dim_sizes.keys())
    template_sizes = []
    template_coords = []
    template_chunks = []

    for template_dim in template_dims:
        if template_dim == dim:
            template_sizes.append(n_chunks)
            template_coords.append(ds[dim][::chunk_size])
            template_chunks.append(1)
        elif template_dim == f'{dim}_frequency':
            template_sizes.append(psd_len)
            template_coords.append(f)
            template_chunks.append(psd_len)
        else:
            template_sizes.append(ds.sizes[template_dim])
            template_coords.append(ds.coords[template_dim])
            template_chunks.append(ds.chunksizes[template_dim][0])

    template_dask = dask.array.random.random(template_sizes, chunks=tuple(template_chunks))
    template = xr.DataArray(template_dask, dims=template_dims, coords=template_coords)
    
    ds_psd = ds.map_blocks(__welch_map, kwargs=kwargs, template=template)
    return ds_psd

def __welch_map(da, kwargs):
    '''
    just gets rid of returned frequency
    '''

    f, Pxx = signal.welch(da.values, **kwargs)
    print(f[-1])
    return Pxx

def filtfilt(ds, dim, b,a, **kwargs):
    '''
    distributed version for filtfilt
    
    For know, I'm going to just build this up for a dataset and see where we get.
    filter only considers a single chunk, therfore causing errors at chunk boundaries

    Parameters
    ----------
    ds : {xr.Dataset, xr.DataArray}
        dataset or dataarray containing data to be filtered.
    dim : str
        dimension to filter accross
    b : array like
        numerator design coefficients for filter
    a : array like
        denominator design coefficients for filter
    kwargs : hashable
        kwargs 

    Return
    ------
    ds_filt : {xr.Dataset, xr.DataArray}
        dask future array that lays out task graph for filtering data
    '''
    dim_idx = list(ds.dims.keys()).index(dim)
    ds_filt = ds.map_blocks(__filtfilt_chunk, args=(dim_idx, b,a, kwargs), template=ds)
    
    return ds_filt

def __filtfilt_chunk(ds, dim_idx, b,a, kwargs):

    ds_filt = {}

    for var in list(ds.data_vars):
        ds_filt[var] = xr.DataArray(signal.filtfilt(b,a,ds[var].values, axis=dim_idx), dims=ds.dims, coords=ds.coords)

    ds_filtx = xr.Dataset(ds_filt)

    return ds_filtx