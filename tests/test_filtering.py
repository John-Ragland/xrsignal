'''
test filtering functions
'''

import xarray as xr
import xrsignal as xrs
import numpy as np
from scipy import signal

def test_filtfilt():
    # create some data

    x = np.random.rand(1000,10)
    xx = xr.DataArray(x, dims=['time', 'distance'], coords={'time':np.arange(1000), 'distance':np.arange(10)})
    b,a = signal.butter(3, 0.5, 'low')

    ## test unchunked operation
    xx_xrs_filt = xrs.filtfilt(xx, dim='time', b=b, a=a)
    xx_sig_filt = signal.filtfilt(b, a, x, axis=0)
    
    # make sure all values are the same
    assert np.sum(xx_xrs_filt.values == xx_sig_filt) == 10000
    assert type(xx_xrs_filt) == xr.core.dataarray.DataArray
    assert xx_xrs_filt.shape == (1000,10)

    ## test chunked operation
    xx = xx.chunk({'time':100, 'distance':1})
    xx_xrs_filt = xrs.filtfilt(xx, dim='time', b=b, a=a)
    assert type(xx_xrs_filt) == xr.core.dataarray.DataArray
    assert xx_xrs_filt.shape == (1000, 10)


def test_hilbert():
    x = np.random.rand(1000,10)
    xx = xr.DataArray(x, dims=['time', 'distance'], coords={'time':np.arange(1000), 'distance':np.arange(10)})
    xx_chunk = xx.chunk({'time':100, 'distance':1})
    ds = xr.Dataset({'xx':xx})
    ds_chunk = xr.Dataset({'xx':xx_chunk})

    # compare values to scipy.signal.hilbert
    xx_xrs_hilb = xrs.hilbert(xx, dim='time')
    xx_sig_hilb = signal.hilbert(x, axis=0)
    assert(np.sum(xx_xrs_hilb.values == xx_sig_hilb) == 10000)
    assert(type(xx_xrs_hilb) == xr.core.dataarray.DataArray)

    ds_xrs_hilb = xrs.hilbert(ds, dim='time').compute()
    assert(type(ds_xrs_hilb) == xr.core.dataset.Dataset)

    ds_xrs_hilb_chunk = xrs.hilbert(ds_chunk, dim='time').compute()
    assert(type(ds_xrs_hilb_chunk) == xr.core.dataset.Dataset)

    xx_chunk_xrs_hilb = xrs.hilbert(xx_chunk, dim='time')
    assert(type(xx_chunk_xrs_hilb) == xr.core.dataarray.DataArray)