import xrsignal as xrs
import xarray as xr
import numpy as np

def test_csd():

    # create some data
    x = np.random.rand(1000,50,50)
    y = np.random.rand(1000,50,50)

    xx = xr.DataArray(x, dims=['time','lat','lon'], coords={'time':np.arange(1000),'lat':np.arange(50),'lon':np.arange(50)})
    yx = xr.DataArray(y, dims=['time','lat','lon'], coords={'time':np.arange(1000),'lat':np.arange(50),'lon':np.arange(50)})
    ds = xr.Dataset({'xx':xx, 'yx':yx})
    ds = ds.chunk({'time':50, 'lat':50, 'lon':50})
    csd = xrs.csd(ds, dim='time', window='hann', nperseg=10, noverlap=5).mean('time')

    assert type(csd) == xr.core.dataarray.DataArray
    assert csd.shape == (6, 50, 50)

def test_welch():
    assert True
    # create some data
    x = np.random.rand(1000,50,50)


    xx = xr.DataArray(x, dims=['time','lat','lon'], coords={'time':np.arange(1000),'lat':np.arange(50),'lon':np.arange(50)})
    xx = xx.chunk({'time':50, 'lat':50, 'lon':50})
    ds = xr.Dataset({'xx':xx})
    psd_ds = xrs.welch(ds, dim='time', window='hann', nperseg=10, noverlap=5).mean('time')
    psd_da = xrs.welch(xx, dim='time', window='hann', nperseg=10, noverlap=5).mean('time')

    assert type(psd_da) == xr.core.dataarray.DataArray
    assert type(psd_ds) == xr.core.dataset.Dataset
    assert psd_da.shape == (6, 50, 50)
    assert psd_ds['xx'].shape == (6, 50, 50)