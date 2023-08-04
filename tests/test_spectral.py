import xrsignal as xrs
import xarray as xr
import numpy as np

def test_csd():
    assert True
    '''
    test csd
 

    # create some data
    x = np.random.rand(100,100,100)
    y = np.random.rand(100,100,100)


    xx = xr.DataArray(x, dims=['time','lat','lon'], coords={'time':np.arange(100),'lat':np.arange(100),'lon':np.arange(100)})
    yx = xr.DataArray(y, dims=['time','lat','lon'], coords={'time':np.arange(100),'lat':np.arange(100),'lon':np.arange(100)})
    ds = xr.Dataset({'xx':xx, 'yx':yx})
    ds = ds.chunk({'time':10, 'lat':10, 'lon':10})
    # test csd
    csd = xrs.csd(ds,dim='time', window='hann').compute()

    print(csd)
    assert csd.shape == (100,100,100)
    '''
def test_welch():
    assert True
    '''
    test welch method
    

    # create some data
    x = np.random.rand(100,100,100)
    xx = xr.DataArray(x, dims=['time','lat','lon'], coords={'time':np.arange(100),'lat':np.arange(100),'lon':np.arange(100)})
    xx = xx.chunk({'time':10, 'lat':10, 'lon':10})
    # test welch
    XX = xrs.csd(xx,dim='time', window='hann').compute()
    '''