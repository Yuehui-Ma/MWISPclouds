from astropy.io import fits
from astropy.table import Table
import numpy as np
import gc
import os

quadrant = 'Q2'
mskfile = '/share/public/catalogQ123/Q2/DBSCANresults/mergedCubedbscanS2P4Con1_Clean.fits'
datafile = '/share/public/catalogQ123/Q2/mergedCube.fits' 
rmsfile = '/share/public/catalogQ123/Q2/100_150_U_rmsCrop.fits' 
catalog = '/share/public/catalogQ123/Q2/DBSCANresults/mergedCubedbscanS2P4Con1_Clean.fit' 
npix_lim = 200.

cat = Table.read(catalog)
lon = cat['x_cen'].data
lat = cat['y_cen'].data
v0 = cat['v_cen'].data
idx = cat['_idx'].data
area_exact = cat['area_exact'].data/0.25
select_id = idx[np.where(area_exact > npix_lim)]
msk = fits.getdata(mskfile)
#hdr = mskhd.header

for clouds in select_id: 
    cidx = np.where(idx == clouds)
    cld = 'G' + '%07.3f' % lon[cidx] + '%+07.3f' % lat[cidx] + '%+07.2f' % v0[cidx]
    dat = fits.getdata(datafile)
    hdr = fits.getheader(datafile)
    dat[msk != clouds] = 0
    loc = np.array(np.where(msk == clouds))
    zr = [np.amin(loc[0, :]), np.amax(loc[0, :])]
    yr = [np.amin(loc[1, :]), np.amax(loc[1, :])]
    xr = [np.amin(loc[2, :]), np.amax(loc[2, :])]
    newdat = dat[zr[0]:zr[1]+1, yr[0]:yr[1]+1, xr[0]:xr[1]+1]
    print(clouds, cld, zr, yr, xr)
   
    
    pth = quadrant+'/'+cld+'/'
    # lr = (xr - np.array(hdr['CRPIX1']) + 1) * np.array(hdr['CDELT1']) + np.array(hdr['CRVAL1'])
    # br = (yr - np.array(hdr['CRPIX2']) + 1) * np.array(hdr['CDELT2']) + np.array(hdr['CRVAL2'])
    # vr = (zr - np.array(hdr['CRPIX3']) + 1) * np.array(hdr['CDELT3']) + np.array(hdr['CRVAL3'])
    
    hdrms = fits.open(rmsfile)[0]
    rms = hdrms.data
    rmscut = rms[yr[0]:yr[1]+1, xr[0]:xr[1]+1]
    
    hd_new = fits.Header()
    hd_new['NAXIS'] = 3
    hd_new['NAXIS1'] = newdat.shape[2]
    hd_new['NAXIS2'] = newdat.shape[1]
    hd_new['NAXIS3'] = newdat.shape[0]
    hd_new['CTYPE1'] = hdr['CTYPE1']
    hd_new['CTYPE2'] = hdr['CTYPE2']
    hd_new['CTYPE3'] = hdr['CTYPE3']
    hd_new['CUNIT1'] = 'deg'
    hd_new['CUNIT2'] = 'deg'
    hd_new['CUNIT3'] = 'm/s'   
    hd_new['CRVAL1'] = hdr['CRVAL1']#lr[0]
    hd_new['CRVAL2'] = hdr['CRVAL2']#br[0]
    hd_new['CRVAL3'] = hdr['CRVAL3']#vr[0]
    hd_new['CRPIX1'] = hdr['CRPIX1'] - xr[0]
    hd_new['CRPIX2'] = hdr['CRPIX2'] - yr[0]
    hd_new['CRPIX3'] = hdr['CRPIX3'] - zr[0]
    hd_new['CDELT1'] = -30./3600.
    hd_new['CDELT2'] = 30./3600.
    hd_new['CDELT3'] = hdr['CDELT3']
    hd_new['RESTFRQ'] = hdr['RESTFRQ']

    hd_new['BUNIT']  = 'K (TMB)'
    if not os.path.exists(pth):
        os.mkdir(pth)
    fits.writeto(pth+cld+'U.fits', newdat, hd_new, overwrite = True)

    hdrms_new = fits.Header()
    hdrms_new['NAXIS'] = 2
    hdrms_new['NAXIS1'] = rmscut.shape[1]
    hdrms_new['NAXIS2'] = rmscut.shape[0]
    hdrms_new['CTYPE1'] = hdr['CTYPE1']
    hdrms_new['CTYPE2'] = hdr['CTYPE2']
    hdrms_new['CUNIT1'] = 'deg'
    hdrms_new['CUNIT2'] = 'deg'
    hdrms_new['CRVAL1'] = hdr['CRVAL1']#lr[0]
    hdrms_new['CRVAL2'] = hdr['CRVAL2']#br[0]
    hdrms_new['CRPIX1'] = hdr['CRPIX1'] - xr[0]
    hdrms_new['CRPIX2'] = hdr['CRPIX2'] - yr[0]
    hdrms_new['CDELT1'] = -30./3600.
    hdrms_new['CDELT2'] = 30./3600.
    
    hdrms_new['BUNIT']  = 'K (TMB)'
    fits.writeto(pth+cld+'U_rms.fits', rmscut, hdrms_new, overwrite = True)   
    gc.collect()
