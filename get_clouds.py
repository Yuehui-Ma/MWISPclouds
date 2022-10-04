#%%
import shutil
from build_NPDF import *
from astropy.table import Table
from astropy.coordinates import SkyCoord
from mwispDBSCAN import MWISPDBSCAN
import glob
from mosaic import *
import gc
def get13(maskfile_12, cube13, catalog, rmsfile_13, quadrant, outprefix = None, npix_lim = None):
    cat = Table.read(catalog)
    lon = cat['x_cen'].data
    lat = cat['y_cen'].data
    v0 = cat['v_cen'].data
    idx = cat['_idx'].data
    area_exact = cat['area_exact'].data/0.25
    if npix_lim is not None:
        select_id = idx[np.where(area_exact > npix_lim)]
    else:
        select_id = idx
    for clouds in select_id: 
        cidx = np.where(idx == clouds)
        cld = 'G' + '%07.3f' % lon[cidx] + '%+07.3f' % lat[cidx] + '%+07.2f' % v0[cidx]
        msk = fits.getdata(maskfile_12)
        hd12 = fits.getheader(maskfile_12)
        msk[msk != clouds] = 0
        loc = np.array(np.where(msk == clouds))
        zr = [np.amin(loc[0, :]), np.amax(loc[0, :])]
        yr = [np.amin(loc[1, :]), np.amax(loc[1, :])]
        xr = [np.amin(loc[2, :]), np.amax(loc[2, :])]
        cmsk = msk[zr[0]:zr[1]+1, yr[0]:yr[1]+1, xr[0]:xr[1]+1]
        print(clouds, cld, zr, yr, xr)
        v12 = (np.arange(zr[0], zr[1]+1, 1) - hd12['CRPIX3'] +1)*hd12['CDELT3'] + hd12['CRVAL3'] 
        varray = np.ones_like(cmsk).astype('float')
        for i in range(len(v12)):
            varray[i, :, :]*=v12[i]
        varray[cmsk == 0] = np.nan
        vmins = np.nanmin(varray, axis=0)
        vmaxs = np.nanmax(varray, axis=0)

        hd13 = fits.getheader(cube13)
        if hd13['NAXIS'] == 4:
            dat13 = fits.getdata(cube13)[0] # for Q1
        else: 
            dat13 = fits.getdata(cube13)
        v13 = (np.arange(hd13['NAXIS3']) - hd13['CRPIX3'] +1)*hd13['CDELT3'] + hd13['CRVAL3'] 
        x, y, z = cov_subregion(hd13, [0, 0], [0, 0], [v12[0], v12[-1]])
        
        varray13 = np.ones((z[1]-z[0]+1, cmsk.shape[1], cmsk.shape[2]))
        
        for i in range(z[1]-z[0]+1):
            varray13[i, :, :]*=v13[z[0]+i]

        mask13 = np.empty_like(varray13)
        # for j in range(cmsk.shape[1]):
        #     for k in range(cmsk.shape[0]):
        #         mask13[:, j, k] = (varray13[:, j, k] >= vmins[j, k]) & (varray13[:, j, k] <= vmaxs[j, k]) 
        for ch in range(len(mask13)):
            mask13[ch, :, :] = (vmins <= varray13[ch, :, :]) & (varray13[ch, :, :]<=vmaxs)
        subdat13 = dat13[z[0]:z[1]+1, yr[0]:yr[1]+1, xr[0]:xr[1]+1]
        subdat13 *= mask13

        pth = quadrant+'/'+cld+'/'

        rms = fits.getdata(rmsfile_13)
        rmscut = rms[yr[0]:yr[1]+1, xr[0]:xr[1]+1]
        
        hd_new = fits.Header()
        hd_new['NAXIS'] = 3
        hd_new['NAXIS1'] = subdat13.shape[2]
        hd_new['NAXIS2'] = subdat13.shape[1]
        hd_new['NAXIS3'] = subdat13.shape[0]
        hd_new['CTYPE1'] = hd13['CTYPE1']
        hd_new['CTYPE2'] = hd13['CTYPE2']
        hd_new['CTYPE3'] = hd13['CTYPE3']
        hd_new['CUNIT1'] = 'deg'
        hd_new['CUNIT2'] = 'deg'
        hd_new['CUNIT3'] = 'm/s'   
        hd_new['CRVAL1'] = hd13['CRVAL1'] #lr[0]
        hd_new['CRVAL2'] = hd13['CRVAL2'] #br[0]
        hd_new['CRVAL3'] = hd13['CRVAL3'] #v13[z[0]]
        hd_new['CRPIX1'] = hd13['CRPIX1'] - xr[0]
        hd_new['CRPIX2'] = hd13['CRPIX2'] - yr[0]
        hd_new['CRPIX3'] = hd13['CRPIX3'] - z[0]
        hd_new['CDELT1'] = -30./3600. 
        hd_new['CDELT2'] = 30./3600. 
        hd_new['CDELT3'] = hd13['CDELT3']
    
        hd_new['BUNIT']  = 'K (TMB)'
        if not os.path.exists(pth):
            os.mkdir(pth)
        fits.writeto(pth+cld+'L.fits', subdat13, hd_new, overwrite = True)


        hdrms_new = fits.Header()
        hdrms_new['NAXIS'] = 2
        hdrms_new['NAXIS1'] = rmscut.shape[1]
        hdrms_new['NAXIS2'] = rmscut.shape[0]
        hdrms_new['CTYPE1'] = hd13['CTYPE1']
        hdrms_new['CTYPE2'] = hd13['CTYPE2']
        hdrms_new['CUNIT1'] = 'deg'
        hdrms_new['CUNIT2'] = 'deg'
        hdrms_new['CRVAL1'] = hd13['CRVAL1']
        hdrms_new['CRVAL2'] = hd13['CRVAL2']
        hdrms_new['CRPIX1'] = hd13['CRPIX1'] - xr[0]
        hdrms_new['CRPIX2'] = hd13['CRPIX2'] - yr[0]
        hdrms_new['CDELT1'] =  -30./3600. 
        hdrms_new['CDELT2'] =  30./3600. 
        
        hdrms_new['BUNIT']  = 'K (TMB)'
        fits.writeto(pth+cld+'L_rms.fits', rmscut, hdrms_new, overwrite = True)

        gc.collect()

# %%
if __name__=='__main__':
    quadrant = 'Q2'
    mskfile = '/share/public/catalogQ123/Q2/DBSCANresults/mergedCubedbscanS2P4Con1_Clean.fits'
    datafile = '/share/public/105_150_L.fits' 
    rmsfile = '/share/public/105_150_L_rms.fits' 
    catalog = '/share/public/catalogQ123/Q2/DBSCANresults/mergedCubedbscanS2P4Con1_Clean.fit' 
    get13(mskfile, datafile, catalog, rmsfile, quadrant, npix_lim=200.)
