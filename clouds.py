'''
Author: @Yuehui Ma
Date: 2021-10 
'''
from astropy.io import fits
import numpy as np
from scipy import stats

class Cloud:
    '''
    1. Calculate physical maps such as tex, optical depth, centroid velocity, and the 1st and 2nd moment maps of a cloud from the 12CO and 13CO MWISP data. 
    2. Get physical parameters of a cloud, such as mass, median and maximum Tex, median column denstiy, dispersion of normalized column density, median optical depth, column density detection limit, sonic Mach number, skewness and Kurtosis of the column density distribution.
    '''
    def __init__(self, name):
        self.name = name 
        self.CO12 = name + 'U.fits'
        self.CO13 = name+'L_13CO.fits'
        self.rms13 = name+'L_rms.fits'

    def cal_tex(self, output=True):
        '''
        Calculate the excitation temperature map from an input 12CO data cube.

        Parameter
        ----------
            output : bool, Default is True. If Flase, the method does not output a fits file of the excitation temperature.

        Output
        ----------
            Tex map in fits format. The output file is in the same path as the input file.

        Return 
        ----------
            Tex : numpy.ndarray 
        '''
        dat = fits.getdata(self.CO12)
        hd = fits.getheader(self.CO12)
        T0 = 5.53213817
        JTbg = (np.exp(T0/2.7)-1)**(-1)
        peak12 = np.amax(dat, axis = 0)
        Tex = T0*(np.log(1 + (peak12/T0 + JTbg)**(-1)))**(-1)
        Tex[Tex == 2.7] = 0
        hd12 = fits.Header()
        hd12['NAXIS'] = 2
        hd12['NAXIS1'] = dat.shape[2]
        hd12['NAXIS2'] = dat.shape[1]
        hd12['CTYPE1'] = 'GLON-CAR'
        hd12['CTYPE2'] = 'GLAT-CAR'
        hd12['CUNIT1'] = 'deg'
        hd12['CUNIT2'] = 'deg'
        hd12['CRVAL1'] = hd['CRVAL1']
        hd12['CRVAL2'] = hd['CRVAL2']
        hd12['CRPIX1'] = hd['CRPIX1']
        hd12['CRPIX2'] = hd['CRPIX2']
        hd12['CDELT1'] = hd['CDELT1']
        hd12['CDELT2'] = hd['CDELT2']
        #hd12 = WCS(hd12).to_header()
        hd12['BUNIT']  = 'K'
        #hd12['LONPOLE'] = 0.
        #hd12['LATPOLE'] = 90.0
        if output is True:
            fits.writeto(self.CO12[:-5]+'_Tex2.fits', Tex, hd12, overwrite=True)
        return Tex

    def cal_tau13_n13(self, msk13 = None, output = True):
        '''
        Calculate the optical depth map and the column density map for an input 13CO data cube.

        Parameter
        ----------
            msk13 : str (Optional)
                The name of the mskfile (3D) of noise.
            output : bool (Optional)
                Default is True. If Flase, the method does not output fits files of the m0, N_H2, and tau maps.

        Output
        ----------
            Moment 0, H2 column density, and optical depth maps of the input 13CO cube in fits format.
            The output files are in the same path as the input file.
        
        Return 
        ----------
            nh2 : numpy.ndarray
            tau13 : numpy.ndarray
        '''
        hd = fits.getheader(self.CO13)
        if hd['NAXIS'] == 4:
            dat = fits.getdata(self.CO13)[0]
        else:
            dat = fits.getdata(self.CO13)
        rms = fits.getdata(self.rms13)
        if msk13 is not None:
            msk = fits.getdata(msk13) > 0
            dat *=msk
        tex = self.cal_tex(output=False) #tex12.copy()
        peak13 = np.amax(dat, axis = 0)
        J13 = (np.exp(5.29/tex) - 1)**(-1)
        tau13 = -np.log(1. - peak13/(5.29*(J13 - 0.164)))
        tau13[(peak13 == 0) | (tau13 < 0)] = 0
        tau13[np.isnan(tau13)] = 0
        m013 = np.sum(dat, axis = 0)*abs(hd['CDELT3'])/1000.
        num = np.count_nonzero(dat > 0, axis=0)
        thresh = 3*np.sqrt(num)*rms*abs(hd['CDELT3'])/1000.
        m0 = m013*(m013>thresh)
        nh2 = 2.42*1e14*7e5*m0/(1.-np.exp(-5.29/tex))*(1. + 0.88/tex) * tau13/(1. - np.exp(-tau13))
        nh2[np.isnan(nh2)] = 0
        hd13 = fits.Header()
        hd13['NAXIS'] = 2
        hd13['NAXIS1'] = m0.shape[1]
        hd13['NAXIS2'] = m0.shape[0]
        hd13['CTYPE1'] = 'GLON-CAR'
        hd13['CTYPE2'] = 'GLAT-CAR'
        hd13['CUNIT1'] = 'deg'
        hd13['CUNIT2'] = 'deg'
        hd13['CRVAL1'] = hd['CRVAL1']
        hd13['CRVAL2'] = hd['CRVAL2']
        hd13['CRPIX1'] = hd['CRPIX1']
        hd13['CRPIX2'] = hd['CRPIX2']
        hd13['CDELT1'] = hd['CDELT1']
        hd13['CDELT2'] = hd['CDELT2']
        #hd13 = WCS(hd12).to_header()
        hd13['BUNIT']  = 'K km s-1'
        if output is True:
            sufix = (self.CO13).split('_13CO')[0]
            fits.writeto(sufix+'_m0.fits', m0, hd13, overwrite = True)
            hd13['BUNIT']  = 'tau'
            fits.writeto(sufix+'_tau.fits', tau13, hd13, overwrite = True)
            hd13['BUNIT']  = 'cm-2'
            fits.writeto(sufix+'_nh2.fits', nh2, hd13, overwrite = True)
        return nh2, tau13
    
    def cal_v(self, output=True):
        '''
        Calculate the moment 1 (centroid velocity) and 2 (velocity dispersion) maps for an input 13CO data cube.
        
        Parameter
        ----------
            output : bool, Default is True. If Flase, the method does not output fits files of the m1 and m2 maps.
   
        Output
        ----------
            Moment 1 and 2 maps of the input 13CO cube in fits format. The output files are in the same path as the input file.
        
        Return 
        ----------
            m1 : numpy.ndarray
            m2 : numpy.ndarray
        '''
        hd = fits.getheader(self.CO13)
        if hd['NAXIS'] == 4:
            dat = fits.getdata(self.CO13)[0]
        else:
            dat = fits.getdata(self.CO13)
        varray = np.ones_like(dat)
        v = (np.arange(hd['NAXIS3']) - hd['CRPIX3'] +1)*hd['CDELT3'] + hd['CRVAL3'] 
        for i in range(hd['NAXIS3']):
            varray[i, :, :]*=v[i]
        m1 = (np.sum(varray*dat, axis=0)/np.sum(dat, axis=0))/1000.
        for j in range(hd['NAXIS3']):
            varray[j, :, :] = (varray[j, :, :]/1000.-m1)**2
        m2 = np.sqrt(np.sum(varray*dat, axis=0)/np.sum(dat, axis=0))
        hdnew = fits.Header()
        hdnew['NAXIS'] = 2
        hdnew['NAXIS1'] = m1.shape[1]
        hdnew['NAXIS2'] = m1.shape[0]
        hdnew['CTYPE1'] = 'GLON-CAR'
        hdnew['CTYPE2'] = 'GLAT-CAR'
        hdnew['CUNIT1'] = 'deg'
        hdnew['CUNIT2'] = 'deg'
        hdnew['CRVAL1'] = hd['CRVAL1']
        hdnew['CRVAL2'] = hd['CRVAL2']
        hdnew['CRPIX1'] = hd['CRPIX1']
        hdnew['CRPIX2'] = hd['CRPIX2']
        hdnew['CDELT1'] = hd['CDELT1']
        hdnew['CDELT2'] = hd['CDELT2']
        #hdnew = WCS(hd12).to_header()
        hdnew['BUNIT']  = 'km s-1'
        if output is True:
            fits.writeto(self.CO13[:-5]+'_m1.fits', m1, hdnew, overwrite = True)
            hdnew['BUNIT']  = 'km s-1'
            fits.writeto(self.CO13[:-5]+'_sigmav.fits', m2, hdnew, overwrite = True)
        return m1, m2


    def get_mass(self, d):
        '''
        Get the mass of an individual cloud.
        Parameters
        ----------
        d : float
            distance in unit of kpc.
        '''
        h2file = self.name+'L_nh2.fits'
        nh2 = fits.getdata(h2file)
        mass = np.sum(nh2[nh2>0])*(np.radians(30/3600.)*(d*1000.*3.08568025e18))**2*(2.8*1.6606e-24)/(1.99e33)
        return mass
    

    def cal_nlim(self):
        '''
        Calculate the median reference detection limit of the column density. It is defined as three channels with intensities above 2 sigma. 
        '''
        ftex = self.name + 'U_Tex.fits'
        ftau = self.name + 'L_tau.fits'
        tex = fits.getdata(ftex)
        tau = fits.getdata(ftau)
        rms = fits.getdata(self.rms13)
        sigma = 3*2*rms*0.166
        nsigma = 2.42*1e14*7e5*sigma/(1.-np.exp(-5.29/tex))*(1. + 0.88/tex) * tau/(1. - np.exp(-tau))
        nsigma[tau==0] = 0
        nlim = np.median(nsigma[tau>0])
        return nlim

    def get_med_max_tex(self):
        '''
        Get median and maximum tex of a cloud. The mask file is the Nh2 map. 
        
        Return
        -------
            The median and maximum Tex of the pixels with column densities above the median reference detection limit.
        '''
        texfile = self.name+'U_Tex.fits'
        tex = fits.getdata(texfile)
        mskfile = self.name+'L_nh2.fits'
        msk = fits.getdata(mskfile)
        thresh = self.cal_nlim()
        medtex = np.nanmedian(tex[(tex>0) & (msk>thresh)])
        maxtex = np.nanmax(tex[(tex>0) & (msk>thresh)])
        return medtex, maxtex

    def get_nh2(self):
        '''
        Get mean of the column densities and standard deviation of the normalized column densities of a cloud.
        '''
        nh2 = fits.getdata(self.name+'L_nh2.fits')
        thresh = self.cal_nlim()
        log_nh2 = np.log10(nh2[nh2>thresh])
        rho = nh2[nh2!=0]/np.mean(nh2[nh2!=0])
        sn = np.std(rho)
        return np.mean(nh2[nh2>thresh]), sn

    def get_tau(self):
        '''
        Get the median optical depth of 13CO among the pixels with nh2 above the detection limit.
        '''
        tau = fits.getdata(self.name+'L_tau.fits')
        msk = fits.getdata(self.name+'L_nh2.fits')
        thresh = self.cal_nlim()
        return np.nanmedian(tau[(tau>0) & (msk>thresh)])

    def get_size(self, d):
        '''
        Return physical size in unit of pc. The only parameter d is distance in unit of kpc. 
        '''
        nh2 = fits.getdata(self.name+'L_nh2.fits')
        npix = len(nh2[nh2>0])
        s = npix*(30.)**2
        angr = np.sqrt(s/np.pi)
        r = angr*d*1000./206265.
        return r

    def get_ms_all(self):
        '''
        Get sonic Mach number within the PPV boundary of a 13CO molecular cloud.
        '''
        mfile = self.CO13
        tmb = fits.getdata(mfile)
        hdt = fits.getheader(mfile)
        v =  ((np.linspace(0, hdt['NAXIS3']-1, hdt['NAXIS3']) - hdt['CRPIX3'] + 1) * hdt['CDELT3'] + hdt['CRVAL3'])/1000.
        #print(np.amax(m1))
        arr = np.zeros_like(tmb)
        for ch in range(hdt['NAXIS3']):
            arr[ch, :, :] = tmb[ch, :, :]*v[ch]
        v0 = np.sum(arr)/np.sum(tmb)
        arr = np.zeros_like(tmb)
        for ch in range(hdt['NAXIS3']):
            arr[ch, :, :] = tmb[ch, :, :]*(v[ch] - v0)**2
        sv2 = np.sum(arr)/np.sum(tmb)
        #sv2 = np.std(m1)
    
        medtex, maxtex = self.get_med_max_tex()
        sound = 0.188*np.sqrt(medtex/10.)
        ms = np.sqrt(3.*(sv2-sound**2))/sound
        print('hello', self.name, ms, v0, sound, np.sqrt(sv2))
        return ms

    def get_skw(self):
        nh2 = fits.getdata(self.name+'L_nh2.fits')
        s = np.log(nh2[nh2>0]/np.nanmean(nh2[nh2>0]))
        skew = stats.skew(s)
        return skew

    def get_kurt(self):
        nh2 = fits.getdata(self.name+'L_nh2.fits')
        s = np.log(nh2[nh2>0]/np.nanmean(nh2[nh2>0]))
        kurt = stats.kurtosis(s)
        return kurt

 