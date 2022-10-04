'''
Author: @Yuehui Ma
Date: 2020-09
'''
from astropy.io import fits 
from astropy.wcs import WCS
import numpy as np
from mosaic import *
from astropy.coordinates import SkyCoord
import matplotlib
from matplotlib import rcParams
rcParams["savefig.dpi"] = 300
rcParams["figure.dpi"] = 300
rcParams["font.size"] = 12
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

class img_tools:
    '''
    Methods for showing MWISP fits images, including 2D projected p-p images, l-v images, velocity channel maps, and spectra grids overlaid on a background image.
    '''
    def __init__(self):
        pass
    
    @staticmethod
    def plt_img(file, unit, field=None, figname=None, cm = None, no_stretch = False, scale=None):
        '''
        Create p-p images.
        Parameters
        ----------
        file : str
            a string of the name of input fits file
        unit : str
            data unit to display aside the colorbar
        
        Optional Parameters
        ----------
        field: list
            l-b ranges for only plot a subregion of a fits file
        figname : str
            output file name. Default name is the fits file name plus '.pdf'
        cm : str
            selected build in color map in matplotlib
        no_stretch : bool, Default: False
            whether not to strech the scale of the color bar, in default the color scale will be stretched 
            using a SymLogNorm method in matplotlib.colors.
        scale : list
            bottom and upper limit of the color scale in data value
        
        Return
        ----------
        matplotlib.pyplot fig, axes

        Example
        --------
        >>> from plt_tools import img_tools
        >>> ax = plt_tools.plt_img(fitsfile, unit, field = [lr_left, lr_right, br_low, br_high], figname = figname, cm='jet', scale = [vmin, vmax])
        '''
        dat = fits.getdata(file)
        hd = fits.getheader(file)
        hd.remove('CROTA3', ignore_missing = True)
        hd.remove('CROTA4', ignore_missing = True)
        # dat[np.isnan(dat)] = 0
        wcs = WCS(hd)
        if cm == None:
            cm = 'jet'
        fig = plt.figure(figsize = (12, 3))
        ax = plt.subplot(projection = wcs)
        ax.tick_params(labelsize = 12)
        if field == None:
            extent = None
            data_field = dat
        else:
            ext = SkyCoord(field[0:2], field[2:], frame='galactic', unit = 'deg')
            xr = wcs.world_to_pixel(ext)
            extent = (np.round([xr[0][0], xr[0][1], xr[1][0], xr[1][1]])).astype(int)
            ax.set_xlim(extent[0:2])
            ax.set_ylim(extent[2:])
            data_field = dat#dat[extent[2]:extent[3], extent[0]:extent[1]]
        if scale is None:
            vmin = np.nanmin(data_field[data_field!=0])
            vmax = np.nanmax(data_field[data_field!=0])
        else:
            vmin = scale[0]
            vmax = scale[1]
        if no_stretch == False:
            im = ax.imshow(dat, cmap = cm, origin='lower', vmin = vmin, vmax = vmax, \
                norm = mcolors.SymLogNorm(linthresh=0.03))
        else:
            im = ax.imshow(dat, cmap = cm, origin='lower', vmin = vmin, vmax = vmax)
        cb = plt.colorbar(im, aspect = 12, pad = 0.03, shrink = 0.8)
        matplotlib.colorbar.ColorbarBase.set_label(cb, unit)
        # ax.grid(color='white', ls='--', lw=0.7)
        ax.set_xlabel('Galactic Longitude')
        ax.set_ylabel('Galactic Latitude')
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_major_formatter('d')
        lat.set_major_formatter('d')
        lon.display_minor_ticks(True)
        lat.display_minor_ticks(True)
        if figname == None:
            # fname = os.path.basename(file)
            figname = file.split('.fits')[0]
        plt.savefig(figname + '.pdf', bbox_inches = 'tight')
        return fig, ax

    @staticmethod
    def plt_lvimg(file, unit, asp, cm=None, field=None, figname=None, no_stretch=False, scale=None):
        '''
        Create p-v images.
        Parameters
        ----------
        file : str
            a string of the name of input fits file
        unit : str
            data unit to display aside the colorbar
        asp : float
            aspect ratio 
        
        Optional Parameters
        ----------
        field: list
            l-v ranges for only plot a subregion of a fits file
        cm : str
            selected build in color map in matplotlib
        figname : str
            output file name. Default name is the fits file name plus '.pdf'
        no_stretch : bool, Default: False
            whether not to strech the scale of the color bar, in default the color scale will be stretched 
            using a SymLogNorm method in matplotlib.colors.
        scale : list
            bottom and upper limit of the color scale in data value
        
        Return
        ----------
        matplotlib.pyplot axes

        Example
        --------
        >>> from plt_tools import img_tools 
        >>> img_tools.plt_lvimg(fitsfile, unit, 0.03, figname = figname, field = [55.6, 12, -103, 300], cm = 'RdBu_r', scale =[vmin, vmax])
        '''
        dat = fits.getdata(file)
        lvhd = fits.getheader(file)
        if cm == None:
            cm = 'jet'
        if field == None:
            xr = np.array([0, lvhd['NAXIS1']-1])
            vr = np.array([0, lvhd['NAXIS2']-1])
            lrange = (xr - lvhd['CRPIX1'] + 1) * lvhd['CDELT1'] + lvhd['CRVAL1']
            vrange = ((vr - lvhd['CRPIX2'] + 1) * lvhd['CDELT2'] + lvhd['CRVAL2'])/1000.
            ext = [lrange[0].astype(int), lrange[1].astype(int), vrange[0], vrange[1]]
        else:
            lrange = np.array(field[0:2]) 
            vrange = np.array(field[2:]) 
            xr = (lrange - lvhd['CRVAL1'])/lvhd['CDELT1'] + lvhd['CRPIX1'] -1
            vr = (vrange*1000. - lvhd['CRVAL2'])/lvhd['CDELT2'] + lvhd['CRPIX2'] -1
            ext = [lrange[0], lrange[1], vrange[0], vrange[1]]
            dat = dat[vr[0].astype(int):vr[1].astype(int)+1, xr[0].astype(int):xr[1].astype(int)+1]
        
        plt.figure(figsize = (12, 3))
        ax = plt.subplot()
        ax.tick_params(labelsize = 12, color = 'orange')
        if scale is None:
            vmin = np.nanmin(dat[dat>0])
            vmax = np.nanmax(dat[dat!=0])
        else:
            vmin = scale[0]
            vmax = scale[1]
        if no_stretch:
            im = ax.imshow(dat, cmap = cm, origin='lower', vmin = vmin, vmax = vmax, \
            extent = ext, aspect = asp)
        else:
            im = ax.imshow(dat, cmap = cm, origin='lower', vmin = vmin, vmax = vmax, \
            extent = ext, norm = mcolors.SymLogNorm(linthresh=0.03), aspect = asp)
        ax.set_xlim(ext[0:2])
        ax.set_ylim(ext[2:])
        cb = plt.colorbar(im, aspect = 12, pad = 0.03, shrink = 0.8)
        matplotlib.colorbar.ColorbarBase.set_label(cb, unit)
        # ax.grid(color='white', ls='--', lw=0.7)
        ax.set_xlabel(r'Galactic Longitude ($^{\circ}$)')
        ax.set_ylabel(r'Velocity (km s$^{-1}$)')
        ax.minorticks_on()
        if figname == None:
            figname = file.split('.fits')[0]
        plt.savefig(figname + '.pdf', bbox_inches = 'tight')
        return ax
     
    @staticmethod
    def cov_hd(hdr, threeD = False):
        xr = (np.array([0, hdr['NAXIS1']-1]) - hdr['CRPIX1'] + 1)*hdr['CDELT1'] + hdr['CRVAL1']
        yr = (np.array([0, hdr['NAXIS2']-1]) - hdr['CRPIX2'] + 1)*hdr['CDELT2'] + hdr['CRVAL2']
        if threeD:
            zr = (np.array([0, hdr['NAXIS3']-1]) - hdr['CRPIX3'] + 1)*hdr['CDELT3'] + hdr['CRVAL3']
            return xr, yr, zr
        else: 
            return xr, yr

    @staticmethod
    def cov_subregion(hdr, phy_xr, phy_yr, phy_vr = None):
        xr = (np.array(phy_xr) - hdr['CRVAL1'])/hdr['CDELT1'] + hdr['CRPIX1'] - 1
        yr = (np.array(phy_yr) - hdr['CRVAL2'])/hdr['CDELT2'] + hdr['CRPIX2'] - 1
        xr = (np.round(xr)).astype(int)
        yr = (np.round(yr)).astype(int)
        if phy_vr is None: 
            return xr, yr 
        else:
            # if np.abs(hdr['CDELT3'])>1:
            #     phy_vr = np.array(phy_vr)
            #     phy_vr *= 1000.
            zr = (np.array(phy_vr) - hdr['CRVAL3'])/hdr['CDELT3'] + hdr['CRPIX3'] - 1
            zr = (np.round(zr)).astype(int)
            return xr, yr, zr 
            
    @staticmethod
    def channelmap(file, row, col, vstart=None, vstep=None):
        '''
        Create velocity channel maps.
        Parameters
        ----------
        file : str
            a string of the name of input fits (3D) file
        row : int
            number of rows in the output figure
        col : int 
            number of columns in the output figure 
        
        Optional Parameters
        ----------
        vstart : 
            start value of velcity, if None, it is the velocity of the first channel of the fits file
        vstep :
            velocity steps per panel, if None, it equals to (v_max-v_min)/(row*col)
        
        Return
        ---------
            matplotlib.pyplot fig, axes
        
        Example
        --------
        >>> from plt_tools import img_tools
        >>> ax = plt_tools.channelmap(fitsfile, row, col, vstart, vstep)
        '''
        pt = img_tools()
        data = fits.getdata(file)
        hd = fits.getheader(file)
        nz, ny, nx = data.shape
        ysize = np.round(ny*row*9/(nx*col)).astype(int)
        fig, axs = plt.subplots(nrows=row, ncols=col, figsize=(9,ysize-1))
        fig.subplots_adjust(hspace=0, wspace=0)
        dv = hd['CDELT3']
        if hd['CDELT3']>1:
            dv = hd['CDELT3']/1000.
        lr, br, vr = pt.cov_hd(hd, threeD=True)
        if lr[0]<0:
            lr+=360
        if (hd['CDELT3']>1) & (vstart != None) & (vstep!= None):     
            vstart *= 1000
            vstep *= 1000
        else:
            vstart = vr[0]
            vstep = (vr[1]-vr[0])/(row*col)
        for ax, i in zip(axs.flat, range(row*col)):
            vs = vstart + i*vstep
            vend = vs + vstep
            vr = np.array([vs, vend])
            pxr = [0, 0]
            pyr = [0, 0]
            a1, a2, a3 = pt.cov_subregion(hd, pxr, pyr, vr)
            a3 = np.sort(a3)
            img = np.sum(data[a3[0]:a3[1], :, :], axis=0)*dv
            im = ax.imshow(img, origin = 'lower', cmap = 'RdBu_r', extent = [lr[0], lr[1], br[0], br[1]], vmin = 0.5*dv, vmax = 20, norm = mcolors.SymLogNorm(linthresh=0.03))
            ax.set_xlim(lr)
            ax.set_ylim(br)
            ax.minorticks_on()
            ax.tick_params(axis='both', color='white', top=True, right = True, which = 'minor')
            if hd['CDELT3']>1: 
                vc = str(np.round((vs+vstep/2)/1000, 2))
            else:
                vc = str(np.round((vs+vstep/2), 2))
            ax.text(0.9, 0.85, vc, transform = ax.transAxes, color = 'white', horizontalalignment='right', verticalalignment='center')
            if i != col*(row-1):
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                ax.set_xlabel(r'Galactic longitude ($^{\circ}$)')
                ax.set_ylabel(r'Galactic latitude ($^{\circ}$)')
        cb = plt.colorbar(im, ax=axs.ravel().tolist(), pad = 0.03, shrink = 0.8)
        matplotlib.colorbar.ColorbarBase.set_label(cb, r'K km s$^{-1}$')
        return fig, ax

    @staticmethod
    def overlay_spec(bgfile, specfile, nbox = None):
        '''
        Overplay spectra grid on a 2D map.
        Parameters
        ----------
        bgfile : str
            a string of the name of the background fits file
        specfile : str
            a string of the name of the 3D spectrual cube
        
        Optional Parameters
        ----------
        nbox : list
            numbers of spectral grids along the x and y axes of the background map. Default is [10, 10]

        Return
        ---------
            matplotlib.pyplot fig, axes
        
        Example
        --------
        >>> from plt_tools import img_tools
        >>> ax = plt_tools.overlay_spec(bgfile, specfile, nbox = [10, 10])
        '''
        img = fits.getdata(bgfile)
        hd = fits.getheader(bgfile)
        pt = img_tools()
        lr, br = pt.cov_hd(hd)
        if lr[0]<0:
            extx = lr+360
        else:
            extx = lr
        fig = plt.figure()
        im = plt.imshow(img, origin = 'lower', extent = [extx[0], extx[1], br[0], br[1]], cmap = 'jet')
        plt.minorticks_on()
        cb = plt.colorbar(im, aspect = 15, pad = 0.03, shrink = 0.8)
        ax = plt.gca()
        ax.set_xlabel(r'Galactic longitude ($^{\circ}$)')
        ax.set_ylabel(r'Galactic latitude ($^{\circ}$)')
        ax.set_xlim(extx)
        ax.set_ylim(br)
        if nbox is None:
            nbox = [10, 10]
        boxsize = [np.abs(np.diff(lr))/nbox[0], np.abs(np.diff(br))/nbox[1]]
        hdspec = fits.getheader(specfile)
        if hdspec['NAXIS'] ==4:
            specdata = fits.getdata(specfile)[0]
        else:
            specdata = fits.getdata(specfile)
        xs = np.linspace(lr[0], lr[1], nbox[0]+1)
        if lr[0]<0:
            flag = 360
        else: 
            flag = 0
        ys = np.linspace(br[0], br[1], nbox[1]+1)
        for i in range(len(xs)-1):
            for j in range(len(ys)-1):
                phy_xr = np.array([xs[i], xs[i+1]])
                phy_yr = np.array([ys[j], ys[j+1]])
                xr, yr = pt.cov_subregion(hd, phy_xr, phy_yr)
                reg = specdata[:, yr[0]: yr[1]+1, xr[0]:xr[1]+1]
                spec = np.nanmean(reg, axis = (1, 2))
                rcParams['axes.linewidth'] =0.5
                inset_ax = inset_axes(ax, width='100%', height='100%', bbox_transform=ax.transData, bbox_to_anchor=(xs[i+1]+flag, ys[j], boxsize[0], boxsize[1]), loc = 3, borderpad=0)
                if np.sum(spec)>0:
                    inset_ax.plot(spec, lw = 0.9)
                inset_ax.set_xticklabels([])
                inset_ax.set_yticklabels([])
                inset_ax.tick_params(length = 0.1, width = 0.1)
        return fig, ax

    @staticmethod
    def plt_nh2(file):
        pt = img_tools()
        hdu = fits.open(file)[0]
        dat = hdu.data
        dat[np.isnan(dat)] = 0
        wcs = WCS(hdu.header)
        cm = 'RdBu_r'
        plt.figure(figsize = (12, 3))
        ax = plt.subplot(projection = wcs)
        ax.tick_params(labelsize = 12, color = 'w')
        im = ax.imshow(dat, cmap = cm, origin='lower', vmin = np.min(dat[dat>0]), vmax = np.amax(dat), \
            norm = mcolors.SymLogNorm(linthresh=0.03))
        cb = plt.colorbar(im, aspect = 15, pad = 0., shrink = 1)
        matplotlib.colorbar.ColorbarBase.set_label(cb, r'cm$^{-2}$', fontsize = 12)
        lr, br = pt.cov_hd(hdu.header)
        if (np.abs(np.diff(lr))<=0.5) | (np.abs(np.diff(br))<=0.5):
            fmt = 'd.d'
        elif ((np.abs(np.diff(lr))>0.5) & (np.abs(np.diff(lr))<=1)) | ((np.abs(np.diff(br))>0.5) & (np.abs(np.diff(br))<=1)):
            fmt = 'd.d'
        elif (np.abs(np.diff(lr))>1) | (np.abs(np.diff(br))>1):
            fmt = 'd'
    
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.grid(color = 'white', alpha = 0.6, linestyle = '--')
        lat.grid(color = 'white', alpha = 0.6, linestyle = '--')
        lon.set_ticks(exclude_overlapping = True)
        lat.set_ticks(exclude_overlapping = True)
        lon.set_axislabel('l')
        lat.set_axislabel('b')
        lon.set_major_formatter(fmt)
        lat.set_major_formatter(fmt)
        lon.display_minor_ticks(True)
        lat.display_minor_ticks(True)
        plt.savefig(file[:-10]+'_column_density.pdf', bbox_inches = 'tight')
        return ax