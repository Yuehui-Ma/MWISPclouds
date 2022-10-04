'''
Author: @Yuehui Ma
Date: 2021-10 
'''
from astropy.io import fits 
import numpy as np
from matplotlib import rcParams
rcParams["savefig.dpi"] = 300
rcParams["figure.dpi"] = 300
rcParams["font.size"] = 12
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import emcee
from math import erf, pi
import os
import corner


class pdf_tools:
    '''
    Methods for building an N-PDF from a given N_h2 fits file, fitting N-PDFs, testing the goodness of fit, classifying N-PDF shapes, recording the fitted parameters. 
    '''
    def __init__(self):
        pass

    @staticmethod
    # MCMC 
    # Model 1, Log-normal (LN)
    def log_prior_ln(theta, x):
        '''
        Logarithmic prior for LN function.
        '''
        mu, sigma_s = theta
        c = x.copy()
        if (np.amin(c) < mu < np.amax(c)) & (0 < sigma_s < 5*np.std(c)):
            return 0.0
        else:
            return -np.inf
    
    @staticmethod
    def pdf_ln(theta, x):
        '''
        LN PDF function. For plotting and calculating the likelihood for MCMC fitting.
        '''
        mu, sigma_s = theta
        f = x.copy()
        f = np.exp(-(x-mu)**2 /2 /sigma_s**2)
        #normalize
        integral = np.sqrt(pi/2) * sigma_s *\
                (erf((np.amax(x)-mu)/sigma_s/np.sqrt(2))\
                - erf((np.amin(x)-mu)/sigma_s/np.sqrt(2)))
        return f/integral

    @staticmethod
    def log_probability_ln(theta, x):
        '''
        Log posterier probability = log prior + log likelihood (for MCMC fitting).
        '''
        pt = pdf_tools()
        lp = pt.log_prior_ln(theta, x)
        if not np.isfinite(lp):
            return -np.inf
        f = pt.pdf_ln(theta, x)
        log_ln =  lp + np.sum(np.log(f[f>0]))
        return log_ln

    @staticmethod
    def redchisq(ydata, ymod, deg, sd = None):
        '''
        Reduced chi-squared.
        '''
        if np.any(sd == None):  
            chisq = np.sum((ydata-ymod)**2)  
        else:  
            chisq = np.sum(((ydata-ymod)/sd)**2)
        nu = np.size(ydata) - 1 - deg 
        return chisq/nu   

    @staticmethod
    def plt_npdf(file, model=None, par=None, low=None):
        '''
        Plot an N-PDF for a given N_H2 fits file. 
        Parameters
        ----------
        model : str (Optional, 'LN', 'LP', 'both')
            Default is None. If not given, only create the histogram of the N-PDF. When model = 'LN' or 'LP', overlay a fitted log-normal or log-normal + power-law function on the histogram. If model = 'both', create a histogram, overlay the two fitted models, and plot the residual below the N-PDF.
        par : list
            List of fitted parameters with 2, 4, or 6 elements.
        low : float
            Lower limit of column denstiy for N-PDF fitting.

        Return
        ----------
            If model = 'both', then return the fig, axs objects and the reduced chi-sqaured of the LN and LN+PL fitting, and the flags of whether the residuals of the LN and LN+PL fittings distribute normally around 0, else only return fig, axs.
        '''
        pt = pdf_tools()
        dat = fits.getdata(file)
        dat[np.isnan(dat)] = 0
        data = dat[dat!=0]
        x = np.log(data/np.mean(data))
        if model=='both':
            fig = plt.figure(figsize = (5, 8))
            axs = fig.add_subplot(2, 1, 1)
        else:
            fig, axs = plt.subplots(figsize = (6, 4))
            fig.subplots_adjust(wspace = 0.4)

        binsize = 0.2
        xr = [(np.round(np.amin(x))).astype(int)-1, (np.round(np.amax(x))).astype(int)+1]
        bins = np.arange(xr[0], xr[1]+binsize, binsize)
        #num_bins = int((np.amax(x)-np.amin(x)) // binsize)
        n, bins, patches = axs.hist(x, bins, density = True,\
                                    log = True, color = 'grey')
        axs.set_xlabel('ln (N/<N>)', fontsize = 15)
        axs.set_ylabel('Probability density', fontsize = 15)
        axs.set_xlim(xr[0], xr[1])
        ymin = 0.5/(len(x)*binsize)
        axs.set_ylim(ymin, 30)
        axs.minorticks_on()
        axs.text(0.63, 0.9, os.path.basename(file).split('L_')[0], fontsize = 15, horizontalalignment='center',\
            verticalalignment='center', transform=axs.transAxes)
        if low == None:
            lows = -np.inf
        else:
            lows = np.log(low/np.mean(data))
            axs.plot([lows, lows], [-10, 50], 'g--')

        x1, x2 = axs.get_xlim()
        y1, y2 = axs.get_ylim()
        xrange = np.exp([x1, x2])*np.mean(data)
        yrange = np.array([y1, y2])*binsize*len(x)

        axs2 = axs.twinx()
        axs2.set_ylim(yrange[0], yrange[1])
        axs2.set_ylabel('Number', fontsize = 15)
        axs2.semilogy()
        axs2.plot([x1, x2], [10, 10], 'g--')
        axs2.minorticks_on()
        axs3 = axs.twiny()
        axs3.set_xlim(xrange[0], xrange[1])
        axs3.semilogx()
        axs3.set_xlabel(r'Column Density (cm$^{-2}$)', fontsize = 15)
        
        if model == None:
            pass
        if (model == 'LN'):
            print('Overlay a log-normal fitting:')
            if len(par)>2: 
                p_ln = par[:2]
            else:
                p_ln = par
            xx = np.linspace(x.min()-1,x.max()+1,200)
            axs.plot(xx, pt.pdf_ln(p_ln, xx),'r-', label = 'LN')
            axs.legend(loc='lower left')
        
                
        if (model == 'LP'):
            print('Overlay a lognormal+powerlaw fitting:')
            if len(par)>4: 
                p_lp = par[2:]
            else:
                p_lp = par
            xx = np.linspace(x.min()-1,x.max()+1,200)
            axs.plot(xx, pt.pdf_lp(p_lp, xx),'b-', label = 'LN+PL')
            axs.legend(loc='lower left')

        if (model == 'both'):
            print('Overlay the LN and LN+PL fittings:')
            p_ln = par[:2]
            p_lp = par[2:]
            xx = np.linspace(x.min()-1,x.max()+1,200)
            axs.plot(xx, pt.pdf_ln(p_ln, xx),'r-', label = 'LN')
            axs.plot(xx, pt.pdf_lp(p_lp, xx),'b-', label = 'LN+PL')
            axs.legend(loc='lower left')
            ax_rs = fig.add_subplot(2, 1, 2)
            inset_ax = fig.add_axes([.21, .35, .25, .1])

            bincenter = bins + binsize/2.
            x0 = bincenter[0:-1]
            yfit_ln = pt.pdf_ln(p_ln, x0)
            rs_ln = n - yfit_ln
            ax_rs.plot(x0[np.where((n>0) & (x0>lows))], rs_ln[np.where((n>0) & (x0>lows))], 'o', label = 'LN', color='indianred')
            yfit_lp = pt.pdf_lp(p_lp, x0)
            rs_lp = n - yfit_lp
            ax_rs.plot(x0[np.where((n>0) & (x0>lows))], rs_lp[np.where((n>0) & (x0>lows))], 'o', label = 'LN+PL', color= 'royalblue')
            ax_rs.plot(xr, [0,0], '--', color = 'grey')
            ax_rs.legend(loc = 'lower left')
            ax_rs.set_xlim(xr[0], xr[1])
            ax_rs.set_ylim(-0.3, 0.3)
            ax_rs.set_xlabel('ln (N/<N>)', fontsize=15)
            ax_rs.set_ylabel('Residual', fontsize=15)
            ax_rs.minorticks_on()
            inset_ax.hist(rs_ln[np.where((n>0) & (x0>lows))], alpha=0.8, color = 'indianred')
            inset_ax.hist(rs_lp[np.where((n>0) & (x0>lows))], alpha=0.6, color='royalblue')
            inset_ax.set_xlim(-0.25, 0.25)
            ydat = n[np.where((n>0) & (x0>lows))]
            ymod_ln = yfit_ln[np.where((n>0) & (x0>lows))]
            ymod_lp = yfit_lp[np.where((n>0) & (x0>lows))]
            sigydat = np.sqrt(n * 1./(len(x)*binsize))
            axs.errorbar(x0[n>0], n[n>0], yerr = sigydat[n>0], color = 'limegreen', marker = '', ls = '', capsize=1.7, elinewidth=0.8)
            sd = sigydat[np.where((n>0) & (x0>lows))]
            rdsq_ln = pt.redchisq(ydat, ymod_ln, 2, sd = sd)
            rdsq_lp = pt.redchisq(ydat, ymod_lp, 4, sd = sd)
            frac_rs_ln1 = np.sum(rs_ln[np.where((n>0) & (x0>lows))]>0)/np.size(rs_ln[np.where((n>0) & (x0>lows))])
            frac_rs_lp1 = np.sum(rs_lp[np.where((n>0) & (x0>lows))]>0)/np.size(rs_lp[np.where((n>0) & (x0>lows))])
            frac_rs_ln2 = np.sum(rs_ln[np.where((n>0) & (x0>lows))]<0)/np.size(rs_ln[np.where((n>0) & (x0>lows))])
            frac_rs_lp2 = np.sum(rs_lp[np.where((n>0) & (x0>lows))]<0)/np.size(rs_lp[np.where((n>0) & (x0>lows))])
            
            if (frac_rs_ln1>0.8) | (frac_rs_ln2>0.8):
                flag_ln = 0
            else:
                flag_ln = 1
        
            if (frac_rs_lp1>0.8) | (frac_rs_lp2>0.8):
                flag_lp = 0
            else:
                flag_lp = 1
        if model == 'both':
            return fig, axs, rdsq_ln, rdsq_lp, flag_ln, flag_lp
        else: 
            return fig, axs

    # Model 2, Log-normal + Power-law (LP)
    @staticmethod
    def log_prior_lp(theta, x):
        mu, sigma_s, br, alpha = theta
        c = x.copy()
        if (np.amin(c) < mu < np.amax(c)) & (0 < sigma_s < 5*np.std(c)) & (mu < br < np.amax(c)) & (-10 < alpha < 0):
            return 0.0
        else:
            return -np.inf

    @staticmethod
    def pdf_lp(theta, x):
        mu, sigma_s, br, alpha = theta
        f = x.copy()
        if br>np.amax(x):
            return 0
        f[x<br] = np.exp(-(x[x<br]-mu)**2 /2 /sigma_s**2)	#gauss
        f[x>=br] = np.exp(alpha*x[x>=br]) / np.exp(alpha*br) * np.exp(-(br-mu)**2/2/sigma_s**2)	#power law
        #normalize
        integral = np.sqrt(pi/2) * sigma_s * (erf((br-mu)/sigma_s/np.sqrt(2))\
                -erf((np.amin(x)-mu)/sigma_s/np.sqrt(2)))\
                + 1/alpha * np.exp(alpha*np.amax(x))/np.exp(alpha*br) * np.exp(-(br-mu)**2/2/sigma_s**2) \
                - 1/alpha * np.exp(-(br-mu)**2/2/sigma_s**2)
        return f/integral
    
    @staticmethod
    def log_probability_lp(theta, x):
        pt = pdf_tools()
        lp = pt.log_prior_lp(theta, x)
        if not np.isfinite(lp):
            return -np.inf
        f = pt.pdf_lp(theta, x)
        log_lp = lp + np.sum(np.log(f[f>0]))
        return log_lp
    
    @staticmethod
    def fit(file, model, low=None):
        '''
        Implement MCMC fitting for N-PDF.
        Parameters
        ---------
        file : str
            name of the nh2 fits file.
        model : str
            model function for which should be used in the fitting process.
        low : float (Optional)
            lower limit of the column density, above which the fitting should be implemented.
        '''
        pt = pdf_tools()
        dat = fits.getdata(file)
        dat[np.isnan(dat)] = 0
        dat = dat[dat !=0]
        x = np.log(dat/np.mean(dat))
        m0 = np.median(x)
        s0 = x.std()
        burnin = 1000
        tau = 50 # auto correlation time
        name = os.path.basename(file).split('.fits')[0]
        print('start '+model+' fitting for cloud '+name+':')
        nsteps = 10000
        nwalkers = 32
        if model == 'LN':
            par_ini = [m0, s0]
            npars = 2
            sufix = 'ln'
        elif model == 'LN+PL':
            b0 = m0 + s0
            k0 = -2.
            par_ini = [m0, s0, b0, k0]
            npars = 4
            sufix = 'lp'
        backfile =  file[:-10]+'_'+sufix+'.h5'
        pos = par_ini + 0.0001 * np.random.randn(nwalkers, npars)
        nwalkers, ndim = pos.shape
        if low == None:
            pass
        else:
            lows = np.log(low/np.mean(dat))
            x = x[x>lows]
        
        if os.path.exists(backfile): os.remove(backfile)
        backend = emcee.backends.HDFBackend(backfile)
        backend.reset(nwalkers, ndim)

        if model == 'LN':
            sampler = emcee.EnsembleSampler(nwalkers, ndim, pt.log_probability_ln, args=[x], backend = backend)
        elif model == 'LN+PL':
            sampler = emcee.EnsembleSampler(nwalkers, ndim, pt.log_probability_lp, args=[x], backend = backend)
        sampler.run_mcmc(pos, nsteps, progress=False)
        fig1, axes = plt.subplots(ndim, figsize=(7, 5), sharex=True)
        samples = sampler.get_chain()
        labels = [r'$\mu$', r'$\sigma_s$', r'$b_r$', r'$\alpha$']
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            # ax.set_ylim(-5, 5)
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("step number")
        axes[-1].minorticks_on()
        fig1.savefig(file[:-10]+ '_chain_'+sufix+'.pdf', bbox_inches = 'tight')

        
        flat_samples = sampler.get_chain(discard=burnin, thin=tau, flat=True)
        fig2 = corner.corner(flat_samples, labels=labels[0:npars], show_titles = True, \
                quantiles = [0.16, 0.84], smooth = True)#  labelpad = 0.3
        fig2.savefig(file[:-10]+ '_corner_'+sufix+'.pdf', bbox_inches = 'tight')
        
        par = np.median(flat_samples, axis = 0)
        perr = np.zeros((npars, 2))
        for i in range(npars):
            perr[i, :]=np.diff(np.percentile(flat_samples[:, i], [16, 50, 84]))
            perr[i]*=np.array([-1, 1])
        return x, par, perr
    
    @staticmethod
    def get_BIC(sample, theta):
        pt = pdf_tools()
        theta_ln = theta[0:2]
        theta_lp = theta[2:]
        f1 = pt.pdf_ln(theta_ln, sample)
        f2 = pt.pdf_lp(theta_lp, sample)
        L1 = np.sum(np.log(f1[f1>0]))
        L2 = np.sum(np.log(f2[f2>0]))
        BIC1 = -2.*L1 + 2.*np.log(len(sample))
        BIC2 = -2.*L2 + 4.*np.log(len(sample))
        delta_BIC = BIC1-BIC2
        if (BIC1<BIC2):
            typ = 'LN'
        elif (BIC1>BIC2):
            typ = 'LN+PL'
        # return BIC1, BIC2, delta_BIC, typ
        return typ

    @staticmethod
    def model_sel(sample, pars, file, low):
        '''
        Select the final model in the fitting results, and label the reduced chi-squared values, the BIC selected model, and the flags of reduced chi-squared on the N-PDF figure.

        Return
        ---------
        fig object and the final category of the N-PDF (LN, LN+PL, or UN).
        '''
        pt = pdf_tools()
        typ_BIC = pt.get_BIC(sample, pars)
        fig, axs, rdsq_ln, rdsq_lp, flag_ln, flag_lp = pt.plt_npdf(file, low = low, model = 'both', par = pars)
        if ((flag_ln == 1) & (rdsq_ln<10) & (typ_BIC == 'LN')):
            tp = 'LN'
        elif ((flag_lp == 1) & (rdsq_lp<10) & (typ_BIC == 'LN+PL')):
            tp = 'LN+PL'
        else:
            tp = 'UN'
        axs.plot([pars[4], pars[4]], [1e-7, 100], color = 'royalblue', ls = '--')
        axs.text(0.97, 0.8,'BIC:'+typ_BIC, fontsize = 13, horizontalalignment='right',\
                verticalalignment='center', transform=axs.transAxes)
        lab = '%.2e' % rdsq_ln
        text = lab.split('e')[0]+r'$\times 10^{'+str(int(lab.split('e')[1]))+'}$'
        axs.text(0.97, 0.7, r'$\rm \chi^2_{\nu, LN}$:'+text+', '+str(flag_ln), fontsize = 13, horizontalalignment='right',\
                verticalalignment='center', transform=axs.transAxes)
        lab = '%.2e' % rdsq_lp
        text = lab.split('e')[0]+r'$\times 10^{'+str(int(lab.split('e')[1]))+'}$'
        axs.text(0.97, 0.6, r'$\rm \chi^2_{\nu, LN+PL}$:'+text+', '+str(flag_lp), fontsize = 13, horizontalalignment='right',\
                verticalalignment='center', transform=axs.transAxes)
        return fig, tp
    
