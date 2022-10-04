# MWISPclouds
Scripts for the analysis of the MWISP data (individual clouds) in the paper: 2022ApJS..262...16M. (https://iopscience.iop.org/article/10.3847/1538-4365/ac7797)   
- The methods for the calculation of the physical maps or parameters are described in detail in the above paper.  
- The N-PDF fittings in 2022ApJS..262...16M are implemented using methods in the pdf_tools.py.  
- The figures in 2022ApJS..262...16M are mostly generated using the functions in the img_tools.py.  

## 1. clouds.py 
> Calculate physical maps such as tex, optical depth, centroid velocity, and the 1st and 2nd moment maps for a cloud from the 12CO and 13CO MWISP data. 
> Get physical parameters of a cloud, such as mass, median and maximum Tex, median column denstiy, dispersion of normalized column density, median optical depth, column density detection limit, sonic Mach number, skewness and Kurtosis of the column density distribution.

### Contents:
- cal_tex: calculate the excitation temperature map from a given 12CO data cube.
- cal_tau13_n13: calculate the optical depth map and the H2 column density map from the 12CO and 13CO data under the local thermal equilibrium (LTE) assumption. Simultaneously, outputs a moment 0 map of the input 13CO data cube.
- cal_v: calculate the centroid velocity and velocity dispersion maps from a given 13CO data cube.
- cal_nlim: calculate the median reference detection limit for the derived H2 column density map.
- get_mass: calculate mass
- get_med_max_tex: calculate the median and the maximum tex for a given cloud.
- get_nh2: calculate the median H2 column density and the dispersion of the normalized H2 column density for a given cloud.
- get_tau: calculate the median optical depth for a given cloud.
- get_size: calculate the physical radius for a given cloud.
- get_ms_all: calculate the sonic Mach number for a given cloud.
- get_skw: calculate the skewness of the distribution of the H2 column density for a given cloud.
- get_kurt: calculate the kurtosis of the distribution of the H2 column density for a given cloud.

## 2. pdf_tools.py
> Methods for building an N-PDF from a given N_H2 fits file, fitting N-PDFs, testing the goodness of fit, classifying N-PDF shapes.

### Contents:
- log_prior_ln: logarithmic lognormal prior function for implementing the MCMC fitting, using as an input function for the calculation of posterior probability.
- pdf_ln: lognormal pdf function for plotting a modeled pdf curve and calculating the likelihood for the MCMC fitting process (an input function for the calculation of posterior probability).
- log_probability_ln: function for calculate the posterior probability, an input function for emcee.EnsembleSampler.
- log_prior_lp: logarithmic prior for LN+PL fitting.
- pdf_lp: pdf function for LN+PL fitting.
- log_probability_lp: input function for emcee.EnsembleSampler for LN+PL fitting.
- plt_npdf: plot an N-PDF for a given N_H2 fits file (users can decided whether to overlay fitted models).
- fit: method for implementing the MCMC fitting for a given H2 column density fits file with a selected model.
- get_BIC: implement model selection using BIC values among the best models of the LN and LN+PL fittings. 
- model_sel: implement model selection according to BIC, distribution of the residuals, and the reduced chi-squared values. 

## 3. img_tools.py
> Functions for visualizing the MWISP fits images, such as moment maps, p-v maps, velocity channel maps, and spectra grid overlying on a background image.

### Contents:

- plt_img: make a 2D plot. 
- plt_lvimg: make a p-v plot. 
- cov_hd: convert pixel range into physical range from a given fits header.
- cov_subregion: convert a given physical range into image pixel range according to a given fits header. 
- channelmap: make a velocity channel map for a given 3D fits cube.
- overlay_spec: overlay grids of average spectra on 2D projected images. The grid size of each average spectrum corresponds to the area from which the average spectrum is calculated.
- plt_nh2: especially for plotting a column density image.

## 4. crop_region.py and get_clouds.py
- crop_region: Extract individual 12CO clouds from a large data cube that contains masks of each cloud, i.e., the voxels within the same cloud are labeled with the same cloud ID. The extracted 12CO clouds are saved as small fits cubes along with the subregions of the RMS map.
- get_clouds: Extract individual 13CO clouds from the large 13CO data cube according to the PPV masks in the 12CO data. The large 12CO and 13CO data cubes are different in velocity channel width. The extracted 13CO clouds are saved as small fits cubes along with the subregions of the RMS map. The 13CO data in each small cube outside of a 12CO cloud mask are eliminated.
