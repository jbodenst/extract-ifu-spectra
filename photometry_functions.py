import time
import os
from datetime import datetime
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import extract_stars
import traceback
from photutils import EPSFBuilder
import lmfit
from julia_utils import spec_functions as spec
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.visualization import simple_norm


# convenience function to simplify plotting
def plot_spatial(image, plotfname='spatial.pdf', coords1='None',
                 coords2='None', stars='None', mags='False', annotate='no',
                 interactive=False):
    fig, axes = plt.subplots(figsize=(14, 14))
    norm = simple_norm(image, 'sqrt', percent=90.)
    axes.imshow(image, aspect=1, origin='lower', cmap='Greys', norm=norm)

    if coords1 is not 'None':
        coords1_x, coords1_y = coords1[0], coords1[1]
        for i in range(len(coords1_x)):
            e = Circle(xy=(coords1_x[i], coords1_y[i]), radius=4)
            e.set_facecolor('none')
            e.set_edgecolor('deeppink')
            axes.add_artist(e)

    if coords2 is not 'None':
        coords2_x, coords2_y = coords2[0], coords2[1]
        for j in range(len(coords2_x)):
            e = Circle(xy=(coords2_x[j], coords2_y[j]), radius=3)
            e.set_facecolor('none')
            e.set_edgecolor('blue')
            axes.add_artist(e)

    if ((stars is not 'None') & (mags == 'False')):
        for star in stars:
            e = Circle(xy=(star.xcoord, star.ycoord), radius=3)
            e.set_facecolor('none')
            e.set_edgecolor('deeppink')
            if annotate == 'yes':
                axes.annotate(str(star.star_id), (star.xcoord, star.ycoord),
                              (5, 5), textcoords='offset points',
                              color='deeppink')
            axes.add_artist(e)
    if ((stars is not 'None') & (mags == 'True')):
        for star in stars:
            e = Circle(xy=(star.xcoord, star.ycoord),
                       radius=(18-star.ir_mag)*2)
            e.set_facecolor('none')
            e.set_edgecolor('C1')
            axes.annotate(str(star.star_id), (star.xcoord, star.ycoord),
                          (5, 5), textcoords='offset points', color='C1')
            axes.add_artist(e)
    axes.set_xlabel('x coordinate [px]')
    axes.set_ylabel('y coordinate [px]')

    fig.savefig(plotfname, bbox_inches='tight')

    if interactive == True:
        plt.show()


# fits the PSF based on hand-picked stars
def get_psf(data, starlist_x, starlist_y, do_plot='no', n_resample=4):
    # create table with x and y coordinates for the epsf stars
    stars_tbl = Table()
    stars_tbl['x'], stars_tbl['y'] = starlist_x, starlist_y

    # create bkg subtracted cutouts around stars
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)
    data -= median_val

    # extraction function requires data as NDData object
    nddata = NDData(data=data)

    # extract 30 x 30 px cutouts around the stars
    stars = extract_stars(nddata, stars_tbl, size=30)

    # build the EPSF from the selected stars
    epsf_builder = EPSFBuilder(oversampling=n_resample, maxiters=10)
    epsf, fitted_stars = epsf_builder(stars)

    # fit gaussian through PSF to estimate FWHM
    params = lmfit.Parameters()
    params.add('h', 0.006, min=0.004, max=0.01, vary=True)
    params.add('std', 5, min=1, max=10, vary=True)
    params.add('cen', 50, min=45, max=55, vary=True)

    len_x = len(epsf.data[0, :])
    x = np.linspace(0, 100, len_x)
    cutthrough = epsf.data[int(len(x)/2), :]
    minimizer = lmfit.Minimizer(spec.single_egauss, params,
                                fcn_args=(x, cutthrough))
    result = minimizer.minimize()

    gauss_std = result.params['std']
    h, cen = result.params['h'], result.params['cen']

    if do_plot == 'yes':
        fig_psf, ax_psf = plt.subplots(figsize=(8, 8))
        ax_psf.imshow(epsf.data, origin='lower', cmap='inferno')
        ax_psf.set_xlabel('x coordinate [px]')
        ax_psf.set_ylabel('y coordinate [px]')
        fig_psf.suptitle('3D PSF')
        fig_psf.savefig('PSF_2D.pdf', bbox_inches='tight')

        # plot cuts through the ePSF at different x positions
        len_x = len(epsf.data[0, :])
        x = np.linspace(0, 100, len_x)

        fig1, ax1 = plt.subplots(figsize=(8, 8))
        xvals = [int(len(x)/4), int(len(x)/2.1), int(len(x)/2)]
        for xval in xvals:
            cutthrough = epsf.data[xval, :]
            lab = 'x = ' + str(xval)
            ax1.plot(x, cutthrough, label=lab)

        g = spec.emission_gaussian(x, h, cen, gauss_std)
        ax1.plot(x, g, label='Gaussian fit')
        ax1.set_xlabel('x coordinate [px]')
        ax1.set_ylabel('EPSF normalized flux')
        ax1.legend()
        fig1.suptitle('2D cut-through of the PSF')
        fig1.savefig('PSF_cutthrough.pdf', bbox_inches='tight')

    # return epsf, std devition of gaussian fit and resampling parameter
    return epsf, gauss_std, n_resample


# perform PSF photometry for a target star with given surrounding stars
def do_phot_star(star_index, fits_path, pos, phot, logfile):
    flux_arr = []  # empty array, to be filled => spectrum
    flux_err_arr = []  # empty array, to be filled => error spectrum

    # load the reduced 3D data cube
    with fits.open(fits_path + 'DATACUBE_FINAL.fits') as fitsfile:
        cube = fitsfile[1].data

    # loop over each wavelength slice (image) in the cube
    for wave_index in range(cube.shape[0]):
        # for wave_index in range(1770, 1800):
        image = cube[wave_index, :, :]

        # do the actual photometry and get flux and fluxerr values
        result_tab = phot.do_photometry(image=image, init_guesses=pos)

        # only get first star in list which is star of interest
        flux_val = result_tab['flux_fit'][0]
        flux_err = result_tab['flux_unc'][0]

        flux_arr.append(flux_val)
        flux_err_arr.append(flux_err)
        logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' +
                      "Image [ %s / %s ] done \n" % (wave_index,
                                                     cube.shape[0]))
    logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' +
                  "Done with photometry for star number %i \n" % star_index)
    return flux_arr, flux_err_arr


# convenience function, not used at the moment => extract_spectra.py
def phot_per_star(final_starlist, obs_id, fits_path, phot, fitspath,
                  star_index):
    start_time = time.time()

    # select current star from master starlist
    star = final_starlist[star_index]

    # spec_min_mag = 17.5
    # if ((star.uv_mag <= spec_min_mag) &
    #     (star.xcoord > 0) & (star.xcoord < cube.shape[1]) &
    #         (star.ycoord > 0) & (star.ycoord < cube.shape[2])):
    filename = obs_id + '_id' + str(star.star_id)
    star.filename = filename

    # create a logfile
    logfilename = './' + star.filename + '.log'
    logfile = open(logfilename, 'w')

    if os.path.isfile(star.filename + '.fits'):
        logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' +
                      "Spectrum already extracted. Continue with next.\n")

    else:
        try:
            logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' +
                          "File does not exist, starting with extraction.\n")

            # star positions to consider when fitting the star of interest
            x_pos, y_pos = [], []

            # first: append the star of interest
            x_pos.append(star.xcoord), y_pos.append(star.ycoord)

            # append all stars that are close by (closer than crit_dist)
            crit_dist = 12.  # px

            # loop over stars in master starlist and find close-by stars
            for star2 in final_starlist:
                if star.xcoord != star2.xcoord:
                    if star2.find_closeby(star, crit_dist):
                        # if star is close by, get position
                        x, y = star2.xcoord, star2.ycoord
                        x_pos.append(x), y_pos.append(y)

            # create positions table to put into photometry
            pos = Table(names=['x_0', 'y_0'], data=[x_pos, y_pos])

            star.flux, star.flux_err = do_phot_star(star.star_id, fits_path,
                                                    pos, phot, logfile)
            # save obtained spectrum, pass MUSE infile for header infos
            logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' +
                          "Saving the spectrum now.\n")

            num_stars = len(pos) - 1
            star.save_spectrum(fitspath, num_stars)
        except Exception:  # as (errno, strerror):
            logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            traceback.print_exc(file=logfile)

    elapsed_time = (time.time() - start_time)
    logfile.write("Fitting required %f s." % elapsed_time)
    logfile.close()
