import math
import numpy as np
import lmfit
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.modeling.fitting import LevMarLSQFitter
from matplotlib.patches import Circle
from photutils import EPSFBuilder
from photutils.psf import extract_stars
from photutils.psf import photometry
from photutils.psf import DAOGroup
import misc


##############################################################################
# Definition of class Star
##############################################################################
class Star:
    def __init__(self, id, xcoord, ycoord, ra, dec, uv_mag, ir_mag,
                 fname='None'):
        self.star_id = id  # star id
        self.xcoord = xcoord  # pixel x coordinate
        self.ycoord = ycoord  # pixel y coordinate
        self.ra = ra  # deg, from HST input list, not shift
        self.dec = dec  # deg, from HST input list, not shifted
        self.uv_mag = uv_mag  # uv magnitude from input list
        self.ir_mag = ir_mag  # ir magnitude from input list
        self.flux = []  # flux ( to be filled )
        self.flux_err = []  # err ( to be filled )
        self.filename = fname

    def find_closeby(self, star2, crit_dist):
        # crit_dist: distance [px] < 2 stars are thought to considered together
        distance = ((self.xcoord - star2.xcoord)**2 +
                    (self.ycoord - star2.ycoord)**2)**0.5

        if distance < crit_dist:
            return True
        else:
            return False

    # save the spectrum of the star
    def save_spectrum(self, fitspath, num_stars):
        # prepare the header
        header = misc.prep_header(fitspath, self.star_id, self.xcoord,
                                  self.ycoord, self.ra, self.dec, self.uv_mag,
                                  self.ir_mag, num_stars)
        #  write out the spectrum
        outfilename = self.filename + '.fits'
        misc.write_extracted_spectrum(outfilename, header, self.flux,
                                      self.flux_err)


# convenience function to simplify plotting
def plot_spatial(image, plotfname='spatial.pdf', coords1='None',
                 coords2='None', stars='None', mags='False', annotate='no',
                 interactive=False):
    fig, axes = plt.subplots(figsize=(14, 14))
    norm = simple_norm(image, 'sqrt', percent=90.)
    axes.imshow(image, aspect=1, origin='lower', cmap='Greys', norm=norm)

    if coords1 != 'None':
        coords1_x, coords1_y = coords1[0], coords1[1]
        for i in range(len(coords1_x)):
            e = Circle(xy=(coords1_x[i], coords1_y[i]), radius=4)
            e.set_facecolor('none')
            e.set_edgecolor('deeppink')
            axes.add_artist(e)

    if coords2 != 'None':
        coords2_x, coords2_y = coords2[0], coords2[1]
        for j in range(len(coords2_x)):
            e = Circle(xy=(coords2_x[j], coords2_y[j]), radius=3)
            e.set_facecolor('none')
            e.set_edgecolor('blue')
            axes.add_artist(e)

    if ((stars != 'None') & (mags == 'False')):
        for star in stars:
            e = Circle(xy=(star.xcoord, star.ycoord), radius=3)
            e.set_facecolor('none')
            e.set_edgecolor('deeppink')
            if annotate == 'yes':
                axes.annotate(str(star.star_id), (star.xcoord, star.ycoord),
                              (5, 5), textcoords='offset points',
                              color='deeppink')
            axes.add_artist(e)
    if ((stars != 'None') & (mags == 'True')):
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

    if interactive:
        plt.show()


def align_coordsystems(starlist, psf_stars_x, psf_stars_y,
                       shift_stars_x, shift_stars_y, wl_image, plot_flag=True):
    epsf, gauss_std, n_resample = get_psf(wl_image, psf_stars_x,
                                          psf_stars_y, do_plot='yes')
    aper_rad = 4 * gauss_std / n_resample

    phot_psf = photometry.BasicPSFPhotometry(group_maker=DAOGroup(7.),
                                             psf_model=epsf,
                                             bkg_estimator=None,
                                             fitter=LevMarLSQFitter(),
                                             fitshape=(21),
                                             aperture_radius=aper_rad)

    pos = Table(names=['x_0', 'y_0'], data=[shift_stars_x, shift_stars_y])

    # determine their positions in MUSE image by fitting their PSFs
    result_tab = phot_psf.do_photometry(image=wl_image, init_guesses=pos)
    shift_phot_x = [i for i in result_tab['x_fit']]
    shift_phot_y = [i for i in result_tab['y_fit']]

    # find closest HST star
    shift_starlist = [0] * len(shift_stars_x)

    for i in range(len(shift_phot_x)):
        distance = 1000.
        x_muse, y_muse = shift_phot_x[i], shift_phot_y[i]
        for star in starlist:
            dist = ((x_muse - star.xcoord)**2 + (y_muse - star.ycoord)**2)**0.5
            if dist < distance:
                shift_starlist[i] = star
                distance = dist

    # get array of coordinates
    x_hst_list = [i.xcoord for i in shift_starlist]
    y_hst_list = [i.ycoord for i in shift_starlist]

    # coordinate transformation parameters including translation and rotation
    params = lmfit.Parameters()
    params.add('delta_x', 1., min=-10, max=10, vary=True)
    params.add('delta_y', 1., min=-10, max=10, vary=True)
    params.add('theta', 0., min=-5., max=5., vary=True)

    # shift the stars and minimize their distance
    minimizer = lmfit.Minimizer(get_coordshift, params,
                                fcn_args=(shift_phot_x, shift_phot_y,
                                          x_hst_list, y_hst_list))
    result = minimizer.minimize()
    print("Done with fitting the coordinate shift.")

    # best-fit parameters
    x_shift = result.params['delta_x']
    y_shift = result.params['delta_y']
    theta = result.params['theta']

    # adjust the coordinates of all stars
    for star in starlist:
        new_x, new_y = trans_rot(star.xcoord, star.ycoord,
                                 x_shift, y_shift, theta)
        star.xcoord, star.ycoord = new_x, new_y

    if plot_flag:
        plot_spatial(wl_image, plotfname='adjusted_coords.pdf', stars=starlist)

    # sort stars by x and y coordinate
    starlist.sort(key=lambda x: x.xcoord)
    starlist.sort(key=lambda x: x.ycoord)

    return starlist


# coordinate transformation applying translation and rotation
def trans_rot(x, y, deltax, deltay, theta_deg):
    theta_rad = math.radians(theta_deg)
    new_x = x * np.cos(theta_rad) + y * np.sin(theta_rad) + deltax
    new_y = -x * np.sin(theta_rad) + y * np.cos(theta_rad) + deltay
    return new_x, new_y


# calculate distance between two sets of stars (in different coord. systems)
def get_coordshift(params, stars1_xarr, stars1_yarr, stars2_xarr, stars2_yarr):
    delta_x = params['delta_x']
    delta_y = params['delta_y']
    theta = params['theta']

    dists = []

    # loop over stars in arrays to obtain distance between stars
    for i in range(len(stars1_xarr)):
        transform_star2_x, transform_star2_y = trans_rot(stars2_xarr[i],
                                                         stars2_yarr[i],
                                                         delta_x, delta_y,
                                                         theta)

        dist = ((stars1_xarr[i] - (transform_star2_x))**2 +
                (stars1_yarr[i] - (transform_star2_y))**2)**0.5
        dists.append(dist)

    return dists


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
    minimizer = lmfit.Minimizer(misc.single_egauss, params,
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

        g = misc.emission_gaussian(x, h, cen, gauss_std)
        ax1.plot(x, g, label='Gaussian fit')
        ax1.set_xlabel('x coordinate [px]')
        ax1.set_ylabel('EPSF normalized flux')
        ax1.legend()
        fig1.suptitle('2D cut-through of the PSF')
        fig1.savefig('PSF_cutthrough.pdf', bbox_inches='tight')

    # return epsf, std devition of gaussian fit and resampling parameter
    return epsf, gauss_std, n_resample


def prep_indata(cfile, min_mag, wl_image, wcs_muse,
                ra_min=0, ra_max=23.99, dec_min=-90, dec_max=90):

    coord_table = pd.read_csv(cfile)

    # create empty master list to which STAR objects will be added
    starlist = list()

    id = 1

    ras, decs = coord_table['RA'], coord_table['DEC']
    for i in range(len(ras)):
        ra_hst, dec_hst = ras[i], decs[i]

        # HST data for larger FoV => presort only stars +/- 30px around the FoV
        if ((ra_hst > ra_min) & (ra_hst < ra_max) &
                (dec_hst > dec_min) & (dec_hst < dec_max)):

            # convert Ra, Dec to pixel coordinates in MUSE WCS
            x, y = wcs_muse.wcs_world2pix(ra_hst, dec_hst, 0)

            uv_mag, ir_mag = coord_table['F336W'][i], coord_table['F814W'][i]

            # apply magnitude cut
            if uv_mag < min_mag:
                s = Star(id, x, y, ra_hst, dec_hst, uv_mag=uv_mag,
                         ir_mag=ir_mag)
                starlist.append(s)
                id = id + 1
    print("%i stars to consider." % len(starlist))

    # plot image with sources from input catalogue
    plot_spatial(wl_image, plotfname='input.pdf', stars=starlist)

    return starlist


def write_all_star_file(outfilename, starlist):
    f = open(outfilename, 'w')
    f.write('id,x,y,ra,dec,f336_mag,f814_mag' + '\n')
    for star in starlist:
        # first: get all stars in master starlist ( < 18.5 ), outfile: f
        if star.uv_mag != 'None':
            # append line to general output file
            line = ('{0:03d}'.format(star.star_id) + ',' +
                    '{:.6f}'.format(star.xcoord) + ',' +
                    '{:.6f}'.format(star.ycoord) + ',' +
                    '{:.6f}'.format(star.ra) + ',' +
                    '{:.6f}'.format(star.dec) + ',' +
                    '{:.3f}'.format(star.uv_mag) + ',' +
                    '{:.3f}'.format(star.ir_mag) + '\n')
            f.write(line)
    f.close()
    print("Positions saved to file %s" % str(outfilename))


def write_infile_per_star(starlist, wl_image, spec_min_mag=17.5, crit_dist=12,
                          max_star_number=17, obs_id=''):
    # write one output file for each star a spectrum should be extracted
    for star in starlist:
        # first: get all stars in master starlist ( < 18.5 ), outfile: f
        if star.uv_mag != 'None':
            # now: write file per star a spectrum is supposed to be extracted
            # only take stars brighter than i.e. 17.5 and inside FoV
            # make one file per star: starf
            if ((star.uv_mag <= spec_min_mag) &
                (star.xcoord > 0) & (star.xcoord < wl_image.shape[0]) &
                    (star.ycoord > 0) & (star.ycoord < wl_image.shape[1])):

                # define filename of input file
                starfname = obs_id + '_id' + str(star.star_id) + '.input'
                starf = open(starfname, 'w')
                starf.write('id,x,y,ra,dec,f336_mag,f814_mag' + '\n')

                # write line for the star to consider
                starf.write('{0:03d}'.format(star.star_id) + ',' +
                            '{:.6f}'.format(star.xcoord) + ',' +
                            '{:.6f}'.format(star.ycoord) + ',' +
                            '{:.6f}'.format(star.ra) + ',' +
                            '{:.6f}'.format(star.dec) + ',' +
                            '{:.3f}'.format(star.uv_mag) + ',' +
                            '{:.3f}'.format(star.ir_mag) + '\n')

                # loop over stars in master starlist and find close-by stars
                stars_to_consider = list()
                for star2 in starlist:
                    if star.xcoord != star2.xcoord:  # not the same star
                        # if star is close by, get position
                        if star2.find_closeby(star, crit_dist):
                            stars_to_consider.append(star2)

                # sort by UV mag in order to only take brightest stars
                stars_to_consider.sort(key=lambda x: x.uv_mag)

                count = 0
                for star2 in stars_to_consider:
                    if count < max_star_number:
                        count += 1
                        starf.write('{0:03d}'.format(star2.star_id) + ',' +
                                    '{:.6f}'.format(star2.xcoord) + ',' +
                                    '{:.6f}'.format(star2.ycoord) + ',' +
                                    '{:.6f}'.format(star2.ra) + ',' +
                                    '{:.6f}'.format(star2.dec) + ',' +
                                    '{:.3f}'.format(star2.uv_mag) + ',' +
                                    '{:.3f}'.format(star2.ir_mag) + '\n')
                starf.close()


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
