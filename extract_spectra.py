import pandas as pd
import sys
import os
import time
import traceback
from datetime import datetime
from astropy.io import fits
from astropy.table import Table
from photutils.psf import DAOGroup
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.psf.photometry import BasicPSFPhotometry
from julia_utils import photometry_functions as phot_funcs


start_time = time.time()
# PSF star list ( pixel positions)
psf_stars_x = [108.0]  # , 47.3, 223.9]  # !!!!
psf_stars_y = [74.4]  # , 235.7, 253.3]  # !!!!

# Load input file containing star positions ( => prepare_extraction.py )
try:
    cfile = sys.argv[1]
except IndexError:
    print('No input file containting star positions was given')
    exit()


# path to data
fits_path = '/STER/julia/data/MUSE/Nov2018_N1_notellurics/'  # !!!!

# base filename to construct logfile and fitsfile name
fname = cfile.split('.input')[0]
outfolder = './'

logfilename = outfolder + fname + '.log'
logfile = open(logfilename, 'w', buffering=1)

# check if spectrum was already extracted
if os.path.isfile(outfolder + fname + '.fits'):
    logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' +
                  "Spectrum already extracted. Continue with next.\n")
# if the spectrum does not exist yet: extract it
else:
    try:
        logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' +
                      "File does not exist, starting with extraction.\n")

        # load the reduced 3D data cube
        with fits.open(fits_path + 'DATACUBE_FINAL.fits') as fitsfile:
            cube = fitsfile[1].data

        # load white-light image to fit the PSF => highest S/N
        with fits.open(fits_path + 'IMAGE_FOV_0001.fits') as im_file:
            wl_image = im_file[1].data

        # definitions for the PSF fitting
        epsf, gauss_std, n_resample = phot_funcs.get_psf(wl_image, psf_stars_x,
                                                         psf_stars_y)
        # aperture radius used for first flux guess
        aper_rad = 4 * gauss_std / n_resample

        # fix the positions to only fit the flux of the star, not its position
        epsf.x_0.fixed = True
        epsf.y_0.fixed = True

        phot = BasicPSFPhotometry(group_maker=DAOGroup(15.),
                                  psf_model=epsf,
                                  bkg_estimator=None,
                                  fitter=LevMarLSQFitter(),
                                  fitshape=(17),
                                  aperture_radius=aper_rad)

        # IDs and positions from input file
        ctable = pd.read_csv(cfile)
        ids, xpos, ypos = ctable['id'], ctable['x'], ctable['y']
        ras, decs = ctable['ra'], ctable['dec']
        uv_mags, ir_mags = ctable['f336_mag'], ctable['f814_mag']

        # star to extract spectrum for (first one in the input file)
        star = phot_funcs.Star(ids[0], xpos[0], ypos[0], ras[0], decs[0],
                               uv_mags[0], ir_mags[0],
                               fname=(outfolder + fname))

        # star positions to consider when fitting the star of interest
        x_pos, y_pos = [], []

        # loop over all stars in input list, first one is star to consider!
        for i in range(len(ids)):
            x_pos.append(xpos[i]), y_pos.append(ypos[i])

        # create positions table to put into photometry
        pos = Table(names=['x_0', 'y_0'], data=[xpos, ypos])

        # number of stars to consider in PSF fitting
        num_stars = len(pos) - 1

        logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' +
                      "%i stars to consider in PSF fitting.\n" % num_stars)

        # Do the PSF photometry
        star.flux, star.flux_err = phot_funcs.do_phot_star(star.star_id,
                                                           fits_path, pos,
                                                           phot, logfile)

        # save obtained spectrum, pass MUSE infile for header infos
        logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' +
                      "Saving the spectrum now.\n")

        star.save_spectrum(fits_path, num_stars)

    # catch any error message and write it to output file
    except Exception:
        logfile.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        traceback.print_exc(file=logfile)

elapsed_time = (time.time() - start_time)
logfile.write("Fitting required %f s." % elapsed_time)
logfile.close()
