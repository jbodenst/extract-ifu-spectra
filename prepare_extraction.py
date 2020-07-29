from astropy.io import fits
from julia_utils import photometry_functions as phot_funcs
from astropy.wcs import WCS

"""
Convenience function to prepare input file required by extract_spectra.py.
Assumes that table with HST photometry exists (columns: RA, DEC, F336W, F814W)
Adjusts the coordinate systems between HST and MUSE data. Requires the
following defintions:
spec_min_mag: faintest star for which a spectrum should be extracted
min_mag: faintest star that should be considered in the extraction of a
         spectrum of another star
crit_dist: pixel distance to a second star up to which it is considered in the
           PSF fitting
max_star_number: maximum number of stars that are considered for one extraction
                 (important for the computation time of the extraction!)
"""
##############################################################################
# 1. load the MUSE white-light image
##############################################################################
fits_path = '/STER/julia/data/MUSE/Nov2018_N1_notellurics/'  # !!!!

# load the white-light image, used to fit the PSF because it has highest S/N
with fits.open(fits_path + 'IMAGE_FOV_0001.fits') as im_file:
    wl_image = im_file[1].data
    wcs_muse = WCS(im_file[1].header)


#############################################################################
# 2. Load the star positions from the HST photometry list
##############################################################################
print("Reading in the HST positions")

# read in masterlist with positions of stars to be considered (i.e. from HST)
hst_path = '/STER/julia/data/MUSE/extract_specs/'  # !!!!!
cfile = hst_path + 'NGC0330_photometry_Milone_missingsources.csv'  # !!!!!

# faintest star for which a spectrum should be extracted
spec_min_mag = 17.5  # !!!!!
# faintest star that should still be considered in extraction of other stars
min_mag = 18.5  # !!!!!

# define region around MUSE FoV (HST typically much larger => too many stars)
ra_l, ra_u = 14.037, 14.105
dec_l, dec_u = -72.474, -72.454

# loop over all stars in input list and append them to the starlist if they are
# .. inside the FoV and brighter than min_mag
hst_starlist = phot_funcs.prep_indata(cfile, min_mag, wl_image, wcs_muse,
                                      ra_l, ra_u, dec_l, dec_u)

##############################################################################
# 3. align the HST coordinate systems to the MUSE coordinate system
##############################################################################
# pixel coordinates of reference stars are used to determine coordinate trafo
shift_stars_x = [31.28, 41.57, 202.37, 297.79, 282.01, 259.29, 82.41, 17.09,
                 76.65]  # !!!!
shift_stars_y = [35.23, 89.55, 62.67, 46.61, 230.16, 285.04, 295.42, 144.33,
                 81.97]  # !!!!

# PSF star list (pixel positions)
# list of stars in MUSE FoV that are used to fit the psf => bright & isolated
psf_stars_x = [108.0]  # , 47.3, 223.9]  # !!!!
psf_stars_y = [74.4]  # , 235.7, 253.3]  # !!!!

hst_starlist = phot_funcs.align_coordsystems(hst_starlist, psf_stars_x,
                                             psf_stars_y, shift_stars_x,
                                             shift_stars_y, wl_image)

##############################################################################
# 4. make the final target list
##############################################################################
# write an output file with all the targets that will be considered
filename = 'inputlist_hstpos_f336_NovN1.csv'  # !!!!!

#  crit_dist: stars within this radius are considered when fitting the PSF
crit_dist = 12.  # px
max_star_number = 17  # maximum number of stars that are considered together


phot_funcs.write_all_star_file(filename, hst_starlist)

phot_funcs.write_infile_per_star(hst_starlist, wl_image, spec_min_mag,
                                 crit_dist, max_star_number,
                                 obs_id='Nov_2019_N1')

# plot the stars with star ids and circle sizes ~ magnitude
phot_funcs.plot_spatial(wl_image, plotfname='final_inputlist.pdf',
                        stars=hst_starlist, mags='True')


print('Done preparing the extraction!')
