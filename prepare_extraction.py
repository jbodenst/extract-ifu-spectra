import math
import pandas as pd
import numpy as np
import lmfit
from astropy.io import fits
from astropy.table import Table
from photutils.psf import DAOGroup
from photutils.psf import photometry
from astropy.modeling.fitting import LevMarLSQFitter
from julia_utils import photometry_functions as phot_funcs
from astropy.wcs import WCS


##############################################################################
# Definition of class star
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


##############################################################################
##############################################################################
# 1. load the data
##############################################################################
obs_id = 'Nov_2019_N1'  # !!!!
fits_path = '/STER/julia/data/MUSE/Nov2018_N1_notellurics/'  # !!!!

# load the reduced 3D data cube
with fits.open(fits_path + 'DATACUBE_FINAL.fits') as fitsfile:
    cube = fitsfile[1].data

# load the white-light image, used to fit the PSF because it has highest S/N
with fits.open(fits_path + 'IMAGE_FOV_0001.fits') as im_file:
    wl_image = im_file[1].data
    wcs_muse = WCS(im_file[1].header)

##############################################################################
# 2. definitions for the PSF fitting
##############################################################################
# PSF star list ( pixel positions)
psf_stars_x = [108.0]  # , 47.3, 223.9]  # !!!!
psf_stars_y = [74.4]  # , 235.7, 253.3]  # !!!!

epsf, gauss_std, n_resample = phot_funcs.get_psf(wl_image, psf_stars_x,
                                                 psf_stars_y, do_plot='yes')
aper_rad = 4 * gauss_std / n_resample

phot_psf = photometry.BasicPSFPhotometry(group_maker=DAOGroup(7.),
                                         psf_model=epsf,
                                         bkg_estimator=None,
                                         fitter=LevMarLSQFitter(),
                                         fitshape=(21),
                                         aperture_radius=aper_rad)

#############################################################################
# 3. Load the source positions from Milone's list
##############################################################################
print("Reading in the HST positions")

# read in masterlist with positions of stars to be considered (i.e. from HST)
hst_path = '/STER/julia/data/MUSE/extract_specs/'
cfile = hst_path + 'NGC0330_photometry_Milone_missingsources.csv'
coord_table = pd.read_csv(cfile)

ras, decs = coord_table['RA'], coord_table['DEC']

# create empty master list to which STAR objects will be added
hst_starlist = list()
min_mag = 18.5
id = 1

# loop over all stars in input list
for i in range(len(ras)):
    ra_hst, dec_hst = ras[i], decs[i]
    # HST data for larger region => presort only stars +/- 30px around the FoV
    if ((ra_hst < 14.105) & (ra_hst > 14.037) &
            (dec_hst > -72.474) & (dec_hst < -72.454)):

        # convert Ra, Dec to pixel coordinates in MUSE WCS
        x, y = wcs_muse.wcs_world2pix(ra_hst, dec_hst, 0)

        uv_mag, ir_mag = coord_table['F336W'][i], coord_table['F814W'][i]

        # apply magnitude cut
        if uv_mag < min_mag:
            s = Star(id, x, y, ra_hst, dec_hst, uv_mag=uv_mag, ir_mag=ir_mag)
            hst_starlist.append(s)
            id = id + 1
print("%i stars to consider." % len(hst_starlist))

# plot image with sources from input catalogue
phot_funcs.plot_spatial(wl_image, plotfname='input.pdf', stars=hst_starlist)

##############################################################################
# 4. properly align the HST coordinate systems to the MUSE coordinate system
##############################################################################
# pixel coordinates of stars that are used for determining the coordinate shift
shift_stars_x = [31.28, 41.57, 202.37, 297.79, 282.01, 259.29, 82.41, 17.09,
                 76.65]  # !!!!
shift_stars_y = [35.23, 89.55, 62.67, 46.61, 230.16, 285.04, 295.42, 144.33,
                 81.97]  # !!!!

pos = Table(names=['x_0', 'y_0'], data=[shift_stars_x, shift_stars_y])

# determine their positions in MUSE image by fitting their PSFs
result_tab = phot_psf.do_photometry(image=wl_image, init_guesses=pos)
shift_phot_x = [i for i in result_tab['x_fit']]
shift_phot_y = [i for i in result_tab['y_fit']]

# find closest HST star
shift_starlist = [0] * len(shift_stars_x)  # array of length of shift_stars_x
for i in range(len(shift_phot_x)):
    distance = 1000.
    x_muse, y_muse = shift_phot_x[i], shift_phot_y[i]
    for star in hst_starlist:
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
for star in hst_starlist:
    new_x, new_y = trans_rot(star.xcoord, star.ycoord, x_shift, y_shift, theta)
    star.xcoord, star.ycoord = new_x, new_y

phot_funcs.plot_spatial(wl_image, plotfname='adjusted_coords.pdf',
                        stars=hst_starlist)

##############################################################################
# 5. make the final target list
##############################################################################
# sort stars by x and y coordinate
hst_starlist.sort(key=lambda x: x.xcoord)
hst_starlist.sort(key=lambda x: x.ycoord)

# write an output file with all the targets that will be considered
filename = 'inputlist_hstpos_f336_NovN1.csv'  # !!!!
f = open(filename, 'w')
f.write('id,x,y,ra,dec,f336_mag,f814_mag' + '\n')

# also write one output file for each star a spectrum should be extracted
spec_min_mag = 17.5
#  crit_dist: stars within this radius are considered when fitting the PSF
crit_dist = 12.  # px
max_star_number = 17

for star in hst_starlist:
    # first: get all stars in master starlist ( < 18.5 )
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

        # now: write file per star a spectrum is supposed to be extracted
        # only take stars brighter than i.e. 17.5 and inside FoV
        if ((star.uv_mag <= spec_min_mag) &
            (star.xcoord > 0) & (star.xcoord < cube.shape[1]) &
                (star.ycoord > 0) & (star.ycoord < cube.shape[2])):

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
            for star2 in hst_starlist:
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
f.close()
print("Positions saved to file %s" % str(filename))

# plot the stars with star ids and circle sizes ~ magnitude
phot_funcs.plot_spatial(wl_image, plotfname='final_inputlist.pdf',
                        stars=hst_starlist, mags='True')


print('Done preparing the extraction!')
