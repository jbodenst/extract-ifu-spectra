import numpy as np
import astropy.io.fits as fits


# prepare a header for a spectrum based on the cube's header
def prep_header(fitspath, star_id, x_pos, y_pos, ra, dec, mag_uv, mag_ir,
                num_stars):
    # load the reduced 3D data cube
    with fits.open(fitspath + 'DATACUBE_FINAL.fits') as fitsfile:
        # get the headers and the wcs from the input file
        header1 = fitsfile[0].header
        header2 = fitsfile[1].header
        # wcs_muse = WCS(fitsfile[1].header, naxis=2)

        # copy primary header from input file
        head = header1.copy()

        # update values from secondary header of input file
        wl0 = header2['CRVAL3']  # Starting wl at CRPIX1
        delt = header2['CD3_3']  # Stepwidth of wl
        pix = header2['CRPIX3']  # Reference Pixel

        # add keywords to header
        # use input Ra, Dec from HST for the header
        head.set('HIERARCH SPECTRUM STAR-ID', star_id,
                 'Star ID from input list')
        head.set('HIERARCH SPECTRUM X', x_pos, 'x pixel position')
        head.set('HIERARCH SPECTRUM Y', y_pos, 'y pixel position')
        head.set('HIERARCH SPECTRUM RA', ra, 'ra [deg]')
        head.set('HIERARCH SPECTRUM DEC', dec, 'dec [deg]')
        head.set('HIERARCH SPECTRUM UVMAG', mag_uv, 'UV magnitude, HST F336W')
        head.set('HIERARCH SPECTRUM IRMAG', mag_ir, 'IR magnitude, HST F884W')
        head.set('HIERARCH SPECTRUM NUM', num_stars,
                 'Number of considered stars')
        head.set('CRVAL1', wl0, 'Starting wavelength')
        head.set('CDELT1', delt, 'Wavelength step')
        head.set('CRPIX1', pix, 'Reference Pixel')
        return head


def write_extracted_spectrum(outfilename, header, flux, flux_err='None'):

    hdul_new = fits.HDUList()
    hdul_new.append(fits.PrimaryHDU(data=flux, header=header))
    if len(flux_err) > 1:
        hdul_new.append(fits.ImageHDU(data=flux_err))
    hdul_new.writeto(outfilename)

    print("Data written to %s" % outfilename)


def emission_gaussian(x, height, center, std):
    gauss = (height * np.exp(-1. * (x - center)**2 / (2.*std**2)))
    return gauss


def single_egauss(params, wavelength, fluxes):
    h = params['h']
    std = params['std']
    cen = params['cen']
    g = emission_gaussian(wavelength, h, cen, std)
    error = (fluxes - g)**2
    return error
