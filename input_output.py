import numpy as np
import astropy.io.fits as fits
from scipy import interpolate
import sys
from astropy.table import Table


def read_file(infile):
    ext = str(infile.split('.')[-1])

    # Check type of input file (fits or ascii) to read in data correctly
    if (ext == 'fits'):
        wave, flux = read_fits(infile)

    elif (ext == 'mt'):
        wave, flux = read_fits(infile)

    elif (ext == 'gz'):
        wave, flux = read_tlusty(infile)

    elif (ext == 'dat' or ext == 'ascii' or ext == 'txt' or ext == 'nspec'):
        wave, flux = read_ascii(infile)

    elif (ext == 'tfits'):
        wave, flux = read_uvespop(infile)

    elif (ext == 'hfits'):
        wave, flux = read_hermes_normalized(infile)

    elif (ext == 'rgs'):
        wave, flux = read_gssp_synthspec(infile)
    else:
        print("ERROR: Could not read the input file - unknown extension.")
        sys.exit()

    return wave, flux


def read_fits(infile):
    # print("%s: Input file is a fits file." % infile)

    header = fits.getheader(infile)

    if 'HIERARCH SPECTRUM STAR-ID' in header:
        wave, flux = read_psfSpec(infile)

    elif 'INSTRUME' in header:
        ins = header['INSTRUME']

        if (ins == 'MUSE'):
            wave, flux = read_pampelMUSE(infile)

        elif (ins == 'HERMES'):
            wave, flux = read_HERMES(infile)

        elif (ins == 'FEROS'):
            wave, flux = read_FEROS(infile)

        elif (ins == 'UVES'):
            wave, flux = read_UVES(infile)

        elif (ins == 'HRS'):
            wave, flux = read_SALT(infile)

        else:  # i.e. BeSS spectra
            wave, flux = read_psfSpec(infile)

    else:
        wave, flux = read_psfSpec(infile)

    return wave, flux


def read_psfSpec(infile):
    print("%s: Input file is a PSF extracted file." % infile)
    try:
        header = fits.getheader(infile)
        flux = fits.getdata(infile, ext=0)
        err = fits.getdata(infile, ext=1)
        wl0 = header['CRVAL1']  # Starting wl at CRPIX1
        delt = header['CDELT1']  # Stepwidth of wl
        pix = header['CRPIX1']  # Reference Pixel
        wave = wl0 - (delt * pix - delt) + np.arange(flux.shape[0]) * delt
        flux_out = flux, err
    except IndexError:
        header = fits.getheader(infile)
        flux = fits.getdata(infile)
        wl0 = header['CRVAL1']  # Starting wl at CRPIX1
        delt = header['CDELT1']  # Stepwidth of wl
        pix = header['CRPIX1']  # Reference Pixel
        wave = wl0 - (delt * pix - delt) + np.arange(flux.shape[0]) * delt
        flux_out = flux

    return wave, flux_out


def read_pampelMUSE(infile):
    print("%s: Input file is a pampelMUSE file." % infile)
    header = fits.getheader(infile)
    flux = fits.getdata(infile, ext=0)
    try:
        err = fits.getdata(infile, ext=1)
    except IndexError:
        err = []
    wl0 = header['CRVAL1']  # Starting wl at CRPIX1
    delt = header['CDELT1']  # Stepwidth of wl
    pix = header['CRPIX1']  # Reference Pixel
    wave = wl0 - (delt * pix - delt) + np.arange(flux.shape[0]) * delt

    # because pampelmuse gives the wl in m --> convert to A
    wave = wave * 10**10.
    flux_out = flux, err

    return wave, flux_out


def read_MUSE(infile):
    print("%s: Input file is a MUSE file." % infile)
    header = fits.getheader(infile)
    data = fits.getdata(infile)
    wl0 = header['CRVAL1']  # Starting wl at CRPIX1
    delt = header['CDELT1']  # Stepwidth of wl
    pix = header['CRPIX1']  # Reference Pixel
    wave = wl0 - (delt * pix - delt) + np.arange(data.shape[0]) * delt

    flux = data

    return wave, flux


def read_HERMES(infile):
    print("%s: Input file is a HERMES file." % infile)
    header = fits.getheader(infile)
    # for files with standard wavelegth array
    if ((header['CTYPE1'] == 'WAVELENGTH') or (header['CTYPE1'] == 'AWAV')):
        print('standard')
        flux = fits.getdata(infile)
        crval = header['CRVAL1']
        cdelt = header['CDELT1']
        naxis1 = header['NAXIS1']
        wave = crval + np.arange(0, naxis1) * cdelt

    # for files that are given in logarithmic wl array
    if (header['CTYPE1'] == 'log(wavelength)'):
        flux = fits.getdata(infile)
        crval = header['CRVAL1']
        cdelt = header['CDELT1']
        naxis1 = header['NAXIS1']
        wave = np.exp(crval + np.arange(0, naxis1)*cdelt)
        print(len(wave))

    else:
        print("Could not read in HERMES fits file - unknown file type.")
        sys.exit()

    return wave, flux


def read_FEROS(infile):
    print("%s: Input file is a FEROS file." % infile)

    if "norm" in str(infile):
        header = fits.getheader(infile)
        if 'CRVAL1' in header:
            wave, flux = read_MUSE(infile)
        else:
            wlmin = header['WAVELMIN']
            wlmax = header['WAVELMAX']
            spec_bin = header['SPEC_BIN']

            wave = np.arange(wlmin, wlmax+spec_bin, spec_bin) * 10  # conversion to Angstrom
            flux = fits.getdata(infile)

    else:

        header = fits.getheader(infile)

        if 'CRVAL1' in header:
            print('here')
            flux = fits.getdata(infile)
            crval = header['CRVAL1']
            crpix = header['CRPIX1']
            cdelt = header['CDELT1']

            wave = crval - (cdelt * crpix -
                            cdelt) + np.arange(flux.shape[0]) * cdelt

        else:
            data = fits.getdata(infile)
            wave = data.field(0)[0]
            flux = data.field(1)[0]
            # err = data.field(2)[0]
    return wave, flux


def read_UVES(infile):
    print("%s: Input file is a UVES file." % infile)
    header = fits.getheader(infile)
    if 'SPEC_COM' in header:
        table = Table.read(infile, hdu=1)
        wave = table['wave']
        flux = table['flux']
    elif 'CRVAL1' in header:
        flux = fits.getdata(infile)
        crval = header['CRVAL1']
        cdelt = header['CDELT1']
        naxis1 = header['NAXIS1']
        wave = crval + np.arange(0, naxis1) * cdelt
    else:
        data = fits.getdata(infile)
        wave = data.field(0)[0]
        flux = data.field(1)[0]
        # err = data.field(2)[0]
    return wave, flux


def read_SALT(infile):
    hdus = fits.open(infile)

    flux = hdus[0].data
    head = hdus[0].header
    crpix = head.get('CRPIX1')-1
    n_points = head['NAXIS1']
    delta_w = head['CDELT1']
    start_w = head['CRVAL1'] - crpix * delta_w
    wave = np.linspace(start_w, start_w+(delta_w * (n_points-1)), n_points)

    return wave, flux


def read_gssp_synthspec(infile):
    wave, flux, a, b = np.loadtxt(infile, unpack=True)

    return wave, flux


def read_tlusty(infile):
    fill_val = 'extrapolate'
    print("%s: Input file is a TLUSTY model." % infile)

    # first the model
    wave, flux = np.loadtxt(infile, unpack=True)

    # wave array not evenly spaced => interpolate it
    s = interpolate.interp1d(wave, flux, 2, fill_value=fill_val)
    flux = s(wave)

    # now the continuum
    contfile = infile.split('.7.')[0] + '.17.gz'
    wave_cont, flux_cont = np.loadtxt(contfile, unpack=True)

    # wave array not evenly spaced => interpolate it
    s = interpolate.interp1d(wave_cont, flux_cont, 1, fill_value=fill_val)
    flux_cont = s(wave)

    # normalize the model
    flux = flux / flux_cont

    return wave, flux


def read_uvespop(infile):
    print("%s: Input file is a UVESPOP file." % infile)
    data = fits.getdata(infile)
    wave = data.field(0)
    flux = data.field(1)

    return wave, flux


def read_hermes_normalized(infile):
    print("%s: Input file is a normalized HERMES file." % infile)
    data = fits.getdata(infile)
    wave = data.field(0)
    # flux = data.field(1)
    norm = data.field(2)
    # cont = data.field(3)

    return wave, norm


def read_ascii(infile):
    # any type of ascii file (typically I call them .dat)
    # assumes that first column is wave and second column is flux
    print("%s: Input file is an ascii file." % infile)
    wave, flux = np.loadtxt(infile, usecols=(0, 1), unpack=True)

    return wave, flux


def cut_muse_specrange(wave, flux):
    # cut the spectrum to MUSE wavelength
    spec_range = ((wave > 4600) & (wave < 9350))
    wave = wave[spec_range]
    flux = flux[spec_range]

    return wave, flux


def mask_muse_laser(wave, flux):
    index = [j for j in range(len(wave)) if 5746.0 < wave[j] < 6017.]
    flux[index] = np.nan

    return wave, flux


def write_spectrum(infile, flux, err, snr_b, snr_g, snr_r, outfile,
                   snrs=True, errs=True):
    hdul_infile = fits.open(infile)
    hdul_new = fits.HDUList()
    primheader = hdul_infile[0].header.copy()

    if snrs is True:
        primheader.set('HIERARCH SPECTRUM SNR-B', snr_b, 'SNR at 5325 A')
        primheader.set('HIERARCH SPECTRUM SNR-G', snr_g, 'SNR at 6405 A')
        primheader.set('HIERARCH SPECTRUM SNR-R', snr_r, 'SNR at 7475 A')

    hdul_new.append(fits.PrimaryHDU(data=flux, header=primheader))

    if errs is True:
        try:
            secondheader = hdul_infile[1].header.copy()
        except IndexError:  # file is a qfits file without 2nd header
            # get generic second header from a random photutils file
            new_infile = ('/lhome/julia/data/NGC330/MUSE/other_epochs/' +
                          'Sep_2018_N1/specs/Sep_2018_N1_id92.fits')

            secondheader = fits.open(new_infile)[1].header.copy()
        hdul_new.append(fits.ImageHDU(data=err, header=secondheader))
    hdul_new.writeto(outfile)

    print("Data written to %s" % outfile)


def write_hermes(infile, flux_cont, outfile):
    hdul_infile = fits.open(infile)
    hdul_new = fits.HDUList()
    primheader = hdul_infile[0].header.copy()

    hdul_new.append(fits.PrimaryHDU(data=flux_cont, header=primheader))
    hdul_new.writeto(outfile)

    print("Data written to %s" % outfile)


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


def write_2Dimage(header, image, outfilename):
    hdul_new = fits.HDUList()
    hdul_new.append(fits.PrimaryHDU(data=image, header=header))
    hdul_new.writeto(outfilename)
