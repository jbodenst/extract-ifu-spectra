import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.special import erf
from scipy.integrate import simps
import lmfit
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

cc = 299792.458  # km/s


def rebin(wave, flux, err=False, stepwidth=1.25):
    # interpolates from old wavelength (wave) to a new wavelength array
    # wl_rebin with new stephwidth
    print('Rebinning the given arrays to a new stepwidth of %f A.' % stepwidth)

    # define new array based on given stepwidth
    wl_start = wave[0]  # define start
    wl_end = wave[-1]  # and end point of new wave array

    wl_rebin = np.arange(wl_start, wl_end+stepwidth, stepwidth)

    # do the interpolation
    intFunc = interp1d(wave, flux, kind="slinear", fill_value='extrapolate')
    fl_rebin = np.array(intFunc(wl_rebin))

    if err is not False:
        errFunc = interp1d(wave, err, kind="slinear", fill_value='extrapolate')
        err_rebin = np.array(errFunc(wl_rebin))
        return wl_rebin, fl_rebin, err_rebin

    elif err is False:
        return wl_rebin, fl_rebin


def estimate_snr(wave, flux, wl_begins=[5240, 6080, 6740],
                 wl_ends=[5320, 6160, 6820], plot_regions=False,
                 plot_values=False):
    # if no other wavelenghts are given: calculate SNR in three regions
    # blue (5310 - 5450 A), green (6700 - 6840 A), red (7100 - 7240 A)
    wls, snrs = [], []

    for i in range(len(wl_begins)):
        cut_range = ((wave > wl_begins[i]) & (wave < wl_ends[i]))
        cut_wave, cut_flux = wave[cut_range], flux[cut_range]
        mean_wl = np.mean(cut_wave)

        noise = np.std(cut_flux)  # noise = sqrt(variance) = stddev
        signal = np.median(cut_flux)  # median of signal in the same range
        snr = signal / noise

        wls.append(mean_wl)
        snrs.append(snr)

        print("SNR = %i in wavelength region from %i - %i A." %
              (snr, wl_begins[i], wl_ends[i]))

        if plot_regions is True:
            fig, ax = plt.subplots()
            ax.plot(cut_wave, cut_flux)
            plt.show()

    if plot_values is True:
        # coefs = poly.polyfit(wls, snrs, 1)
        # ffit = poly.polyval(wave, coefs)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # ax.plot(wave, ffit)
        ax.scatter(wls, snrs, color='orange')
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("SNR")
        plt.show()

    return snrs


def estimate_snr_from_errs(wave, flux, errs, wl_begins=[5240, 6080, 6740],
                           wl_ends=[5320, 6160, 6820], plot_regions=False,
                           plot_values=False):
    # calculate the SNR from the error spectrum  as median(flux/error)
    # if no other wavelenghts are given: calculate SNR in three regions
    # blue (5310 - 5450 A), green (6700 - 6840 A), red (7100 - 7240 A)
    wls, snrs = [], []

    for i in range(len(wl_begins)):
        cut = ((wave > wl_begins[i]) & (wave < wl_ends[i]))
        cut_wave, cut_flux, cut_err = wave[cut], flux[cut], errs[cut]
        mean_wl = np.nanmean(cut_wave)

        if np.nanmedian(cut_err) == 0.:
            print("For some reason the error is zero.")
            cut_err = 0.000001

        # print(cut_flux / cut_err)
        # rel_err = cut_err / cut_flux
        # print(rel_err)
        snr = np.nanmedian(cut_flux) / np.nanmedian(cut_err)
        wls.append(mean_wl)
        snrs.append(snr)

        print("SNR = %i in wavelength region from %i - %i A." %
              (snr, wl_begins[i], wl_ends[i]))

        if plot_regions is True:
            fig, ax = plt.subplots()
            ax.plot(cut_wave, cut_flux)
            ax.fill_between(cut_wave, cut_flux-cut_err, cut_flux+cut_err,
                            color='k', alpha=0.1)

    if plot_values is True:
        # coefs = poly.polyfit(wls, snrs, 1)
        # ffit = poly.polyval(wave, coefs)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # ax.plot(wave, ffit)
        ax.scatter(wls, snrs, color='orange')
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("SNR")
        plt.show()

    return snrs


def median_smooth(wave, flux, window):
    # provide flux, wavelength and number of points you wish to combine.
    flux_smooth = []
    for i in range(window, len(wave)-window):
        f = np.median(flux[i-window:i+window])
        flux_smooth.append(f)
        wave_smooth = wave[window:-window]
    return np.array(wave_smooth), np.array(flux_smooth)


def apply_snr(flux, snr):
    # Apply random noise to a given array
    print('Applying a S/N of %f to the data.' % snr)

    val = 1. / snr
    noise = np.random.normal(0, val, len(flux))
    noised_flux = np.array(flux + noise)

    return noised_flux


def doppler_shift(wave, vrad):
    # IMPORTANT!!!
    # Bug fix 10.01.2018!!! 1. - vrad / cc instead of 1 + xxx
    # applies doppler shift to wavelength array depending on given vrad
    print("Shifting wavelength array for vrad = %f." % vrad)
    wave_out = wave * (1. - vrad / cc)
    return wave_out


def rotation_broadening(wave_in, flux_in, vrot, fwhm=0.25, epsilon=0.6):
    print("Applying rotational broadening for vrot = %f." % vrot)
    # -- first a wavelength Gaussian convolution:
    if fwhm > 0:
        # convert fwhm to 1 sigma (FWHM = 2 * sqrt(2 * ln(2)) sigma)
        sigma = fwhm / 2.3548
        # -- make sure it's equidistant
        wave_ = np.linspace(wave_in[0], wave_in[-1], len(wave_in))
        flux_ = np.interp(wave_, wave_in, flux_in)
        dwave = wave_[1] - wave_[0]  # step in wave array
        n = int(round(2 * 4 * sigma / dwave, 0))
        wave_k = np.arange(n) * dwave
        wave_k = wave_k - wave_k[-1]/2.
        kernel = np.exp(- (wave_k)**2 / (2*sigma**2))
        kernel = kernel / sum(kernel)
        flux_conv = fftconvolve(1 - flux_, kernel, mode='same')
        wave_step = wave_in + dwave / 2
        flux_spec = np.interp(wave_step, wave_, 1-flux_conv, left=1, right=1)
    if vrot > 0:
        # -- convert wavelength array into velocity space, this is easier
        #   we also need to make it equidistant!
        wave_ = np.log(wave_in)
        velo_ = np.linspace(wave_[0], wave_[-1], len(wave_))
        flux_ = np.interp(velo_, wave_, flux_spec)
        dvelo = velo_[1] - velo_[0]
        vrot = vrot / cc
        # -- compute the convolution kernel and normalise it
        n = int(2 * vrot / dvelo)
        velo_k = np.arange(n)*dvelo
        velo_k = velo_k - velo_k[-1] / 2.
        y = 1 - (velo_k / vrot)**2  # transformation of velocity
        G = (2 * (1 - epsilon) * np.sqrt(y) + np.pi*epsilon / 2. * y) / \
            (np.pi * vrot * (1-epsilon / 3.0))  # the kernel
        G = G / sum(G)

        # -- convolve the flux with the kernel
        flux_conv = fftconvolve(1 - flux_, G, mode='same')
        velo_ = np.arange(len(flux_conv)) * dvelo + velo_[0]
        wave_conv = np.exp(velo_)
        return wave_conv, 1-flux_conv
    return np.array(wave_in), np.array(flux_spec)


def macro_broad(xdata, ydata, vmacro):
    """
    Edited broadening routine from http://dx.doi.org/10.5281/zenodo.10013

      This broadens the data by a given macroturbulent velocity.
    It works for small wavelength ranges. I need to make a better
    version that is accurate for large wavelength ranges! Sorry
    for the terrible variable names, it was copied from
    convol.pro in AnalyseBstar (Karolien Lefever)
    """

    # Make the kernel
    sq_pi = np.sqrt(np.pi)
    lambda0 = np.median(xdata)
    xspacing = xdata[1] - xdata[0]
    mr = vmacro * lambda0 / cc
    ccr = 2 / (sq_pi * mr)

    px = np.arange(-len(xdata) / 2, len(xdata) / 2 + 1) * xspacing
    pxmr = abs(px) / mr
    profile = ccr * (np.exp(-pxmr ** 2) + sq_pi * pxmr * (erf(pxmr) - 1.0))

    # Extend the xy axes to avoid edge-effects
    before = ydata[int(-profile.size / 2 + 1):]
    after = ydata[:int(profile.size / 2)]
    extended = np.r_[before, ydata, after]

    first = xdata[0] - float(int(profile.size / 2.0 + 0.5)) * xspacing
    last = xdata[-1] + float(int(profile.size / 2.0 + 0.5)) * xspacing
    x2 = np.linspace(first, last, extended.size)
    print(first)
    print(last)
    print(x2)
    newxdata = np.linspace(first, last, extended.size)

    conv_mode = "valid"

    # Do the convolution
    newydata = fftconvolve(extended, profile / profile.sum(), mode=conv_mode)
    print(len(newydata))
    print(len(newxdata))
    return newydata


def measure_equiwidth(wave, flux, snr):
    # measure the equivalent width of a spectral line
    # integration range ( length of wavelength bin to consider )
    int_range = wave[-1] - wave[0]
    # step width in lambda
    step = wave[1] - wave[0]

    # EW = int(1 - F_lambda / F0) dlambda = int(1 - Flambda) dlambda
    equiwidth = simps((1-flux), x=wave)

    # error from Ramirez-Tannus et al. 2018 :
    # sig_EW = sqrt(2 integration_range spec_dispersion) / S/N
    err = (2 * int_range * step)**0.5 / snr
    return equiwidth, err


def gaussian(x, height, center, std):
    gauss = 1. - (height * np.exp(-1. * (x - center)**2 / (2.*std**2)))
    return gauss


def single_fit(params, wavelength, fluxes):
    h, std, cen = params['h'], params['std'], params['cen']
    gauss = gaussian(wavelength, h, cen, std)
    error = (fluxes - gauss)**2
    return error


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


def measure_equiwidth2(wave, flux):

    params = lmfit.Parameters()
    params.add('h', min(flux), min=0.001, max=2., vary=True)
    params.add('std', 10, min=0.1, max=20., vary=True)
    params.add('cen', np.median(wave), min=wave[0], max=wave[-1], vary=True)

    minimizer = lmfit.Minimizer(single_fit, params, fcn_args=(wave, flux))
    res = minimizer.minimize()

    h, std, cen = res.params['h'], res.params['std'], res.params['cen']

    g = gaussian(wave, h, cen, std)

    equiwidth = simps((1 - g), x=wave)
    return equiwidth, h, std, cen


def get_vrad(lam, lam0):
    vrad = ((lam0 - lam) / lam0) * cc
    return vrad
