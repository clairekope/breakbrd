import os
import sys
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from photutils import CircularAnnulus, aperture_photometry

from utilities import *#get, url_sbhalos, args, folder, littleh

a0 = 1/(1+args.z)

#
# Functions for getting information from FITs files
#

def rmag_from_fits(sub_id):

    file = folder+"illustris_fits/broadband_{}{}.fits".format(
                                'rest_' if args.z!=0.0 else '', sub_id)

    # analyze; if file doesn't exist, error must be handled by calling code
    print("Rank {} reading id {}".format(rank, sub_id)); sys.stdout.flush()
    abs_mag_r = np.array(fits.getdata(file, ext=13)[4][13:21:2])
        
    return abs_mag_r

def gr_from_fits(sub_id, sub_dict):

    file = folder+"illustris_fits/broadband_{}{}.fits".format(
            'rest_' if args.z!=0.0 else '', sub_id)
    exten = 14 + sub_dict['view']
    
    # Prepare broadband images for magnitude calculation
    hdr = fits.getheader(file, ext=exten)
    unit = u.Unit(hdr['IMUNIT']) # spectral flux density
    npix = hdr['NAXIS1'] # pixels per dim (square image)
    pix_size = hdr['CD1_1'] * u.kpc
    assert pix_size.value == hdr['CD2_2']
    
    r_to_nu = ((6201.4 * u.Angstrom).to(u.m))**2 / c.c # from per-lambda to per-nu
    g_to_nu = ((4724.1 * u.Angstrom).to(u.m))**2 / c.c
    
    solid_ang = (pix_size)**2 / (10*u.pc)**2 # place object at 10 pc for abs mag
    solid_ang = solid_ang.to(u.sr, equivalencies=u.dimensionless_angles())
    
    sdss_g = fits.getdata(file, ext=exten)[3] * unit
    sdss_r = fits.getdata(file, ext=exten)[4] * unit
    
    sdss_g_mod = sdss_g * solid_ang * g_to_nu
    sdss_r_mod = sdss_r * solid_ang * r_to_nu
    
    Jy = u.Unit("erg / s / Hz / cm**2") # astropy Jy cancels extra dims
    f_zero = 3631e-23 * Jy # zero-point flux
    
    # Construct annulus for photometery
    R_half = sub_dict['half_mass_rad']*u.kpc
    center = (npix-1)/2
    pos = (center, center)
    rad_in = 2*u.kpc/pix_size
    rad_out = 2*R_half/pix_size
    ann = CircularAnnulus(pos, rad_in, rad_out)
    
    g_tot_flux = aperture_photometry(sdss_g_mod, ann)['aperture_sum'][0]
    r_tot_flux = aperture_photometry(sdss_r_mod, ann)['aperture_sum'][0]
    g_mag = -2.5*np.log10(g_tot_flux/f_zero)
    r_mag = -2.5*np.log10(r_tot_flux/f_zero)
    
    return g_mag - r_mag


#
# Functions for getting information from spectra
#
pc2cm   = (u.pc).to(u.cm)
lsun    = (u.solLum).to(u.erg/u.s)
mag2cgs = np.log10(lsun/4.0/np.pi/(pc2cm*pc2cm)/100.0)

def tsum(xin, yin):

    tsum = np.sum(np.abs((xin[1:]-xin[:-1]))*(yin[1:]+yin[:-1])/2. )

    return tsum

def band_mag(wave, spec, transmission_file):

    #load transmission function
    through = np.genfromtxt(transmission_file, skip_header = 1)

    #I only want the magnitude if the spectrum covers the band
    if np.min(wave) <= np.min(through[:,0]) \
       and np.max(wave) >= np.max(through[:,0]):

        # select the right part of the wavelengths
        bandw = np.logical_and(wave >= np.min(through[:,0]), 
                                 wave <= np.amax(through[:,0]))

        # interpolate the transmission function
        interp_band = interp1d(through[:,0],through[:,1])
        
        # the actual transmission for the spectrum
        trans = interp_band(wave[bandw]) \
                * 1.0/tsum(wave[bandw], interp_band(wave[bandw]) \
                * 1.0/wave[bandw])

        # calculate the magnitude by integrating the spectrum times the
        # transmission over the right wavelength range 
        # and convert to the right units
        mag = -2.5*np.log10( tsum(wave[bandw],
                                   spec[bandw]*trans*1.0/wave[bandw]
                            )) - 48.60 - 2.5*mag2cgs
    else:
        mag = 0

    return mag

def rmag_from_spectra(sub_id):

    # Use spectra made from whole subhalo
    file = folder+"spectra/{}inst/{}dust/full/spectra_{:06d}.txt".format(
                             "" if args.inst_sfr else "no_",
                             "" if args.dusty else "no_",
                             sub_id)

    dat = np.genfromtxt(file)
    wave = dat[0,:]
    spec = dat[1,:]

    r_mag = band_mag(wave, spec, 'SDSS_r_transmission.txt')

    return r_mag

def gr_from_spectra(sub_id):
    
    # Use spectra made only from "disk"
    file = folder+"spectra/{}inst/{}dust/disk/spectra_{:06d}.txt".format(
                             "" if args.inst_sfr else "no_",
                             "" if args.dusty else "no_",
                             sub_id)

    dat = np.genfromtxt(file)
    wave = dat[0,:]
    spec = dat[1,:]

    r_mag = band_mag(wave, spec, 'SDSS_r_transmission.txt')
    g_mag = band_mag(wave, spec, 'SDSS_g_transmission.txt')

    return g_mag - r_mag

