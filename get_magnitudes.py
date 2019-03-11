import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from photutils import CircularAnnulus, aperture_photometry

from utilities import get, url_sbhalos, args

#
# Functions for getting information from FITs files
#

def rmag_from_fits(sub_id):

    file = folder+"illustris_fits/broadband_{}{}.fits".format(
                                'rest_' if args.z!=0.0 else '', sub_id)

    # analyze; if file doesn't exist, error must be handled by calling code
    print("Rank {} reading id {}".format(rank, sub_id)); sys.stdout.flush()
    abs_mag_r = np.array(fits.getdata(file, ext=13)[4][13:21:2])
        

    if (abs_mag_r < -19).any():
        subhalo = get(url_sbhalos + str(sub_id))
        return {"M_r":abs_mag_r,
               "view":np.argmin(abs_mag_r),
               "half_mass_rad":subhalo["halfmassrad_stars"]*a/0.704, # ckpc/h to kpc
               "stellar_mass":subhalo['mass_stars']*1e10/0.704} 
    else:
        # delete subhalos that fail cut
        os.remove(file)

def gr_from_fits(sub_id, cut2_dict):

        file = folder+"illustris_fits/broadband_{}{}.fits".format(
                                    'rest_' if args.z!=0.0 else '', sub_id)
        exten = 14 + cut2_dict[sub_id]['view']
        
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
        R_half = cut2_dict[sub_id]['half_mass_rad']*u.kpc
        center = (npix-1)/2
        pos = (center, center)
        rad_in = 2*u.kpc/pix_size
        rad_out = 2*R_half/pix_size
        ann = CircularAnnulus(pos, rad_in, rad_out)
        
        g_tot_flux = aperture_photometry(sdss_g_mod, ann)['aperture_sum'][0]
        r_tot_flux = aperture_photometry(sdss_r_mod, ann)['aperture_sum'][0]
        g_mag = -2.5*np.log10(g_tot_flux/f_zero)
        r_mag = -2.5*np.log10(r_tot_flux/f_zero)
        
        if g_mag - r_mag > 0.655:
            return {'g-r':g_mag-r_mag, 
                     'view':cut2_dict[sub_id]['view'],
                     'half_mass_rad':cut2_dict[sub_id]['half_mass_rad'],
                     'M_r':cut2_dict[sub_id]['M_r'],
                     'stellar_mass':cut2_dict[sub_id]['stellar_mass']}


#
# Functions for getting information from spectra
#

def rmag_from_spectra(sub_id):

    # need wave and spec

    #load transmission function
    SDSS_r_through = np.asarray(np.loadtxt(datafolder+'SDSS_r_transmission.txt', skiprows = 1))

    #I only want the magnitude if the spectrum covers the band
    if np.min(wave) <= np.min(SDSS_r_through[:,0]) and np.max(wave) >= np.max(SDSS_r_through[:,0]):

        # select the right part of the wavelengths
        bandw_r = np.logical_and(wave >= np.amin(SDSS_r_through[:,0]), wave <= np.amax(SDSS_r_through[:,0]))
        # interpolate the transmission function
        interp_band_r = interp.interp1d(SDSS_r_through[:,0],SDSS_r_through[:,1])
        # the actual transmission for the spectrum
        trans_r = interp_band_r(wave[bandw_r])*1.0/tsum(wave[bandw_r],interp_band_r(wave[bandw_r])*1.0/wave[bandw_r])
        # calculate the magnitude by integrating the spectrum times the transmission over the right wavelength range and convert to the right units
        magr = -2.5*np.log10(tsum(wave[bandw_r],spec[bandw_r]*trans_r*1.0/wave[bandw_r])) - 48.60 - 2.5*mag2cgs
    else:
        magr = 0
    return magr

