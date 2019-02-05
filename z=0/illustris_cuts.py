from __future__ import print_function, division
import os
import sys
import pickle
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from photutils import CircularAnnulus, aperture_photometry
#from mpi4py import MPI
from copy import deepcopy
from utilities import *

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

snap_url = "http://www.illustris-project.org/api/Illustris-1/snapshots/135/subhalos/"
if rank == 0:
    z = get(snap_url[:-9])['redshift']
    a = 1/(1+z)
else:
    a = None
a = comm.bcast(a, root=0)

#
# How many galaxies total?
# With stellar mocks?
#
if rank == 0:

    total = get(snap_url)
    print("Total subhalos:",total['count'])

    #
    # Get galaxies with stellar mass > min_mass
    #
    # Only subhalos (galaxies) with $M_*>10^{10}$ have stellar mocks 
    #([source](http://www.illustris-project.org/data/docs/specifications/#sec4a))
    
    # convert log solar masses into group catalog units
    min_mass = 0.704 # 1e10/1e10 * 0.704
    
    # form query
    search_query = "?mass_stars__gt=" + str(min_mass)
    
    cut1 = get(snap_url + search_query)
    scatter_total = cut1['count']
    print("Galaxies from mass cut:", cut1['count'])


#
# Cut on $M_R < -19$
#
if not os.path.isfile("cut2_M_r.pkl"):
   
    if rank == 0:
        # Re-get `cut1` with all desired subhalos so I don't have to paginate
        cut1 = get(snap_url + search_query, {'limit':cut1['count']})
        subhalo_ids = np.array([sub['id'] for sub in cut1['results']], dtype='i')
    else:
        subhalo_ids = None

    halo_subset = scatter_work(subhalo_ids, rank, size)
    
    # will gather my_cut2_M_r into cut2_M_r
    my_cut2_M_r = {}
    #cut2_M_r = {}
    
    # ignore padding added for scattering
    good_ids = np.where(halo_subset > -1)[0]
    
    # download all fits files
    for sub_id in halo_subset[good_ids]:
    #for sub_id in subhalo_ids:
        file = "illustris_fits/broadband_{}.fits".format(sub_id)
        
        # skip if not fetched    
        if not os.path.isfile(file):
            #print("Rank {} fetching id {}".format(rank, sub_id)); sys.stdout.flush()
            #rband_url = snap_url + str(sub_id) + "/stellar_mocks/broadband.fits"
            #try:
            #    get(rband_url, fpath="illustris_fits/")
            #except requests.HTTPError:
            print("Subhalo {} not found".format(sub_id)); sys.stdout.flush()
            continue

        # analyze
        print("Rank {} reading id {}".format(rank, sub_id)); sys.stdout.flush()
        try:
            abs_mag_r = np.array(fits.getdata(file, ext=13)[4][13:21:2])
        except OSError:
            continue

        if (abs_mag_r < -19).any():
            subhalo = get(snap_url + str(sub_id))
            my_cut2_M_r[sub_id] = {"M_r":abs_mag_r,
                               "view":np.argmin(abs_mag_r),
                               "half_mass_rad":subhalo["halfmassrad_stars"]*a/0.704, # ckpc/h to kpc
                               "stellar_mass":subhalo['mass_stars']*1e10/0.704} 
            #cut2_M_r[sub_id] = abs_mag_r
            
        else:
            # delete subhalos that fail cut
            os.remove(file)

    cut2_M_r_lst = comm.gather(my_cut2_M_r, root=0)
    if rank==0:
        cut2_M_r = {}
        for dic in cut2_M_r_lst:
            for k, v in dic.items():
                cut2_M_r[k] = v
        with open("cut2_M_r.pkl", "wb") as f:
            pickle.dump(cut2_M_r, f)
    else:
        cut2_M_r = None
else: # cut2_M_r dict already generated
    if rank == 0:
        with open("cut2_M_r.pkl","rb") as f:
            cut2_M_r = pickle.load(f)
        print("cut2_M_r.pkl exists")
    else:
        cut2_M_r = None

# broadcast dict (either created or read in)
cut2_M_r = comm.bcast(cut2_M_r, root=0)
if rank == 0:
    print("Galaxies from M_r cut:",len(cut2_M_r))
    cut2_M_r_subhalos = np.array([k for k in cut2_M_r.keys()])
else:
    cut2_M_r_subhalos = None
    
    

# 
# Clean up M_r for parent sample
#
if rank==0:

    cut2_M_r_new = {}

    for sub_id in cut2_M_r.keys():
        if cut2_M_r[sub_id]['stellar_mass'] < 1e12 and cut2_M_r[sub_id]['half_mass_rad'] > 2:
            cut2_M_r_new[sub_id] = cut2_M_r[sub_id]

    with open("parent.pkl","wb") as f:
        pickle.dump(cut2_M_r_new, f)
else:
    cut2_M_r_new = None

#cut2_M_r_new = comm.bcast(cut2_M_r_new, root=0)
if rank == 0:
    print("Galaxies from cleaning cut:",len(cut2_M_r_new))
    cut2_M_r_subhalos = np.array([k for k in cut2_M_r_new.keys()])
else:
    cut2_M_r_subhalos = None    


#
# Cut on g-r in disk
#
if not os.path.isfile("cut3_g-r.pkl"):

    halo_subset2 = scatter_work(cut2_M_r_subhalos, rank, size)
    good_ids = np.where(halo_subset2 > -1)

    my_cut3_gr = {}

    for sub_id in halo_subset2[good_ids]:
        file = "illustris_fits/broadband_{}.fits".format(sub_id)
        exten = 14 + cut2_M_r[sub_id]['view']
        
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
        R_half = cut2_M_r[sub_id]['half_mass_rad']*u.kpc
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
            my_cut3_gr[sub_id] = {'g-r':g_mag-r_mag, 
                               'view':cut2_M_r[sub_id]['view'],
                               'half_mass_rad':cut2_M_r[sub_id]['half_mass_rad'],
                               'M_r':cut2_M_r[sub_id]['M_r'],
                               'stellar_mass':cut2_M_r[sub_id]['stellar_mass']}

    cut3_gr_lst = comm.gather(my_cut3_gr, root=0)
    if rank==0:
        cut3_gr = {}
        for dic in cut3_gr_lst:
            for k, v in dic.items():
                cut3_gr[k] = v
            
        with open("cut3_g-r.pkl", "wb") as f:
            pickle.dump(cut3_gr, f)
    else:
        cut3_gr = None
else: # cut3_gr dict already generated                                                                
    if rank == 0:
        with open("cut3_g-r.pkl","rb") as f:
            cut3 = pickle.load(f)
        print("cut3_g-r.pkl exists")
    else:
        cut3_gr = None


