from __future__ import print_function, division
import os
import sys
import requests
import pickle
import warnings
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from photutils import CircularAnnulus, aperture_photometry
from mpi4py import MPI
from copy import deepcopy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def scatter_work(array, mpi_rank, mpi_size, root=0):
    """ array should only exist on root & be None elsewhere"""
    if mpi_rank == root:
        scatter_total = array.size
        mod = scatter_total % mpi_size
        if mod != 0:
            print("Padding array for scattering...")
            pad = -1 * np.ones(mpi_size - mod, dtype='i')
            array = np.concatenate((array, pad))
            scatter_total += mpi_size - mod
            assert scatter_total % mpi_size == 0
            assert scatter_total == array.size
    else:
        scatter_total = None
        #array = None
    scatter_total = comm.bcast(scatter_total, root=root)
    subset = np.empty(scatter_total//mpi_size, dtype='i')
    comm.Scatter(array, subset, root=root)    

    return subset

headers = {"api-key":"5309619565f744f9248320a886c59bec"}

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
       
    # for binary data (FITS or HDF5)
    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

snap_url = "http://www.illustris-project.org/api/Illustris-1/snapshots/135/subhalos/"

if rank == 0:

    #
    # How many galaxies total?
    #
    total = get(snap_url)
    print("Total subhalos:",total['count'])

    #
    # Get galaxies with stellar mass > min_mass
    #
    # Only subhalos (galaxies) with $M_*>10^{10}$ have stellar mocks 
    #([source](http://www.illustris-project.org/data/docs/specifications/#sec4a))
    
    # convert log solar masses into group catalog units
    min_mass = 1 * 0.704 # 1e10/1e10 * 0.704
    
    # form query
    search_query = "?mass_stars__gt=" + str(min_mass)
    
    cut1 = get(snap_url + search_query)
    scatter_total = cut1['count']
    print("Galaxies from mass cut:", cut1['count'])

if not os.path.isfile("cut2.pkl"):

    #
    # Cut on $M_R < -19$
    #
   
    if rank == 0:
        # Re-get `cut1` with all desired subhalos so I don't have to paginate
        cut1 = get(snap_url + search_query, {'limit':cut1['count']})
        subhalo_ids = np.array([sub['id'] for sub in cut1['results']], dtype='i')
    else:
        subhalo_ids = None

    halo_subset = scatter_work(subhalo_ids, rank, size)
    
    # will gather my_cut2 into cut2
    my_cut2 = {}
    #cut2 = {}
    
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
            #    get(rband_url)
            #except requests.HTTPError:
            print("Subhalo {} not found".format(sub_id)); sys.stdout.flush()
            continue

        # analyze
        #print("Rank {} reading id {}".format(rank, sub_id)); sys.stdout.flush()
        abs_mag_r = np.array(fits.getdata(file, ext=13)[4][13:21:2])
        if (abs_mag_r < -19).any():
            subhalo = get(snap_url + str(sub_id))
            my_cut2[sub_id] = {"M_r":abs_mag_r,
                               "view":np.argmin(abs_mag_r),
                               "half_mass_rad":subhalo["halfmassrad_stars"]/0.704, # kpc/h to kpc
                               "stellar_mass":subhalo['mass_stars']*1e10/0.704} 
            #cut2[sub_id] = abs_mag_r
            
        else:
            # delete subhalos that fail cut
            os.remove(file)

    cut2_lst = comm.gather(my_cut2, root=0)
    if rank==0:
        cut2 = {}
        for dic in cut2_lst:
            for k, v in dic.items():
                cut2[k] = v
        with open("cut2.pkl", "wb") as f:
            pickle.dump(cut2, f)
    else:
        cut2 = None
else: # cut2 dict already generated
    if rank == 0:
        with open("cut2.pkl","rb") as f:
            cut2 = pickle.load(f)
        print("cut2.pkl exists")
    else:
        cut2 = None

# broadcast dict (either created or read in)
cut2 = comm.bcast(cut2, root=0)
if rank == 0:
    print("Galaxies from M_r cut:",len(cut2))
    cut2_subhalos = np.array([k for k in cut2.keys()])
else:
    cut2_subhalos = None
    

# cut 2.5: pass the M_r cut but less than 1e12 Msun in stars
if rank==0:
    print("copying...")
    cut2_new = deepcopy(cut2)
    print("copy done")
    for k in cut2.keys():
        if cut2_new[k]['stellar_mass'] > 1e12:
            cut2_new.pop(k)
    with open("cut2.5.pkl", "wb") as f:
        pickle.dump(cut2_new, f)

        
if not os.path.isfile("cut3.pkl"):
    #
    # Cut on g-r in disk
    #

    halo_subset2 = scatter_work(cut2_subhalos, rank, size)
    good_ids = np.where(halo_subset2 > -1)

    my_cut3 = {}

    for sub_id in halo_subset2[good_ids]:
        file = "illustris_fits/broadband_{}.fits".format(sub_id)
        exten = 14 + cut2[sub_id]['view']
        
        # Prepare broadband images for magnitude calculation
        unit = u.Unit(fits.getheader(file, ext=exten)['IMUNIT']) # spectral flux density
        Jy = u.Unit("erg / s / Hz / cm**2") # astropy Jy cancels extra dims
        npix = fits.getheader(file, ext=exten)['NAXIS1'] # pixels per dim (square image)
        r_to_nu = ((6201.4 * u.Angstrom).to(u.m))**2 / c.c # from per-lambda to per-nu
        g_to_nu = ((4724.1 * u.Angstrom).to(u.m))**2 / c.c
        R_half = cut2[sub_id]['half_mass_rad']*u.kpc
        pix_size = 10*R_half/256 # FOV is 10 stellar half mass radii
        solid_ang = (pix_size)**2 / (10*u.pc)**2 # place object at 10 pc for abs mag
        solid_ang = solid_ang.to(u.sr, equivalencies=u.dimensionless_angles())
        
        sdss_g = fits.getdata(file, ext=14)[3] * unit
        sdss_r = fits.getdata(file, ext=14)[4] * unit
        
        sdss_g_mod = sdss_g * solid_ang * g_to_nu
        sdss_r_mod = sdss_r * solid_ang * r_to_nu
        
        f_zero = 3631e-23 * Jy # zero-point flux
        
        # Construct annulus for photometery
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
            my_cut3[sub_id] = {'g-r':g_mag-r_mag, 
                               'view':cut2[sub_id]['view'],
                               'half_mass_rad':cut2[sub_id]['half_mass_rad'],
                               'M_r':cut2[sub_id]['M_r'],
                               'stellar_mass':cut2[sub_id]['stellar_mass']}

    cut3_lst = comm.gather(my_cut3, root=0)
    if rank==0:
        cut3 = {}
        for dic in cut3_lst:
            for k, v in dic.items():
                cut3[k] = v
            
        with open("cut3.pkl", "wb") as f:
            pickle.dump(cut3, f)
    else:
        cut3 = None
else: # cut3 dict already generated                                                                
    if rank == 0:
        with open("cut3.pkl","rb") as f:
            cut3 = pickle.load(f)
        print("cut3.pkl exists")
    else:
        cut3 = None

# Make cut 4, a polish of cut 3
cut3 = comm.bcast(cut3, root=0)
if rank==0:
    print("Galaxies from g-r cut:", len(cut3))

    cut4 = {}

    for sub_id in cut3.keys():
        if cut3[sub_id]['stellar_mass'] < 1e12 and cut3[sub_id]['half_mass_rad'] > 2:
            cut4[sub_id] = cut3[sub_id]
            cut4[sub_id]['stellar_mass'] = star_mass * 1e10 / 0.704 # Msun

    with open("cut4.pkl","wb") as f:
        pickle.dump(cut4, f)
    
