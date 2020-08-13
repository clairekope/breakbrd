import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import h5py
import readsubfHDF5
import readhaloHDF5
import snapHDF5
import numpy as np
import astropy.units as u
from astropy.constants import m_p, k_B, G
from scipy.stats import binned_statistic_2d
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder, snapnum, littleh, omegaL/M
from utilities import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


nbins = 100
r_edges = np.logspace(-1, 0, nbins+1)


if rank==0:

    part_data = np.genfromtxt(folder+"parent_particle_data.csv", names=True)
    sub_list = part_data['id'].astype(np.int32)
    sat = part_data['satellite'].astype(np.bool)

    np.random.seed(6841325)
    subset = np.random.choice(sub_list[sat], size=500, replace=False)
    
    del part_data
    del sat
    
else:
    subset = None
    sub_list = None
                                   
#my_subs = scatter_work(sub_list, rank, size)
my_subs = scatter_work(subset, rank, size)
sub_list = comm.bcast(sub_list, root=0)

boxsize = get(url_dset)['boxsize']
z = args.z
a0 = 1/(1+z)

H0 = littleh * 100 * u.km/u.s/u.Mpc
rhocrit = 3*H0**2/(8*np.pi*G)
rhocrit = rhocrit.to(u.Msun/u.kpc**3)

good_ids = np.where(my_subs > -1)[0]
my_profiles = {}

for sub_id in my_subs[good_ids]:

    sub = get(url_sbhalos + str(sub_id))
    rhalf_star = sub["halfmassrad_stars"] * u.kpc * a0 / littleh
    #rhalf_gas = sub["halfmassrad_gas"] * u.kpc * a0 / littleh

    gas = True
    if not args.local:
        # Read particle data
        # gas_file = folder+"gas_cutouts/cutout_{}.hdf5".format(sub_id)
        gas_file = "/home/claire/cutout_{}.hdf5".format(sub_id)
        
        # Gas
        try:
            with h5py.File(gas_file) as f:
                coords = f['PartType0']['Coordinates'][:,:]
                dens = f['PartType0']['Density'][:]
                mass = f['PartType0']['Masses'][:]
                inte = f['PartType0']['InternalEnergy'][:]
                elec = f['PartType0']['ElectronAbundance'][:]
                dm_coords = f['PartType1']['Coordinates'][:]
        except KeyError:
            gas = False

    else:
        readhaloHDF5.reset()

        try:
            # Gas
            coords = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                           "POS ", 0, -1, sub_id, long_ids=True,
                                           double_output=False).astype("float32")
            dens = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                         "RHO ", 0, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")
            mass = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                         "MASS", 0, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")
            inte = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                         "U   ", 0, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")
            elec = readhaloHDF5.readhalo(args.local, "snap", snapnum,
                                         "NE  ", 0, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")
            dm_coords = readhaloHDF5.readhalo(args.local, "snap", snapnum,
                                              "POS ", 1, -1, sub_id, long_ids=True)
        except AttributeError:
            gas = False


    if gas:
        #
        # Calculate Entropy
        #

        # For conversion of internal energy to temperature, see
        # https://www.tng-project.org/data/docs/faq/#gen4
        X_H = 0.76
        gamma = 5./3.
        mu = 4/(1 + 3*X_H + 4*X_H*elec) * m_p
        temp = ( (gamma-1) * inte/k_B * mu * 1e10*u.erg/u.g ).to('K')

        dens = dens * 1e10*u.Msun/littleh * (u.kpc*a0/littleh)**-3
        ne = elec * X_H*dens/m_p
        ent = k_B * temp/ne**(gamma-1)
        ent = ent.to('eV cm^2', equivalencies=u.temperature_energy())

        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc * a0/littleh
        y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc * a0/littleh
        z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc * a0/littleh
        r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)

        mass = mass * 1e10 / littleh * u.Msun

        # TODO calculate r200 and bin K in scaled radial bins
        dm_x = dm_coords[:,0]
        dm_y = dm_coords[:,1]
        dm_z = dm_coords[:,2]
        dm_x_rel = periodic_centering(dm_x, sub['pos_x'], boxsize) * u.kpc * a0/littleh
        dm_y_rel = periodic_centering(dm_y, sub['pos_y'], boxsize) * u.kpc * a0/littleh
        dm_z_rel = periodic_centering(dm_z, sub['pos_z'], boxsize) * u.kpc * a0/littleh
        dm_r = np.sqrt(dm_x_rel**2 + dm_y_rel**2 + dm_z_rel**2)

        dm_rsort = np.sort(dm_r)
        # count DM particles inside a sphere to find density at that radius
        dm_dens = 7.5e6*u.Msun*np.arange(1,dm_rsort.size+1) / np.power(dm_rsort,3)

        try:
            # pick first particle beyond density cutoff for r200. Density falls with r!
            r200 = dm_rsort[dm_dens < 200*rhocrit][0]

            per_off = ( (mass.sum() - mass[r < r200].sum()) / mass.sum() ).value * 100
            print(sub_id, per_off)

        except IndexError:
            print(sub_id, "No r200 found")
            continue

#     else: # no gas
#         my_profiles[sub_id] = np.nan

# profile_list = comm.gather(my_profiles, root=0)

# if rank==0:

#     all_profiles = np.zeros( (len(sub_list), nbins+1) )
#     i=0
#     for dic in profile_list:
#         for k,v in dic.items():
#             all_profiles[i,0] = k
#             all_profiles[i,1:] = v
#             i+=1

#     sort = np.argsort(all_profiles[:,0])

#     header = "SubID"
#     for r in radii:
#         header += " {:.2f}".format(r)

#     np.savetxt(folder+'entropy_profiles.csv', all_profiles[sort], 
#                delimiter=',', header=header)

