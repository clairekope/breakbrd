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
binned_r = r_edges[:-1] + np.diff(r_edges)


if rank==0:

    part_data = np.genfromtxt(folder+"parent_particle_data.csv", names=True)
    sub_list = part_data['id'].astype(np.int32)
    sat = part_data['satellite'].astype(np.bool)

    # np.random.seed(6841325)
    # subset = np.random.choice(sub_list[sat], size=500, replace=False)
    
    del part_data
    del sat
    
else:
    subset = None
    sub_list = None
                                   
my_subs = scatter_work(sub_list, rank, size)
#my_subs = scatter_work(subset, rank, size)
sub_list = comm.bcast(sub_list, root=0)

boxsize = get(url_dset)['boxsize']
z = args.z
a0 = 1/(1+z)

H0 = littleh * 100 * u.km/u.s/u.Mpc

good_ids = np.where(my_subs > -1)[0]
my_profiles = {}

for sub_id in my_subs[good_ids]:

    my_profiles[sub_id] = {}

    sub = get(url_sbhalos + str(sub_id))
    dm_halo = sub["mass_dm"] * 1e10 / littleh * u.Msun
    r_half = sub["halfmassrad_stars"] * u.kpc * a0/littleh
    r_half_gas = sub["halfmassrad_gas"] * u.kpc * a0/littleh

    my_profiles[sub_id]['dm_mass'] = dm_halo

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
                elec = f['PartType0']['ElectronAbundance'][:] #x_e = n_e/n_H

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
        ne = elec * X_H*dens/m_p # elec frac defined as n_e/n_H
        ent = k_B * temp/ne**(gamma-1)
        ent = ent.to('eV cm^2', equivalencies=u.temperature_energy())

        pres = dens/m_p * k_B * temp

        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc * a0/littleh
        y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc * a0/littleh
        z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc * a0/littleh
        r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)

        mass = mass * 1e10 / littleh * u.Msun

        r200 = (G*dm_halo/(100*H0**2))**(1/3)
        r200 = r200.to('kpc')

        r_scale = (r/r200).value
        rbinner = np.digitize(r_scale, r_edges)

        dthresh = 0.1 * u.cm**-3 * m_p # cannonical star formation density threshold

        dense_disk = np.logical_and(dens > dthresh, r < 2*r_half)
        if dense_disk.sum() > 0:
            disk_avg_ent = np.average(ent[dense_disk], weights=mass[dense_disk])
            my_profiles[sub_id]['disk_ent_avg'] = disk_avg_ent
        else:
            my_profiles[sub_id]['disk_ent_avg'] = np.nan * u.eV*u.cm**2

        inner_CGM = np.logical_and(r > 2*r_half, r < r_half_gas)
        dense_inner = np.logical_and(dens > dthresh, inner_CGM)
        if dense_inner.sum() > 0:
            inner_avg_ent = np.average(ent[dense_inner], weights=mass[dense_inner])
            my_profiles[sub_id]['inner_ent_avg'] = inner_avg_ent
        else:
            my_profiles[sub_id]['inner_ent_avg'] = np.nan * u.eV*u.cm**2

        binned_ent_avg = np.ones_like(binned_r)*np.nan * u.eV*u.cm**2
        binned_ent_med = np.ones_like(binned_r)*np.nan * u.eV*u.cm**2

        binned_pres_avg = np.ones_like(binned_r)*np.nan * u.dyn/u.cm**2
        binned_pres_med = np.ones_like(binned_r)*np.nan * u.dyn/u.cm**2

        for i in range(1, r_edges.size):
            this_bin = rbinner==i
            if np.sum(mass[this_bin]) != 0: # are there particles in this bin

                binned_ent_avg[i-1] = np.average(ent[this_bin],
                                             weights = mass[this_bin])
                binned_ent_med[i-1] = np.median(ent[this_bin])

                binned_pres_avg[i-1] = np.average(pres[this_bin],
                                             weights = mass[this_bin])
                binned_pres_med[i-1] = np.median(pres[this_bin])

        my_profiles[sub_id]['ent_avg'] = binned_ent_avg
        my_profiles[sub_id]['ent_med'] = binned_ent_med
        my_profiles[sub_id]['pres_avg'] = binned_pres_avg
        my_profiles[sub_id]['pres_med'] = binned_pres_med

    else: # no gas
        my_profiles[sub_id]['disk_ent_avg'] = np.nan * u.eV*u.cm**2
        my_profiles[sub_id]['inner_ent_avg'] = np.nan * u.eV*u.cm**2
        my_profiles[sub_id]['ent_avg'] = np.nan
        my_profiles[sub_id]['ent_med'] = np.nan
        my_profiles[sub_id]['pres_avg'] = np.nan
        my_profiles[sub_id]['pres_med'] = np.nan

profile_list = comm.gather(my_profiles, root=0)

if rank==0:

    all_entprof = np.zeros( (len(sub_list), 2*nbins+4) )
    all_presprof = np.zeros( (len(sub_list), 2*nbins+2) )

    i=0
    for dic in profile_list:
        for k,v in dic.items():
            all_entprof[i,0] = k
            all_entprof[i,1] = v['dm_mass'].value
            all_entprof[i,2] = v['disk_ent_avg'].value
            all_entprof[i,3] = v['inner_ent_avg'].value
            all_entprof[i,4::2] = v['ent_avg']
            all_entprof[i,5::2] = v['ent_med']
            
            all_presprof[i,0] = k
            all_presprof[i,1] = v['dm_mass'].value
            all_presprof[i,2::2] = v['pres_avg']
            all_presprof[i,3::2] = v['pres_med']
            
            i+=1

    ent_sort = np.argsort(all_entprof[:,0])
    pres_sort = np.argsort(all_presprof[:,0])

    header = "SubID   DMmass   DskEnt   CGMEnt"
    for r in binned_r:
        header += "   {:.4f} avg med".format(r)

    np.savetxt(folder+'parent_entropy_profiles.csv', all_entprof[ent_sort], 
               delimiter=',', header=header)
    np.savetxt(folder+'parent_pressure_profiles.csv', all_presprof[pres_sort],
               delimiter=',', header=header)
