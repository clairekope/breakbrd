import pickle
import h5py
import sys
import os
import gc
import readsubfHDF5
import readhaloHDF5
import snapHDF5
import numpy as np
import astropy.units as u
#import matplotlib; matplotlib.use('agg')
#import matplotlib.pyplot as plt
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder, snapnum, littleh, omegaL/M
from utilities import * 

a0 = 1/(1+args.z)

if args.z==0.0:
    id_file = 'TNG/final_cut_TNG_z01_SubhaloNrsAtz00.txt'
elif args.z==0.1:
    id_file = 'TNG/final_cut_TNG_z00_SubhaloNrsAtz01.txt'

if rank==0:

    sub_ids = np.genfromtxt(id_file, dtype=np.int32)

    if args.local:
        cat = readsubfHDF5.subfind_catalog(args.local, snapnum, 
                                           keysel=['GroupFirstSub','SubhaloGrNr'])
        sat = np.zeros(cat.SubhaloGrNr.size, dtype=bool)
        sat[sub_ids] = (sub_ids != cat.GroupFirstSub[cat.SubhaloGrNr[sub_ids]])
        del cat
        gc.collect()
else:
    sub_ids = None
    if args.local:
        sat = None

my_subs = scatter_work(sub_ids, rank, size)
if args.local:
    sat = comm.bcast(sat, root=0)
my_particle_data = {}

boxsize = get(url_dset)['boxsize']
dthresh = 6.4866e-4 # 0.13 cm^-3 in code units -> true for TNG?

good_ids = np.where(my_subs > -1)[0]

for sub_id in my_subs[good_ids]:

    # Get half mass radius
    sub = get(url_sbhalos+str(sub_id))

    gas = True

    readhaloHDF5.reset()

    try:
        # Gas
        coords = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                       "POS ", 0, -1, sub_id, long_ids=True,
                                       double_output=False).astype("float32")
        mass = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                     "MASS", 0, -1, sub_id, long_ids=True,
                                     double_output=False).astype("float32")
        dens = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                     "RHO ", 0, -1, sub_id, long_ids=True,
                                     double_output=False).astype("float32")
        sfr = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                    "SFR ", 0, -1, sub_id, long_ids=True,
                                    double_output=False).astype("float32")
    except AttributeError:
        gas = False

    # Stars
    scoords = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                        "POS ", 4, -1, sub_id, long_ids=True,
                                        double_output=False).astype("float32")
    smass = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                      "MASS", 4, -1, sub_id, long_ids=True,
                                      double_output=False).astype("float32")
    a = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                  "GAGE", 4, -1, sub_id, long_ids=True,
                                  double_output=False).astype("float32")

    my_particle_data[sub_id] = {}

    if gas:
        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc * a0/littleh
        y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc * a0/littleh
        z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc * a0/littleh
        r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        mass = mass * 1e10 / littleh * u.Msun
        sfr = sfr * u.Msun/u.yr
        
        inr_reg = r < 2*u.kpc
        
        tot_dense = dens > dthresh
        inr_dense = np.logical_and(inr_reg, dens > dthresh)
        
        gas_tot = np.sum(mass)
        gas_inr = np.sum(mass[inr_reg])

        SFgas_tot = np.sum(mass[tot_dense])
        SFgas_inr = np.sum(mass[inr_dense])

        sfr_tot = np.sum(sfr)
        sfr_inr = np.sum(sfr[inr_reg])
        
        my_particle_data[sub_id]['total_gas'] = gas_tot
        my_particle_data[sub_id]['inner_gas'] = gas_inr

        my_particle_data[sub_id]['total_SFgas'] = SFgas_tot
        my_particle_data[sub_id]['inner_SFgas'] = SFgas_inr

        my_particle_data[sub_id]['total_SFR'] = sfr_tot
        my_particle_data[sub_id]['inner_SFR'] = sfr_inr

        my_particle_data[sub_id]['total_SFE'] = sfr_tot / SFgas_tot
        my_particle_data[sub_id]['inner_SFE'] = sfr_inr / SFgas_inr
                                   
    sx = scoords[:,0]
    sy = scoords[:,1]
    sz = scoords[:,2]
    sx_rel = periodic_centering(sx, sub['pos_x'], boxsize) * u.kpc * a0/littleh
    sy_rel = periodic_centering(sy, sub['pos_y'], boxsize) * u.kpc * a0/littleh
    sz_rel = periodic_centering(sz, sub['pos_z'], boxsize) * u.kpc * a0/littleh
    sr = np.sqrt(sx_rel**2 + sy_rel**2 + sz_rel**2)    
    smass = smass * 1e10 / littleh * u.Msun

    sinr_reg = sr < 2*u.kpc

    star_tot = np.sum(smass)
    star_inr = np.sum(smass[sinr_reg])

    my_particle_data[sub_id]['total_star'] = star_tot
    my_particle_data[sub_id]['inner_star'] = star_inr

    if args.local:
        my_particle_data[sub_id]['satellite'] = sat[sub_id]
        
particle_lst = comm.gather(my_particle_data, root=0)

if rank==0:
    
    all_particle_data = {}
    for dic in particle_lst:
        for k,v in dic.items():
            all_particle_data[k] = v

    with open(folder+"tracked_particle_data_{}.pkl".format(
            "z01ATz00" if args.z==0.0 else "z00ATz01"),"wb") as f:
        pickle.dump(all_particle_data,f)

