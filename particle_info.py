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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

z = args.z
a0 = 1/(1+z)

if rank==0:
    # Get the halos to loop over. It is now "all" of them.
    min_mass = littleh # 1e10 Msun in 1/1e10 Msun / h
    max_mass = 100 * littleh # 1e12 Msun
    search_query = "?mass_stars__gt=" + str(min_mass) \
                 + "&mass_stars__lt=" + str(max_mass) \
                 + "&halfmassrad_stars__gt=" + str(2 / a0 * littleh) # 2 kpc

    cut1 = get(url_sbhalos + search_query)
    cut1['count']
    cut1 = get(url_sbhalos + search_query, {'limit':cut1['count'], 'order_by':'id'})

    sub_list = cut1['results']
    sub_ids = np.array([sub['id'] for sub in cut1['results']], dtype='i')

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

    print(f"{rank}: {sub_id}", flush=True)

    # Get half mass radius
    sub = get(url_sbhalos+str(sub_id))
    r_half = sub["halfmassrad_stars"] * u.kpc * a0 / littleh

    gas = True

    if not args.local:
        # Read particle data
        gas_file = folder+"gas_cutouts/cutout_{}.hdf5".format(sub_id)
        star_file = folder+"stellar_cutouts/cutout_{}.hdf5".format(sub_id)

        # Gas
        try:
            with h5py.File(gas_file) as f:
                coords = f['PartType0']['Coordinates'][:,:]
                mass = f['PartType0']['Masses'][:]
                dens = f['PartType0']['Density'][:]
                #inte = f['PartType0']['InternalEnergy'][:]
                #HI = f['PartType0']['NeutralHydrogenAbundance'][:]
                sfr = f['PartType0']['StarFormationRate'][:]
        except KeyError:
            #print(sub_id, "no gas"); sys.stdout.flush()
            gas = False

        # Stars
        try:
            with h5py.File(star_file) as f:
                scoords = f['PartType4']['Coordinates'][:]
                smass = f['PartType4']['Masses'][:]
                a = f['PartType4']['GFM_StellarFormationTime']
        except KeyError:
            #print(sub_id, "no stars"); sys.stdout.flush()
            pass

    else:
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
        otr_reg = np.logical_and(r > 2*u.kpc, r < 2*r_half)
        far_reg = r > 2*r_half
        dsk_reg = r > 2*u.kpc
        
        tot_dense = dens > dthresh
        inr_dense = np.logical_and(inr_reg, dens > dthresh)
        otr_dense = np.logical_and(otr_reg, dens > dthresh)
        far_dense = np.logical_and(far_reg, dens > dthresh)
        dsk_dense = np.logical_and(dsk_reg, dens > dthresh)
        
        gas_tot = np.sum(mass)
        gas_inr = np.sum(mass[inr_reg])
        gas_otr = np.sum(mass[otr_reg])
        gas_far = np.sum(mass[far_reg])
        gas_dsk = np.sum(mass[dsk_reg])

        SFgas_tot = np.sum(mass[tot_dense])
        SFgas_inr = np.sum(mass[inr_dense])
        SFgas_otr = np.sum(mass[otr_dense])
        SFgas_far = np.sum(mass[far_dense])
        SFgas_dsk = np.sum(mass[dsk_dense])

        sfr_tot = np.sum(sfr)
        sfr_inr = np.sum(sfr[inr_reg])
        sfr_otr = np.sum(sfr[otr_reg])
        sfr_far = np.sum(sfr[far_reg])
        sfr_dsk = np.sum(sfr[dsk_reg])
        
        my_particle_data[sub_id]['total_gas'] = gas_tot
        my_particle_data[sub_id]['inner_gas'] = gas_inr
        my_particle_data[sub_id]['outer_gas'] = gas_otr
        my_particle_data[sub_id]['far_gas']   = gas_far
        my_particle_data[sub_id]['disk_gas']  = gas_dsk

        my_particle_data[sub_id]['total_SFgas'] = SFgas_tot
        my_particle_data[sub_id]['inner_SFgas'] = SFgas_inr
        my_particle_data[sub_id]['outer_SFgas'] = SFgas_otr
        my_particle_data[sub_id]['far_SFgas']   = SFgas_far
        my_particle_data[sub_id]['disk_SFgas']  = SFgas_dsk

        my_particle_data[sub_id]['total_SFR'] = sfr_tot
        my_particle_data[sub_id]['inner_SFR'] = sfr_inr
        my_particle_data[sub_id]['outer_SFR'] = sfr_otr
        my_particle_data[sub_id]['far_SFR']   = sfr_far
        my_particle_data[sub_id]['disk_SFR']  = sfr_dsk

        my_particle_data[sub_id]['total_SFE'] = sfr_tot / SFgas_tot
        my_particle_data[sub_id]['inner_SFE'] = sfr_inr / SFgas_inr
        my_particle_data[sub_id]['outer_SFE'] = sfr_otr / SFgas_otr
        my_particle_data[sub_id]['far_SFE']   = sfr_far / SFgas_far
        my_particle_data[sub_id]['disk_SFE']  = sfr_dsk / SFgas_dsk

    else:
        my_particle_data[sub_id]['total_gas'] = np.nan
        my_particle_data[sub_id]['inner_gas'] = np.nan
        my_particle_data[sub_id]['outer_gas'] = np.nan
        my_particle_data[sub_id]['far_gas']   = np.nan
        my_particle_data[sub_id]['disk_gas']  = np.nan

        my_particle_data[sub_id]['total_SFgas'] = np.nan
        my_particle_data[sub_id]['inner_SFgas'] = np.nan
        my_particle_data[sub_id]['outer_SFgas'] = np.nan
        my_particle_data[sub_id]['far_SFgas']   = np.nan
        my_particle_data[sub_id]['disk_SFgas']  = np.nan

        my_particle_data[sub_id]['total_SFR'] = np.nan
        my_particle_data[sub_id]['inner_SFR'] = np.nan
        my_particle_data[sub_id]['outer_SFR'] = np.nan
        my_particle_data[sub_id]['far_SFR']   = np.nan
        my_particle_data[sub_id]['disk_SFR']  = np.nan

        my_particle_data[sub_id]['total_SFE'] = np.nan
        my_particle_data[sub_id]['inner_SFE'] = np.nan
        my_particle_data[sub_id]['outer_SFE'] = np.nan
        my_particle_data[sub_id]['far_SFE']   = np.nan
        my_particle_data[sub_id]['disk_SFE']  = np.nan
                                   
    sx = scoords[:,0]
    sy = scoords[:,1]
    sz = scoords[:,2]
    sx_rel = periodic_centering(sx, sub['pos_x'], boxsize) * u.kpc * a0/littleh
    sy_rel = periodic_centering(sy, sub['pos_y'], boxsize) * u.kpc * a0/littleh
    sz_rel = periodic_centering(sz, sub['pos_z'], boxsize) * u.kpc * a0/littleh
    sr = np.sqrt(sx_rel**2 + sy_rel**2 + sz_rel**2)    
    smass = smass * 1e10 / littleh * u.Msun

    sinr_reg = sr < 2*u.kpc
    sotr_reg = np.logical_and(sr > 2*u.kpc, sr < 2*r_half)
    sfar_reg = sr > 2*r_half
    sdsk_reg = sr > 2*u.kpc

    star_tot = np.sum(smass)
    star_inr = np.sum(smass[sinr_reg])
    star_otr = np.sum(smass[sotr_reg])
    star_far = np.sum(smass[sfar_reg])
    star_dsk = np.sum(smass[sdsk_reg])

    my_particle_data[sub_id]['total_star'] = star_tot
    my_particle_data[sub_id]['inner_star'] = star_inr
    my_particle_data[sub_id]['outer_star'] = star_otr
    my_particle_data[sub_id]['far_star']   = star_far
    my_particle_data[sub_id]['disk_star']  = star_dsk

    if args.local:
        my_particle_data[sub_id]['satellite'] = sat[sub_id]
        
particle_lst = comm.gather(my_particle_data, root=0)

if rank==0:
    
    # Assemble full dictionary from components
    all_particle_data = {}
    for dic in particle_lst:
        for k,v in dic.items():
            all_particle_data[k] = v

    # Save dictionary to CSV
    names = [('id',int)]
    names.extend([(s,float) if s!='satellite' else (s, bool) 
                  for s in all_particle_data[sub_id].keys()])

    d = np.recarray(len(all_particle_data), names)
    names = np.array(names) # from list of tuples for ease of use

    for i, sub in enumerate(all_particle_data.items()):
        sub_id, sub_dict = sub
        d[i]['id'] = sub_id
        for name in names[:,0]:

            if name=='id':
                continue

            try:
                d[i][name] = sub_dict[name].value
            except AttributeError:
                d[i][name] = sub_dict[name]

    np.savetxt(folder+'parent_particle_data.csv', d, header=', '.join(names[:,0]))
