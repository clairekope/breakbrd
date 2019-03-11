import pickle
import h5py
import sys
import os
import numpy as np
import astropy.units as u
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder
from utilities import * 

do_inst_cut = False     # "permanently" off

if rank==0:
    # Get the halos to loop over. It is now "all" of them.
    min_mass = littleh # 1e10 Msun in 1/1e10 Msun / h
    max_mass = 100 * littleh # 1e12 Msun
    search_query = "?mass_stars__gt=" + str(min_mass) \
                 + "&mass_stars__lt=" + str(max_mass) \
                 + "&halfmassrad_stars__gt=" + str(2 / a * littleh) # 2 kpc

    cut1 = get(url_sbhalos + search_query)
    cut1['count']
    cut1 = get(url_sbhalos + search_query, {'limit':cut1['count']})

    sub_list = cut1['results']

else:
    subs = {}
    sub_list = None

subs = comm.bcast(subs,root=0)
my_subs = scatter_work(sub_list, rank, size)
if do_inst_cut: my_cut_inst_ssfr = {}
my_all_gas_data= {}

gas_cutout = {"gas":
    "Coordinates,Density,Masses,NeutralHydrogenAbundance,StarFormationRate,InternalEnergy"}
star_cutout = {"star":
  "Coordinates,GFM_StellarFormationTime,GFM_InitialMass,GFM_Metallicity,Masses,Velocities"}

boxsize = get(url_dset)['boxsize']
z = get(url_dset + "snapshots/103")['redshift']
sf = 1/(1+z)
dthresh = 6.4866e-4 # 0.13 cm^-3 in code units

good_ids = np.where(my_subs > -1)[0]

if not os.path.isdir(folder+"gas_cutouts"):
    os.mkdir(folder+"gas_cutouts")
if not os.path.isdir(folder+"stellar_cutouts"):
    os.mkdir(folder+"stellar_cutouts")

for sub_id in my_subs[good_ids]:
    gas_file = folder+"gas_cutouts/cutout_{}.hdf5".format(sub_id)
    star_file = folder+"stellar_cutouts/cutout_{}.hdf5".format(sub_id)

    # Get half mass radius
    sub = get(url_sbhalos+str(sub_id))
    r_half = subs[sub_id]['half_mass_rad'] * u.kpc * sf / littleh

    # Read particle data
    try:
        with h5py.File(gas_file) as f:
            coords = f['PartType0']['Coordinates'][:,:]
            mass = f['PartType0']['Masses'][:]
            dens = f['PartType0']['Density'][:]
            #inte = f['PartType0']['InternalEnergy'][:]
            #HI = f['PartType0']['NeutralHydrogenAbundance'][:]
            sfr = f['PartType0']['StarFormationRate'][:]
    except KeyError:
        print(sub_id); sys.stdout.flush()
        continue
    try:
        with h5py.File(star_file) as f:
            scoords = f['PartType4']['Coordinates'][:]
            smass = f['PartType4']['Masses'][:]
            a = f['PartType4']['GFM_StellarFormationTime']
    except KeyError:
        print(sub_id, "no stars"); sys.stdout.flush()
        continue

    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc * sf/littleh
    y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc * sf/littleh
    z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc * sf/littleh
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    mass = mass * 1e10 / littleh * u.Msun
    sfr = sfr * u.Msun/u.yr
    
    inner_region = r < 2*u.kpc
    mid_region   = np.logical_and(r > 2*u.kpc, r < r_half)
    outer_region = np.logical_and(r > r_half,  r < 2*r_half)
    far_region   = r > 2*r_half
    
    inner_dense = np.logical_and(r < 2*u.kpc,  dens > dthresh)
    mid_dense   = np.logical_and(mid_region,   dens > dthresh)
    outer_dense = np.logical_and(outer_region, dens > dthresh)
    far_dense   = np.logical_and(r > 2*r_half, dens > dthresh)
    
    inner_sfr = np.sum(sfr[inner_region])
    mid_sfr   = np.sum(sfr[mid_region])
    outer_sfr = np.sum(sfr[outer_region])
    far_sfr   = np.sum(sfr[far_region])
    
    my_all_gas_data[sub_id] = {}
    my_all_gas_data[sub_id]['inner_SFR'] = inner_sfr
    my_all_gas_data[sub_id]['total_SFR'] = np.sum(sfr)
    my_all_gas_data[sub_id]['inner_gas'] = np.sum(mass[inner_region])
    my_all_gas_data[sub_id]['total_gas'] = np.sum(mass)
    my_all_gas_data[sub_id]['inner_sfe'] = inner_sfr  / np.sum(mass[inner_dense])
    my_all_gas_data[sub_id]['mid_sfe']   = mid_sfr    / np.sum(mass[mid_dense])
    my_all_gas_data[sub_id]['outer_sfe'] = outer_sfr  / np.sum(mass[outer_dense])
    my_all_gas_data[sub_id]['far_sfe']   = far_sfr    / np.sum(mass[far_dense])
    my_all_gas_data[sub_id]['total_sfe'] = np.sum(sfr)/ np.sum(mass[dens > dthresh])
                               
    sx = scoords[:,0]
    sy = scoords[:,1]
    sz = scoords[:,2]
    sx_rel = periodic_centering(sx, sub['pos_x'], boxsize) * u.kpc * sf/littleh
    sy_rel = periodic_centering(sy, sub['pos_y'], boxsize) * u.kpc * sf/littleh
    sz_rel = periodic_centering(sz, sub['pos_z'], boxsize) * u.kpc * sf/littleh
    sr = np.sqrt(sx_rel**2 + sy_rel**2 + sz_rel**2)    
    smass = smass * 1e10 / littleh * u.Msun

    ssfr = inner_sfr / np.sum(smass[sr < 2*u.kpc]) 
    
    my_all_gas_data[sub_id]['inner_sSFR'] = ssfr
    if ssfr > 1e-11/u.yr and do_inst_cut:
        my_cut_inst_ssfr[sub_id] = subs[sub_id]
        my_cut_inst_ssfr[sub_id]['inner_inst_sSFR'] = ssfr

if do_inst_cut:
    cut_ssfr_lst = comm.gather(my_cut_inst_ssfr, root=0)

all_gas_lst = comm.gather(my_all_gas_data, root=0)

if rank==0:
    if do_inst_cut:
        cut_ssfr = {}
        for dic in cut_ssfr_lst:
            for k,v in dic.items():
                cut_ssfr[k] = v
        with open(folder+"cut_inst_ssfr.pkl","wb") as f:
            pickle.dump(cut_ssfr, f)
    
    all_gas = {}
    for dic in all_gas_lst:
        for k,v in dic.items():
            all_gas[k] = v
    with open(folder+"{}_gas_info.pkl".format(
                        "parent" if do_parent else "cut3_g-r"
                                              ),"wb") as f:
        pickle.dump(all_gas,f)
