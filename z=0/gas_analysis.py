import pickle
import h5py
import sys
import os
import numpy as np
import astropy.units as u
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from utilities import *
#import readsubfHDF5

offline = False
if offline:
    import readsubfHDF5

do_parent = True
#do_inst_cut = False

if rank==0:
    if not do_parent:
        with open("cut3_g-r.pkl","rb") as f:
            subs = pickle.load(f)
    else:
        with open("parent.pkl","rb") as f:
            subs = pickle.load(f)
    sub_list = np.array([k for k in subs.keys()])
else:
    subs = {}
    sub_list = None
subs = comm.bcast(subs,root=0)
my_subs = scatter_work(sub_list, rank, size)
my_cut_inst_ssfr = {}
my_all_gas_data= {}

if not offline:
    url = "http://www.illustris-project.org/api/Illustris-1/snapshots/135/subhalos/"
    gas_cutout = {"gas":
        "Coordinates,Density,Masses,NeutralHydrogenAbundance,StarFormationRate,InternalEnergy"}
    star_cutout = {"star":
        "Coordinates,GFM_StellarFormationTime,GFM_InitialMass,GFM_Metallicity,Masses,Velocities"}
else:
    treebase = "/mnt/xfs1/home/sgenel/myceph/PUBLIC/Illustris-1/"
    if rank==0:
        cat = readsubfHDF5.subfind_catalog(treebase, 135, keysel=['SubhaloPos'])
    else:
        cat = None
    cat = comm.bcast(cat, root=0)

boxsize = get("http://www.illustris-project.org/api/Illustris-1")['boxsize']
z = get("http://www.illustris-project.org/api/Illustris-1/snapshots/135")['redshift']
sf = 1/(1+z)
dthresh = 6.4866e-4 # 0.13 cm^-3 in code units

good_ids = np.where(my_subs > -1)[0]

for sub_id in my_subs[good_ids]:
    gas_file = "gas_cutouts/cutout_{}.hdf5".format(sub_id)
    star_file = "stellar_cutouts/cutout_{}.hdf5".format(sub_id)
    if not offline:
        if not os.path.isfile(gas_file):
            print("Rank", rank, "downloading gas", sub_id); sys.stdout.flush()
            try:
                get(url + str(sub_id) + "/cutout.hdf5", gas_cutout, 'gas_cutouts/')
            except requests.exceptions.HTTPError:
                print("Gas", sub_id, "not found"); sys.stdout.flush()
                continue
        if not os.path.isfile(star_file):
            print("Rank", rank, "downloading star", sub_id); sys.stdout.flush()
            try:
                get(url + str(sub_id) + "/cutout.hdf5", star_cutout, 'stellar_cutouts/')
            except requests.exceptions.HTTPError:
                print("Stars", sub_id, "not found"); sys.stdout.flush()
                continue
        sub = get(url+str(sub_id))
    else:
        pos = cat.SubhaloPos[sub_id,3]
        sub = {'pos_x':pos[0],
               'pos_y':pos[1],
               'pos_z':pos[2]}
    r_half = subs[sub_id]['half_mass_rad']*u.kpc

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
    x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc * sf/0.704
    y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc * sf/0.704
    z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc * sf/0.704
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    mass = mass * 1e10 / 0.704 * u.Msun
    sfr = sfr * u.Msun/u.yr
    
    inner_region = r < 2*u.kpc
    mid_region   = np.logical_and(r > 2*u.kpc, r < r_half)
    outer_region = np.logical_and(r > r_half,  r < 2*r_half)
    far_region   = r > 2*r_half
    disk_region  = np.logical_and(r > 2*u.kpc, r < 2*r_half)
    
    inner_dense = np.logical_and(r < 2*u.kpc,  dens > dthresh)
    mid_dense   = np.logical_and(mid_region,   dens > dthresh)
    outer_dense = np.logical_and(outer_region, dens > dthresh)
    far_dense   = np.logical_and(r > 2*r_half, dens > dthresh)
    
    inner_sfr = np.sum(sfr[inner_region])
    mid_sfr   = np.sum(sfr[mid_region])
    outer_sfr = np.sum(sfr[outer_region])
    far_sfr   = np.sum(sfr[far_region])
    disk_sfr  = np.sum(sfr[disk_region])
    
    my_all_gas_data[sub_id] = {}

    my_all_gas_data[sub_id]['inner_SFR'] = inner_sfr
    my_all_gas_data[sub_id]['mid_SFR']   = mid_sfr
    my_all_gas_data[sub_id]['outer_SFR'] = outer_sfr
    my_all_gas_data[sub_id]['far_SFR']   = far_sfr
    my_all_gas_data[sub_id]['disk_SFR']  = disk_sfr
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
    sx_rel = periodic_centering(sx, sub['pos_x'], boxsize) * u.kpc * sf/0.704
    sy_rel = periodic_centering(sy, sub['pos_y'], boxsize) * u.kpc * sf/0.704
    sz_rel = periodic_centering(sz, sub['pos_z'], boxsize) * u.kpc * sf/0.704
    sr = np.sqrt(sx_rel**2 + sy_rel**2 + sz_rel**2)    
    smass = smass * 1e10 / 0.704 * u.Msun

    inner_sregion = sr < 2*u.kpc
    mid_sregion   = np.logical_and(sr > 2*u.kpc, sr < r_half)
    outer_sregion = np.logical_and(sr > r_half,  sr < 2*r_half)
    far_sregion   = sr > 2*r_half
    disk_sregion  = np.logical_and(sr > 2*u.kpc, sr < 2*r_half)
    
    my_all_gas_data[sub_id]['inner_sSFR'] = inner_sfr / np.sum(smass[inner_sregion]) 
    my_all_gas_data[sub_id]['mid_sSFR']   = mid_sfr   / np.sum(smass[mid_sregion])
    my_all_gas_data[sub_id]['outer_sSFR'] = outer_sfr / np.sum(smass[outer_sregion])
    my_all_gas_data[sub_id]['far_sSFR']   = far_sfr   / np.sum(smass[far_sregion])
    my_all_gas_data[sub_id]['disk_sSFR']  = disk_sfr  / np.sum(smass[disk_sregion])
    my_all_gas_data[sub_id]['total_sSFR'] = np.sum(sfr) / np.sum(smass)

#     if ssfr > 1e-11/u.yr and do_inst_cut:
#         my_cut_inst_ssfr[sub_id] = subs[sub_id]
#         my_cut_inst_ssfr[sub_id]['inner_inst_sSFR'] = ssfr

# if do_inst_cut:
#     cut_ssfr_lst = comm.gather(my_cut_inst_ssfr, root=0)
all_gas_lst = comm.gather(my_all_gas_data, root=0)
if rank==0:
    # if do_inst_cut:
    #     cut_ssfr = {}
    #     for dic in cut_ssfr_lst:
    #         for k,v in dic.items():
    #             cut_ssfr[k] = v
    #     with open("cut_inst_ssfr.pkl","wb") as f:
    #         pickle.dump(cut_ssfr, f)
    
    all_gas = {}
    for dic in all_gas_lst:
        for k,v in dic.items():
            all_gas[k] = v
    with open("{}_gas_info.pkl".format("parent" if do_parent else "cut3_g-r"),"wb") as f:
        pickle.dump(all_gas,f)
