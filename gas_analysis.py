import pickle
import requests
import h5py
import sys
import os
import numpy as np
import astropy.units as u
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9
from mpi4py import MPI
#import readsubfHDF5

offline = False
if offline:
    import readsubHDF5

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def scatter_work(array, mpi_rank, mpi_size, root=0):
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

def get(path, params=None):
    # make HTTP GET request to path
    headers = {"api-key":"5309619565f744f9248320a886c59bec"}
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = "gas_cutouts/" + r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

def periodic_centering(x, center, boxsixe):
    quarter = boxsize/4
    upper_qrt = boxsize-quarter
    lower_qrt = quarter
    
    if center > upper_qrt:
        # some of our particles may have wrapped around to the left half 
        x[x < lower_qrt] += boxsize
    elif center < lower_qrt:
        # some of our particles may have wrapped around to the right half
        x[x > upper_qrt] -= boxsize
    
    return x - center

# MAIN

if rank==0:
    with open("cut4.pkl","rb") as f:
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
    cutout = {"gas":
        "Coordinates,Density,Masses,NeutralHydrogenAbundance,StarFormationRate,InternalEnergy"}
else:
    treebase = "/mnt/xfs1/home/sgenel/myceph/PUBLIC/Illustris-1/"
    if rank==0:
        cat = readsubfHDF5.subfind_catalog(treebase, 135, keysel=['SubhaloPos'])
    else:
        cat = None
    cat = comm.bcast(cat, root=0)

boxsize = get("http://www.illustris-project.org/api/Illustris-1")['boxsize']

good_ids = np.where(my_subs > -1)[0]

for sub_id in my_subs[good_ids]:
    file = "gas_cutouts/cutout_{}.hdf5".format(sub_id)
    if not offline:
        if not os.path.isfile(file):
            print("Rank", rank, "downloading",sub_id); sys.stdout.flush()
            get(url + str(sub_id) + "/cutout.hdf5", cutout)
        sub = get(url+str(sub_id))
    else:
        pos = cat.SubhaloPos[sub_id,3]
        sub = {'pos_x':pos[0],
               'pos_y':pos[1],
               'pos_z':pos[2]}
    r_half = subs[sub_id]['half_mass_rad']*u.kpc

    try:
        with h5py.File(file) as f:
            coords = f['PartType0']['Coordinates'][:,:]
            mass = f['PartType0']['Masses'][:]
            dens = f['PartType0']['Density'][:]
            #inte = f['PartType0']['InternalEnergy'][:]
            #HI = f['PartType0']['NeutralHydrogenAbundance'][:]
            sfr = f['PartType0']['StarFormationRate'][:]
    except KeyError:
        print(sub_id); sys.stdout.flush()
        continue
    with h5py.File("stellar_cutouts/cutout_{}.hdf5".format(sub_id)) as f:
        scoords = f['PartType4']['Coordinates'][:]
        smass = f['PartType4']['Masses'][:]
        a = f['PartType4']['GFM_StellarFormationTime']

    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc / 0.704
    y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc / 0.704
    z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc / 0.704
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    mass = mass * 1e10 / 0.704 * u.Msun
    sfr = sfr * u.Msun/u.yr
    
    inner_region = r < 2*u.kpc
    mid_region   = np.logical_and(r > 2*u.kpc, r < r_half)
    outer_region = np.logical_and(r > r_half,  r < 2*r_half)
    far_region   = r > 2*r_half
    
    inner_dense = np.logical_and(r < 2*u.kpc,  dens > 0.13)
    mid_dense   = np.logical_and(mid_region,   dens > 0.13)
    outer_dense = np.logical_and(outer_region, dens > 0.13)
    far_dense   = np.logical_and(r > 2*r_half, dens > 0.13)
    
    inner_sfr = np.sum(sfr[inner_region])
    mid_sfr   = np.sum(sfr[mid_region])
    outer_sfr = np.sum(sfr[outer_region])
    far_sfr   = np.sum(sfr[far_region])
    
    my_all_gas_data[sub_id] = {}
    my_all_gas_data[sub_id]['inner_SFR'] = inner_sfr
    my_all_gas_data[sub_id]['inner_gas'] = np.sum(mass[inner_region])
    my_all_gas_data[sub_id]['inner_sfe'] = inner_sfr  / np.sum(mass[inner_dense])
    my_all_gas_data[sub_id]['mid_sfe']   = mid_sfr    / np.sum(mass[mid_dense])
    my_all_gas_data[sub_id]['outer_sfe'] = outer_sfr  / np.sum(mass[outer_dense])
    my_all_gas_data[sub_id]['far_sfe']   = far_sfr    / np.sum(mass[far_dense])
    my_all_gas_data[sub_id]['total_sfe'] = np.sum(sfr)/ np.sum(mass[dens > 0.13])
                               
    sx = scoords[:,0]
    sy = scoords[:,1]
    sz = scoords[:,2]
    sx_rel = periodic_centering(sx, sub['pos_x'], boxsize) * u.kpc / 0.704
    sy_rel = periodic_centering(sy, sub['pos_y'], boxsize) * u.kpc / 0.704
    sz_rel = periodic_centering(sz, sub['pos_z'], boxsize) * u.kpc / 0.704
    sr = np.sqrt(sx_rel**2 + sy_rel**2 + sz_rel**2)    
    smass = smass * 1e10 / 0.704 * u.Msun

    ssfr = inner_sfr / np.sum(smass[sr < 2*u.kpc]) 
    
    my_all_gas_data[sub_id]['inner_sSFR'] = ssfr
    if ssfr > 1e-11/u.yr:
        my_cut_inst_ssfr[sub_id] = subs[sub_id]
        my_cut_inst_ssfr[sub_id]['inner_inst_sSFR'] = ssfr

cut_ssfr_lst = comm.gather(my_cut_inst_ssfr, root=0)
all_gas_lst = comm.gather(my_all_gas_data, root=0)
if rank==0:
    cut_ssfr = {}
    for dic in cut_ssfr_lst:
        for k,v in dic.items():
            cut_ssfr[k] = v
    with open("cut_inst_ssfr.pkl","wb") as f:
        pickle.dump(cut_ssfr, f)
    
    all_gas = {}
    for dic in all_gas_lst:
        for k,v in dic.items():
            all_gas[k] = v
    with open("cut4_gas_info.pkl","wb") as f:
        pickle.dump(all_gas,f)
