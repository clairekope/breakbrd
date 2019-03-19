import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import readsubfHDF5
import readhaloHDF5
import snapHDF5
import pickle
from astropy.units import Msun, yr
from utilities import *

if rank == 0:
    treebase = '/mnt/xfs1/home/sgenel/myceph/PUBLIC/IllustrisTNG100/'
    filein = treebase
    snapnum = 99
    snapstr = "%03d" %snapnum
    print("snapshot ", snapnum)
    cat = readsubfHDF5.subfind_catalog(treebase, snapnum, keysel=['SubhaloMassType','SubhaloHalfmassRadType','GroupFirstSub', 'SubhaloGrNr', 'SubhaloPos'])

    # Get list of halos
    h_small=0.6774
    mgal_str = cat.SubhaloMassType[:,4].copy() * 1.e10 / h_small  #intrinsic units is 10^10 Msun
    subids = np.arange(mgal_str.size)
    ids = subids[np.logical_and(mgal_str > 1e10, mgal_str < 1e12)]
    print("number of galaxies > 1e10 and < 1e12: ", ids.size)
    sublist = ids

    # Get supplementary halo information
    rhalf_str = cat.SubhaloHalfmassRadType[:,4].copy()*1.0/h_small
    mgal_gas = cat.SubhaloMassType[:,0].copy() * 1.e10 / h_small  #intrinsic units is 10^10 Msun 
    pos0 = cat.SubhaloPos[:,0].copy() # have to be in code units for centering
    pos1 = cat.SubhaloPos[:,1].copy()
    pos2 = cat.SubhaloPos[:,2].copy()

    # Print fraction of sattelites
    id_first = cat.GroupFirstSub.copy()
    sat = (subids != id_first[cat.SubhaloGrNr[subids]]) # bool; sat or not
    del cat
    gc.collect()
    print("frac of sample that are satellites: ", np.sum(sat[ids])*1.0/len(ids))

else:
    # Variable names need to be declared for any data you want to distribute
    sublist = None
    pos0 = None
    pos1 = None
    pos2 = None
    sat = None
    mgal_str = None
    mgal_gas = None
    rhalf_str = None

# This helper function from utilities.py pads and scatters the arrays
halo_subset = scatter_work(sublist, rank, size, dtype=np.int64)
pos0 = comm.bcast(pos0, root = 0)
pos1 = comm.bcast(pos1, root = 0)
pos2 = comm.bcast(pos2, root = 0)
sat = comm.bcast(sat, root = 0)
rhalf_str = comm.bcast(rhalf_str, root = 0)
mgal_str = comm.bcast(mgal_str, root = 0)
mgal_gas = comm.bcast(mgal_gas, root = 0)

treebase = '/mnt/xfs1/home/sgenel/myceph/PUBLIC/IllustrisTNG100/'
snapnum = 99


boxsize = 75000.0
z = 2.22044604925031e-16
a0 = 1/(1+z)
G = 4.302e-6 # [kpc Msun^-1 (km/s)^2 ]
h_small = 0.6774
H0 = h_small * 100
omegaM = 0.2726
omegaL = 0.6911

# Because scattered arrays have to be the same size, they are padded with -1
good_ids = np.where(halo_subset > -1)[0]
start = 1

my_part_data = {}

for sub in halo_subset[good_ids]:
    
    # go through all the z=0  subhalos in the sample

    rhalfstar = float(rhalf_str[sub])
    mtotstar = mgal_str[sub]
    mtotgas = mgal_gas[sub]
    subpos0 = pos0[sub]
    subpos1 = pos1[sub]
    subpos2 = pos2[sub]

    readhaloHDF5.reset()

    if rhalfstar > 2.0 and mtotstar > 0:

        starp = readhaloHDF5.readhalo(treebase, "snap", snapnum, "POS ", 4, -1, sub, long_ids=True, double_output=False).astype("float32") 
        stara = readhaloHDF5.readhalo(treebase, "snap", snapnum, "GAGE", 4, -1, sub, long_ids=True, double_output=False).astype("float32") 
        starmass = readhaloHDF5.readhalo(treebase, "snap", snapnum, "MASS", 4, -1, sub, long_ids=True, double_output=False).astype("float32") 

        starp = starp[stara > 0]  # this removes the few wind particles that can be in the stellar particle list
        starmass = starmass[stara > 0]*1.0e10/h_small

        starx = periodic_centering(starp[:,0], subpos0, boxsize) * a0/h_small
        stary = periodic_centering(starp[:,1], subpos1, boxsize) * a0/h_small
        starz = periodic_centering(starp[:,2], subpos2, boxsize) * a0/h_small
        stard = np.sqrt(starx**2 + stary**2 + starz**2)

        star_inr_reg = stard < 2.0
        star_mid_reg = np.logical_and(stard > 2.0, stard < rhalfstar)
        star_out_reg = np.logical_and(stard > rhalfstar, stard < 2.0*rhalfstar)
        star_far_reg = stard > 2.0*rhalfstar

        star_tot = np.sum(starmass)
        star_inr = np.sum(starmass[star_inr_reg])
        star_mid = np.sum(starmass[star_mid_reg])
        star_out = np.sum(starmass[star_out_reg])
        star_far = np.sum(starmass[star_far_reg])

        my_part_data[sub] = {}
        my_part_data[sub]['total_star'] = star_tot * Msun
        my_part_data[sub]['inner_star'] = star_inr * Msun
        my_part_data[sub]['mid_star']   = star_mid * Msun
        my_part_data[sub]['outer_star'] = star_out * Msun
        my_part_data[sub]['far_star']   = star_far * Msun

        if mtotgas > 0:
            gasp = readhaloHDF5.readhalo(treebase, "snap", snapnum, "POS ", 0, -1, sub, long_ids=True, double_output=False).astype("float32") 
            gasmass = readhaloHDF5.readhalo(treebase, "snap", snapnum, "MASS", 0, -1, sub, long_ids=True, double_output=False).astype("float32") 
            sfr = readhaloHDF5.readhalo(treebase, "snap", snapnum, "SFR ", 0, -1, sub, long_ids=True, double_output=False).astype("float32") 

            gasmass = gasmass*1.0e10/h_small

            gasx = periodic_centering(gasp[:,0], subpos0, boxsize) * a0/h_small
            gasy = periodic_centering(gasp[:,1], subpos1, boxsize) * a0/h_small
            gasz = periodic_centering(gasp[:,2], subpos2, boxsize) * a0/h_small
            gasd = np.sqrt(gasx**2 + gasy**2 + gasz**2)
            
            gas_inr_reg = gasd < 2.0
            gas_mid_reg = np.logical_and(gasd > 2.0, gasd < rhalfstar)
            gas_out_reg = np.logical_and(gasd > rhalfstar, gasd < 2.0*rhalfstar)
            gas_far_reg = gasd > 2.0*rhalfstar

            gas_tot = np.sum(gasmass)
            gas_inr = np.sum(gasmass[gas_inr_reg])
            gas_mid = np.sum(gasmass[gas_mid_reg])
            gas_out = np.sum(gasmass[gas_out_reg])
            gas_far = np.sum(gasmass[gas_far_reg])

            sfr_tot = np.sum(sfr)
            sfr_inr = np.sum(sfr[gas_inr_reg])
            sfr_mid = np.sum(sfr[gas_mid_reg])
            sfr_out = np.sum(sfr[gas_out_reg])
            sfr_far = np.sum(sfr[gas_far_reg])

            my_part_data[sub]['total_gas'] = gas_tot * Msun
            my_part_data[sub]['inner_gas'] = gas_inr * Msun
            my_part_data[sub]['mid_gas']   = gas_mid * Msun
            my_part_data[sub]['outer_gas'] = gas_out * Msun
            my_part_data[sub]['far_gas']   = gas_far * Msun

            my_part_data[sub]['total_SFR'] = sfr_tot * Msun/yr
            my_part_data[sub]['inner_SFR'] = sfr_inr * Msun/yr
            my_part_data[sub]['mid_SFR']   = sfr_mid * Msun/yr
            my_part_data[sub]['outer_SFR'] = sfr_out * Msun/yr
            my_part_data[sub]['far_SFR']   = sfr_far * Msun/yr

        if sat[sub]:
            satid = 1
        else:
            satid = 0

        my_part_data[sub]['satellite'] = satid

result_lst = comm.gather(my_part_data, root=0)

if rank == 0:
    part_data = {}
    for dic in result_lst:
        for k,v in dic.items():
            part_data[k] = v

    with open(folder+"all_gt1e10_lt1e12_particle_data.pkl", "wb") as f:
        pickle.dump(part_data, f)
