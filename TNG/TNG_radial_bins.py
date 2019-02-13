import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import sys
import gc
import subprocess as sub
import string
import numpy as np
import matplotlib.pyplot as plt
import readsubfHDF5
import readhaloHDF5
import snapHDF5

import pickle
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def scatter_work(array, mpi_rank, mpi_size, root=0):
    """ will only work if MPI has been initialized by calling script.
        array should only exist on root & be None elsewhere"""
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
        sendbuf = np.empty([mpi_size, scatter_total//mpi_size], dtype='i')
        sendbuf[:] = np.reshape(array, sendbuf.shape)
    else:
        scatter_total = None
        sendbuf = None                                                                               
    scatter_total = comm.bcast(scatter_total, root=root)
    subset = np.empty(scatter_total//mpi_size, dtype='i')
    comm.Scatter(sendbuf, subset, root=root)

    return subset





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
    pos0 = cat.SubhaloPos[:,0].copy() * 1.0 / h_small
    pos1 = cat.SubhaloPos[:,1].copy() * 1.0 / h_small
    pos2 = cat.SubhaloPos[:,2].copy() * 1.0 / h_small

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
halo_subset = scatter_work(sublist, rank, size)
pos0 = comm.bcast(pos0, root = 0)
pos1 = comm.bcast(pos1, root = 0)
pos2 = comm.bcast(pos2, root = 0)
sat = comm.bcast(sat, root = 0)
rhalf_str = comm.bcast(rhalf_str, root = 0)
mgal_str = comm.bcast(mgal_str, root = 0)
mgal_gas = comm.bcast(mgal_gas, root = 0)

# Because scattered arrays have to be the same size, they are padded with -1
good_ids = np.where(halo_subset > -1)
start = 1

# think we don't need this: my_storage = {} # every rank needs their own way of story results, to be combined later
for isub in range(len(halo_subset[good_ids])):
    # do stuff
    
    # go through all the z=0  subhalos in the sample
    treebase = '/mnt/xfs1/home/sgenel/myceph/PUBLIC/IllustrisTNG100/'
    G = 4.302e-6 # [kpc Msun^-1 (km/s)^2 ]
    h_small = 0.6774
    subnr = halo_subset[good_ids][isub]
    subnrstr = '%d' %subnr 
    snapnum = 99
    print("worker ", rank, " of size ", size, ", doing subhalo ", isub,  " of ", len(halo_subset[good_ids]))
    
    rhalfstar = rhalf_str[halo_subset[good_ids][isub]]
    mtotstar = mgal_str[halo_subset[good_ids][isub]]
    mtotgas = mgal_gas[halo_subset[good_ids][isub]]
    subpos0 = pos0[halo_subset[good_ids][isub]]
    subpos1 = pos1[halo_subset[good_ids][isub]]
    subpos2 = pos2[halo_subset[good_ids][isub]]

    readhaloHDF5.reset()
    if rhalfstar > 2.0 and mtotgas > 0 and mtotstar > 0:
        starp = readhaloHDF5.readhalo(treebase, "snap", snapnum, "POS ", 4, -1, subnr, long_ids=True, double_output=False).astype("float32") 
        stara = readhaloHDF5.readhalo(treebase, "snap", snapnum, "GAGE", 4, -1, subnr, long_ids=True, double_output=False).astype("float32") 
        starmass = readhaloHDF5.readhalo(treebase, "snap", snapnum, "MASS", 4, -1, subnr, long_ids=True, double_output=False).astype("float32") 
        starp = starp[(stara > 0)]  # this removes the few wind particles that can be in the stellar particle list
        starmass = starmass[(stara > 0)]*1.0e10/h_small
        starp = (starp*1./h_small - [subpos0,subpos1,subpos2])  #make it relative to the center of the subhalo in kpc
        stard = np.sqrt(starp[:,0]*starp[:,0] + starp[:,1]*starp[:,1] + starp[:,2]*starp[:,2])
        startot = np.log10(np.sum(starmass))
        star2kpc = np.log10(np.sum(starmass[(stard < 2.0)]))
        star2kpc_1re = np.log10(np.sum(starmass[np.logical_and(stard < 2.0, stard < float(rhalfstar))])) #- gas2kpc
        star1re_2re = np.log10(np.sum(starmass[np.logical_and(stard > float(rhalfstar), stard < float(2.0*rhalfstar))])) #- gas2kpc_1re - gas2kpc
        starHalo = np.log10(np.sum(starmass[(stard > 2.0*rhalfstar)]))
        gasp = readhaloHDF5.readhalo(treebase, "snap", snapnum, "POS ", 0, -1, subnr, long_ids=True, double_output=False).astype("float32") 
        gasmass = readhaloHDF5.readhalo(treebase, "snap", snapnum, "MASS", 0, -1, subnr, long_ids=True, double_output=False).astype("float32") 
        sfr = readhaloHDF5.readhalo(treebase, "snap", snapnum, "SFR ", 0, -1, subnr, long_ids=True, double_output=False).astype("float32") 
        gasmass = gasmass*1.0e10/h_small
        gasp = (gasp*1./h_small - [subpos0,subpos1,subpos2]) #make it relative to the center of the subhalo in kpc
        gasd = np.sqrt(gasp[:,0]*gasp[:,0] + gasp[:,1]*gasp[:,1] + gasp[:,2]*gasp[:,2])
        
        gastot = np.log10(np.sum(gasmass))
        gas2kpc = np.log10(np.sum(gasmass[(gasd < 2.0)]))
        gas2kpc_1re = np.log10(np.sum(gasmass[np.logical_and(gasd < 2.0, gasd < float(rhalfstar))])) #- gas2kpc
        gas1re_2re = np.log10(np.sum(gasmass[np.logical_and(gasd > float(rhalfstar), gasd < float(2.0*rhalfstar))])) #- gas2kpc_1re - gas2kpc
        gasHalo = np.log10(np.sum(gasmass[(gasd > 2.0*rhalfstar)]))
        sfrtot = np.log10(np.sum(sfr))
        sfr2kpc = np.log10(np.sum(sfr[(gasd < 2.0)]))
        sfr2kpc_1re = np.log10(np.sum(sfr[np.logical_and(gasd < 2.0, gasd < float(rhalfstar))])) #- gas2kpc
        sfr1re_2re = np.log10(np.sum(sfr[np.logical_and(gasd > float(rhalfstar), gasd < float(2.0*rhalfstar))])) #- gas2kpc_1re - gas2kpc
        sfrHalo = np.log10(np.sum(sfr[(gasd > 2.0*rhalfstar)]))
        if sat[halo_subset[good_ids][isub]]:
            satid = 1
        else:
            satid = 0
 
        if start == 1:
            my_array = np.reshape([halo_subset[good_ids][isub], rhalfstar, startot, star2kpc, star2kpc_1re, star1re_2re, starHalo, gastot, gas2kpc, gas2kpc_1re, gas1re_2re, gasHalo, sfrtot, sfr2kpc, sfr2kpc_1re, sfr1re_2re, sfrHalo, satid], (1, 18))
            start = 0
        else:
            my_array = np.concatenate((my_array, np.reshape([halo_subset[good_ids][isub], rhalfstar, startot, star2kpc, star2kpc_1re, star1re_2re, starHalo, gastot, gas2kpc, gas2kpc_1re, gas1re_2re, gasHalo, sfrtot, sfr2kpc, sfr2kpc_1re, sfr1re_2re, sfrHalo, satid], (1, 18))), axis = 0)

result_lst = comm.gather(my_array, root=0)

if rank == 0:
    array = []
    for li in result_lst:
        for v in li:
            array.append(v)
    array = np.asarray(array)
    print(array.size, array.shape, array)
    array = np.reshape(array, (array.size/18, 18))
    print(array.shape)
    np.savetxt('MISFIRED-TNG_allgt1e10_z05.txt', array, fmt = "%f", header = "0: SubhaloID, 1: HalfMassRadStar [kpc], 2: log10(total stellar mass)[Msun], 3: log10(stellar mass in 2 kpc)[Msun], 4: log10(stellar mass 2kpc < r < 1Rhalf)[Msun], 5: log10(stellar mass 1Rhalf < r < 2Rhalf)[Msun], 6: log10(stellar mass 2Rhalf < r)[Msun], 7: log10(total gas mass)[Msun], 8: log10(gas mass in 2 kpc)[Msun], 9: log10(gas mass 2kpc < r < 1Rhalf)[Msun], 10: log10(gas mass 1Rhalf < r < 2Rhalf)[Msun], 11: log10(gas mass 2Rhalf < r)[Msun], 12: log10(total SFR)[Msun/yr], 13: log10(SFR in 2 kpc)[Msun/yr], 14: log10(SFR 2kpc < r < 1Rhalf)[Msun/yr], 15: log10(SFR 1Rhalf < r < 2Rhalf)[Msun/yr], 16: log10(SFR 2Rhalf < r)[Msun/yr], 17: satellite? (1:yes, 0: central)")
