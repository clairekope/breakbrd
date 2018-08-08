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
import pdb

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

use_inst = True # include instantaneous SFR

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
        filename = "stellar_cutouts/" + r.headers['content-disposition'].split("filename=")[1]
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
    #with open("cut4.pkl","rb") as f:
    with open("cut2.5.pkl","rb") as f:
        subs = pickle.load(f)
    sub_list = np.array([k for k in subs.keys()])
    if use_inst:
        #with open("cut4_gas_info.pkl","rb") as f:
        with open("parent_gas_info.pkl","rb") as f:
            inst_sfr = pickle.load(f)
else:
    subs = {}
    sub_list = None
    if use_inst:
        inst_sfr = {}
subs = comm.bcast(subs,root=0)
my_subs = scatter_work(sub_list, rank, size)
if use_inst:
    inst_sfr = comm.bcast(inst_sfr, root=0)
my_cut_radii = {}
my_cut_ssfr = {}
my_all_ssfr = {}

url = "http://www.illustris-project.org/api/Illustris-1/snapshots/135/subhalos/"
cutout = {"stars":
        "Coordinates,GFM_StellarFormationTime,GFM_InitialMass,GFM_Metallicity,Masses,Velocities"}

boxsize = get("http://www.illustris-project.org/api/Illustris-1")['boxsize']
H0 = 0.704 * 100
omegaM = 0.2726
omegaL = 0.7274
timenow = 2.0/(3.0*H0) * 1./np.sqrt(omegaL) \
            * np.log(np.sqrt(omegaL*1./omegaM) \
            + np.sqrt(omegaL*1./omegaM+1))\
            * 3.08568e19/3.15576e16 \
            * u.Gyr

good_ids = np.where(my_subs > -1)[0]

for sub_id in my_subs[good_ids]:
    if use_inst:
        if sub_id not in inst_sfr: # it doesnt have gas!
            continue
            
    file = "stellar_cutouts/cutout_{}.hdf5".format(sub_id)
    if not os.path.isfile(file):
        print("Rank", rank, "downloading",sub_id); sys.stdout.flush()
        get(url + str(sub_id) + "/cutout.hdf5", cutout)
    sub = get(url+str(sub_id))
    r_half = subs[sub_id]['half_mass_rad']
    print("Rank", rank, "processing", sub_id); sys.stdout.flush()

    with h5py.File(file) as f:
        coords = f['PartType4']['Coordinates'][:,:]
        a = f['PartType4']['GFM_StellarFormationTime'][:] # as scale factor
        init_mass = f['PartType4']['GFM_InitialMass'][:]
        curr_mass = f['PartType4']['Masses'][:]

    stars = [a > 0] # throw out wind particles (a < 0)
    x = coords[:,0][stars]
    y = coords[:,1][stars]
    z = coords[:,2][stars]
    x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc / 0.704
    y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc / 0.704
    z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc / 0.704
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    
    init_mass = init_mass[stars] * 1e10 / 0.704 * u.Msun
    curr_mass = curr_mass[stars] * 1e10 / 0.704 * u.Msun
    a = a[stars]

    form_time = 2.0/(3.0*H0) * 1./np.sqrt(omegaL) \
                * np.log(np.sqrt(omegaL*1./omegaM*(a)**3) \
                + np.sqrt(omegaL*1./omegaM*(a)**3+1)) \
                * 3.08568e19/3.15576e16  \
                * u.Gyr
    age = timenow-form_time

    bins = [0, 2, 1*r_half, 2*r_half] * u.kpc
    try:
        binner = np.digitize(r, bins) # index len(bins) is overflow
    except ValueError:
        continue

    time_bins = np.arange(0,14.01,0.01) # 0 to 14 Gyr in 10 Myr bins
    dt = time_bins[1:] - time_bins[:-1] # if we change to unequal bins this supports that

    #
    # Radial SFH Cut
    #

    # Array groups sorted in the same way:
    # 1) form_time, age, init_mass, curr_mass, a, {coordinates}
    # 2) form_history, age_progrssn, mass_history, mass_frac

    for r_bin in range(1, bins.size+1):
        form_history = np.sort(form_time[binner==r_bin])
        sort = np.argsort(form_time[binner==r_bin])
        age_progrssn = age[binner==r_bin][sort]       # sort ages by formation time; oldest first
        mass_history = init_mass[binner==r_bin][sort] # sort initial mass by formation time; 
                                                      #     early mass first
        mass_frac = np.cumsum(mass_history)/np.sum(mass_history)
    
        assert np.all(mass_frac[1:] >= mass_frac[:-1])       # monotonically increasing
        assert np.all(age_progrssn[1:] <= age_progrssn[:-1]) # monotonically decreasing

        if r_bin==1:
            #plt.plot(np.log10(age_progrssn.value), mass_frac, c='pink', # X11 colors
            #         label="$<2\mathrm{\ kpc}$")
            time80_inner = age_progrssn[np.where(mass_frac >= 0.8)[0][0]]

            #
            # sSFR cuts on central r < 2 kpc (SDSS fiber)
            #
            
            # place stars in this radial bin into formation time bins
            t_binner = np.digitize(form_history, bins=time_bins)
            
            # find SFR(t) from the beginning of the universe
            sfr = np.array([ mass_history.value[t_binner==j].sum()/dt[j] for j in range(dt.size) ])
            sfr *= u.Msun/u.Gyr
            sfr = sfr.to(u.Msun/u.yr) # divide by 1e9
            
            if use_inst:
                # Add instantaneous SFR from gas to last bin (i.e., now)
                sfr[-1] = inst_sfr[sub_id]['inner_SFR']
            
            # unweighted avg b/c time bins are currently equal sized
            # denom is current mass in this radial bin
            ssfr_1Gyr = np.average(sfr[-101:])/np.sum(curr_mass[binner==r_bin])
            ssfr_100Myr = np.average(sfr[-11:])/np.sum(curr_mass[binner==r_bin])
            ssfr_50Myr = np.average(sfr[-6:])/np.sum(curr_mass[binner==r_bin])
            
            lim = 1e-11 / u.yr
            
            if ssfr_1Gyr > lim: #or ssfr_100Myr > lim or ssfr_50Myr > lim:
                my_cut_ssfr[sub_id] = subs[sub_id]
                my_cut_ssfr[sub_id]["inner_sSFR_1Gyr"]   = ssfr_1Gyr
                my_cut_ssfr[sub_id]["inner_sSFR_100Myr"] = ssfr_100Myr
                my_cut_ssfr[sub_id]["inner_sSFR_50Myr"]  = ssfr_50Myr

        #elif r_bin==2:
            #plt.plot(np.log10(age_progrssn.value), mass_frac, c='plum',
            #         label="$2\mathrm{\ kpc} - 1R_{M_{1/2}}$")
        elif r_bin==3:
            #plt.plot(np.log10(age_progrssn.value), mass_frac, c='orchid',
            #         label="$1R_{M_{1/2}} - 2R_{M_{1/2}}$")
            time80_outer = age_progrssn[np.where(mass_frac >= 0.8)[0][0]]
        #elif r_bin==4:
            #plt.plot(np.log10(age_progrssn.value), mass_frac, c='purple',
            #         label="$>2R_{M_{1/2}}$")

    #lim = (0, np.max(np.log10(age.value)))
    #plt.xlim(*lim)
    #plt.ylim(0,1)
    #plt.hlines([0.5,0.8], *lim, linestyle=":", color="gray")
    #plt.xlabel("log(Age) (Gyr)")
    #plt.ylabel("Stellar Mass Fraction")
    #plt.legend()
    #print("Rank", rank, "saving", sub_id); sys.stdout.flush()
    #plt.savefig("growth_{}.png".format(sub_id))
    
    if time80_inner < time80_outer:
        my_cut_radii[sub_id] = subs[sub_id]

    my_all_ssfr[sub_id] = ssfr_1Gyr

cut_radii_lst = comm.gather(my_cut_radii, root=0)
cut_ssfr_lst = comm.gather(my_cut_ssfr, root=0)
all_ssfr_lst = comm.gather(my_all_ssfr, root=0)
if rank==0:
    cut_radii = {}
    for dic in cut_radii_lst:
        for k, v in dic.items():
            cut_radii[k] = v            
    #with open("cut_radii.pkl", "wb") as f:
    #    pickle.dump(cut_radii, f)

    cut_ssfr = {}
    for dic in cut_ssfr_lst:
        for k,v in dic.items():
            cut_ssfr[k] = v
    #with open("cut_ssfr.pkl","wb") as f:
    #    pickle.dump(cut_ssfr, f)

    all_ssfr = {}
    for dic in all_ssfr_lst:
        for k,v in dic.items():
            all_ssfr[k] = v
    #with open("cut4_ssfr.pkl","wb") as f:
    #    pickle.dump(all_ssfr, f)
        
    cut5 = {}
    for k in cut_radii.keys():
        if k in cut_ssfr:
            cut5[k] = cut_ssfr[k]
    #with open("cut5.pkl", "wb") as f:
    #    pickle.dump(cut5, f)
    print(len(cut_radii), len(cut_ssfr), len(cut5))
