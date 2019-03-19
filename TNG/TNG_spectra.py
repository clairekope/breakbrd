import sys
import gc
import subprocess as sub
import string
import numpy as np
import readsubfHDF5
import readhaloHDF5
import snapHDF5
import fsps

from utilities import *

inst = args.inst_sfr
dust = args.dusty

sp = fsps.StellarPopulation(zcontinuous=1, sfh=3)

sp.params['add_agb_dust_model'] = True 
sp.params['add_dust_emission'] = True if dust else False
sp.params['add_igm_absorption'] = False
sp.params['add_neb_emission'] = True
sp.params['add_neb_continuum'] = True
sp.params['add_stellar_remnants'] = False
    
sp.params['dust_type'] = 0 # Charlot & Fall type; parameters from Torrey+15
sp.params['dust_tesc'] = np.log10(3e7)
sp.params['dust1'] = 1 if dust else 0.0
sp.params['dust2'] = 1.0/3.0 if dust else 0.0

sp.params['imf_type'] = 1 # Chabrier (2003)

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
    pos0 = cat.SubhaloPos[:,0].copy()
    pos1 = cat.SubhaloPos[:,1].copy()
    pos2 = cat.SubhaloPos[:,2].copy()
    del cat
    gc.collect()

else:
    # Variable names need to be declared for any data you want to distribute
    sublist = None
    pos0 = None
    pos1 = None
    pos2 = None
    mgal_str = None
    mgal_gas = None
    rhalf_str = None

# This helper function from utilities.py pads and scatters the arrays
halo_subset = scatter_work(sublist, rank, size, dtype=np.int64)
pos0 = comm.bcast(pos0, root = 0)
pos1 = comm.bcast(pos1, root = 0)
pos2 = comm.bcast(pos2, root = 0)
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
timenow = 2.0/(3.0*H0) * 1./np.sqrt(omegaL) \
          * np.log(np.sqrt(omegaL*1./omegaM*a0**3) \
          + np.sqrt(omegaL*1./omegaM*a0**3+1))\
          * 3.08568e19/3.15576e16 # Gyr
met_center_bins = np.array([-2.5, -2.05, -1.75, -1.45, -1.15, -0.85, -0.55,
                            -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25,
                            0.4, 0.5]) # log solar, based on Miles

met_bins = np.empty(met_center_bins.size)
half_width = (met_center_bins[1:] - met_center_bins[:-1])/2
met_bins[:-1] = met_center_bins[:-1] + half_width
met_bins[-1] = 9

time_bins = np.arange(0, timenow+0.01, 0.01)
time_avg = (time_bins[:-1] + time_bins[1:])/2 # formation time for fsps
dt = time_bins[1:] - time_bins[:-1] # if we change to unequal bins this supports that

# Because scattered arrays have to be the same size, they are padded with -1
good_ids = np.where(halo_subset > -1)[0]

for sub in halo_subset[good_ids]:

    # go through all the z=0 subhalos in the sample
    
    rhalfstar = rhalf_str[sub]
    mtotstar = mgal_str[sub]
    mtotgas = mgal_gas[sub]
    subpos0 = pos0[sub]
    subpos1 = pos1[sub]
    subpos2 = pos2[sub]

    readhaloHDF5.reset()

    # redefine for every halo because rhalfstar changes
    regions = {'inner': lambda r: r < 2.0,
               'disk': lambda r: np.logical_and(2.0 < r, r < 2*rhalfstar),
               'full': lambda r: np.ones(r.shape, dtype=bool)}

    if rhalfstar > 2.0 and mtotstar > 0:

        starp = readhaloHDF5.readhalo(treebase, "snap", snapnum, "POS ", 4, -1, sub, long_ids=True, double_output=False).astype("float32") 
        stara = readhaloHDF5.readhalo(treebase, "snap", snapnum, "GAGE", 4, -1, sub, long_ids=True, double_output=False).astype("float32")
        starimass = readhaloHDF5.readhalo(treebase, "snap", snapnum, "GIMA", 4, -1, sub, long_ids=True, double_output=False).astype("float32")
        starmetal = readhaloHDF5.readhalo(treebase, "snap", snapnum, "GZ  ", 4, -1, sub, long_ids=True, double_output=False).astype("float32") 

        # this removes the few wind particles that can be in the stellar particle list
        starp = starp[stara > 0]

        # make coords relative to the center of the subhalo in kpc
        starx = periodic_centering(starp[:,0], subpos0, boxsize) * a0/h_small
        stary = periodic_centering(starp[:,1], subpos1, boxsize) * a0/h_small
        starz = periodic_centering(starp[:,2], subpos2, boxsize) * a0/h_small

        stard = np.sqrt(starx**2 + stary**2 + starz**2)

        if inst and mtotgas > 0:
            # Add instantaneous SFR from gas to last bin (i.e., now)
            # This requres using gas information
            gasp = readhaloHDF5.readhalo(treebase, "snap", snapnum, "POS ", 0, -1, sub, long_ids=True, double_output=False).astype("float32") 
            gassfr = readhaloHDF5.readhalo(treebase, "snap", snapnum, "SFR ", 0, -1, sub, long_ids=True, double_output=False).astype("float32") 
            
            # make coords relative to the center of the subhalo in kpc
            gasx = periodic_centering(gasp[:,0], subpos0, boxsize) * a0/h_small
            gasy = periodic_centering(gasp[:,1], subpos1, boxsize) * a0/h_small
            gasz = periodic_centering(gasp[:,2], subpos2, boxsize) * a0/h_small
            gasd = np.sqrt(gasx**2 + gasy**2 + gasz**2)

        for reg_name, reg_func in regions.items():

            reg_stars = reg_func(stard)

            starimass_r = starimass[stara > 0][reg_stars]*1.0e10/h_small
            starmetal_r = starmetal[stara > 0][reg_stars] / 0.0127 # double check Zsun
            stara_r = stara[stara > 0][reg_stars]

            form_time = 2.0/(3.0*H0) * 1./np.sqrt(omegaL) \
                        * np.log(np.sqrt(omegaL*1./omegaM*(stara_r)**3) \
                        + np.sqrt(omegaL*1./omegaM*(stara_r)**3+1)) \
                        * 3.08568e19/3.15576e16

            z_binner = np.digitize(np.log10(starmetal_r), met_bins)

            # one row for each different metallicity's spectrum
            spec_z = np.zeros((met_center_bins.size+1, 5994)) 
            
            for i in range(1, met_center_bins.size): # garbage metallicities have i == 0
                sp.params['logzsol'] = met_center_bins[i]

                # find the SFH for this metallicity
                pop_form = form_time[z_binner==i]
                pop_mass = starimass_r[z_binner==i]
                t_binner = np.digitize(pop_form, time_bins)
                sfr = np.array([ pop_mass[t_binner==j].sum()/dt[j] for j in range(dt.size) ])
                sfr /= 1e9 # to Msun/yr

                if inst and mtotgas > 0:
                    reg_gas = reg_func(gasd)
                    reg_sfr = np.sum(gassfr[reg_gas])
                    sfr[-1] += reg_sfr

                sp.set_tabular_sfh(time_avg, sfr)
                wave, spec = sp.get_spectrum(tage=timenow)
                spec_z[i] = spec

            full_spec = np.nansum(spec_z, axis=0)
            print("Rank",rank,"writing spectra_{:06d}.txt".format(sub)); sys.stdout.flush()
            np.savetxt(folder+"spectra/{}inst/{}dust/{}/spectra_{:06d}.txt".format(
                                                     "no_" if not inst else "",
                                                     "no_" if not dust else "",
                                                     reg_name,
                                                     sub),
                       np.vstack((wave, full_spec)))
