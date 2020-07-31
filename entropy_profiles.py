import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import h5py
import readsubfHDF5
import readhaloHDF5
import snapHDF5
import numpy as np
import astropy.units as u
from astropy.constants import m_p, k_B
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder, snapnum, littleh, omegaL/M
from utilities import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#rbins = np.logspace(-1, np.log10(300), 151) 

def ent_fit(r, K0, K1, alpha):
    """
    r in kpc, K0 & K1 in eV*cm^2
    """
    return K0 + K1*np.power(r/10, alpha)


if rank==0:

    part_data = np.genfromtxt(folder+"parent_particle_data.csv", names=True)
    sub_list = part_data['id'].astype(np.int32)
    sat = part_data['satellite'].astype(np.bool)

    np.random.seed(6841325)
    subset = np.random.choice(sub_list[~sat], size=100, replace=False)
    
    del part_data
    del sat

else:
    subset = None
    sub_list = None
                                   
#my_subs = scatter_work(sub_list, rank, size)
my_subs = scatter_work(subset, rank, size)
sub_list = comm.bcast(sub_list, root=0)

boxsize = get(url_dset)['boxsize']
z = args.z
a0 = 1/(1+z)

good_ids = np.where(my_subs > -1)[0]
my_profiles = {}

for sub_id in my_subs[good_ids]:
    sub = get(url_sbhalos + str(sub_id))
    rhalf = sub["halfmassrad_stars"] * u.kpc * a0 / littleh

    gas = True
    if not args.local:
        # Read particle data
        gas_file = folder+"gas_cutouts/cutout_{}.hdf5".format(sub_id)

        # Gas
        try:
            with h5py.File(gas_file) as f:
                coords = f['PartType0']['Coordinates'][:,:]
                dens = f['PartType0']['Density'][:]
                mass = f['PartType0']['Masses'][:]
                inte = f['PartType0']['InternalEnergy'][:]
                elec = f['PartType0']['ElectronAbundance'][:]
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
        ne = elec * X_H*dens/m_p
        ent = k_B * temp/ne**(gamma-1)
        ent = ent.to('eV cm^2', equivalencies=u.temperature_energy())

        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc * a0/littleh
        y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc * a0/littleh
        z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc * a0/littleh
        r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)

        mass = mass * 1e10 / littleh * u.Msun

        #
        # Make bins for r & K; plot mass dist
        #

        rbins = np.logspace(np.log10(r.value.min()), np.log10(r.value.max()))
        Kbins = np.logspace(np.log10(ent.value.min()), np.log10(ent.value.max()))

        stat, rbins, Kbins, binnum = binned_statistic_2d(r, ent, mass, 
                                                         bins=(rbins,Kbins))

        fig, ax = plt.subplots()

        mesh = ax.pcolormesh(rbins, Kbins, stat.T,cmap='magma')

        ax.set_xscale('log')
        ax.set_yscale('log')

        cb = fig.colorbar(mesh)
        cb.set_label('<Mass> (Msun)')

        #
        # Find weighted radial profile
        #
        
        rbinner = np.digitize(r.value, rbins) # 0 & len(rbins) are under/overflow
        binned_r = rbins[:-1] + np.diff(rbins) 
        binned_ent = np.ones_like(binned_r)*np.nan * u.eV*u.cm**2
        binned_std = np.ones_like(binned_r)*np.nan * u.eV*u.cm**2
        
        for i in range(1, rbins.size):
            this_bin = rbinner==i
            if np.sum(mass[this_bin]) != 0:
                binned_ent[i-1] = np.average(ent[this_bin],
                                             weights=mass[this_bin])
                binned_std[i-1] = np.sqrt(np.average(
                    np.power(ent[this_bin]-binned_ent[i-1], 2),
                    weights=mass[this_bin])
                )

        #
        # Plot profile
        #

        # ax.fill_between(binned_r, binned_ent-binned_std, binned_ent+binned_std, 
        #                  color='C0', alpha=0.15)
        # ax.plot(binned_r, binned_ent+binned_std, '--', color='C0')
        # ax.plot(binned_r, binned_ent-binned_std, '--', color='C0')

        r_cut = binned_r > 2*rhalf.value
        ax.plot(binned_r[r_cut], binned_ent[r_cut])
        ax.axvline(2*rhalf.value, ls=':', color='k')

        #
        # Fit & plot
        #

        mask = np.logical_and( np.isfinite(binned_ent), r_cut )

        try:
            params, cov = curve_fit(ent_fit, binned_r[mask], binned_ent[mask],
                                    sigma=binned_ent[mask], absolute_sigma=True)

        except RuntimeError: # could not fit
            fig.savefig('{:d}_fit.png'.format(sub_id))
            plt.close(fig)
            continue 

        ax.plot(binned_r[r_cut], ent_fit(binned_r[r_cut], *params))
        
        fig.savefig('{:d}_fit.png'.format(sub_id))
        plt.close(fig)

#     else: # no gas
#         my_profiles[sub_id] = np.nan

# profile_list = comm.gather(my_profiles, root=0)

# if rank==0:

#     all_profiles = np.zeros( (len(sub_list), nbins) ) # 1 id col, nbins-1 data col
#     i=0
#     for dic in profile_list:
#         for k,v in dic.items():
#             all_profiles[i,0] = k
#             all_profiles[i,1:] = v
#             i+=1

#     sort = np.argsort(all_profiles[:,0])

#     header = "SubID"
#     for r in radii:
#         header += " {:.2f}".format(r)

#     np.savetxt(folder+'entropy_profiles.csv', all_profiles[sort], 
#                delimiter=',', header=header)

    
        
