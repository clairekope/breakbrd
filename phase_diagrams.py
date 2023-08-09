import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import gc
import sys
import h5py
import readsubfHDF5
import readhaloHDF5
import snapHDF5
import numpy as np
import astropy.units as u
from astropy.constants import m_p, k_B, G
from scipy.stats import binned_statistic_2d

# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder, snapnum, littleh, omegaL/M
from utilities import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


nbins = 100
r_edges = np.logspace(-1, 0, nbins+1)
binned_r = r_edges[:-1] + np.diff(r_edges)


if rank==0:

    min_mass = 100 * littleh # 1e12 Msun
    max_mass = 1000 * littleh # 1e13 Msun
    search_query = "?mass_dm__gt=" + str(min_mass) \
                 + "&mass_dm__lt=" + str(max_mass)

    cut = get(url_sbhalos + search_query)
    cut = get(url_sbhalos + search_query, {'limit':cut['count'], 'order_by':'id'})

    sub_list = np.array([sub['id'] for sub in cut['results']], dtype='i')

    if args.local:
        cat = readsubfHDF5.subfind_catalog(args.local, snapnum, #grpcat=False, subcat=False,
                                           keysel=['GroupFirstSub','SubhaloGrNr'])
        sat = (sub_list != cat.GroupFirstSub[cat.SubhaloGrNr[sub_list]])
        del cat
        gc.collect()

    print(sub_list.size, 
          get(url_sbhalos + search_query + "&mass_stars__gt="+str(littleh))['count'])

    np.random.seed(6841325)
    subset = np.random.choice(sub_list[~sat], size=10, replace=False)
    
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

H0 = littleh * 100 * u.km/u.s/u.Mpc

good_ids = np.where(my_subs > -1)[0]

for sub_id in my_subs[good_ids]:

    sub = get(url_sbhalos + str(sub_id))
    dm_halo = sub["mass_dm"] * 1e10 / littleh * u.Msun
    r_half = sub["halfmassrad_stars"] * u.kpc * a0/littleh

    r200 = (G*dm_halo/(100*H0**2))**(1/3)
    r200 = r200.to('kpc')

    gas = True
    if not args.local:
        # Read particle data
        # gas_file = folder+"gas_cutouts/cutout_{}.hdf5".format(sub_id)
        gas_file = "/home/claire/cutout_{}.hdf5".format(sub_id)
        
        # Gas
        try:
            with h5py.File(gas_file) as f:
                coords = f['PartType0']['Coordinates'][:,:]
                dens = f['PartType0']['Density'][:]
                mass = f['PartType0']['Masses'][:]
                inte = f['PartType0']['InternalEnergy'][:]
                elec = f['PartType0']['ElectronAbundance'][:] #x_e = n_e/n_H

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
        # Calculate Temperature
        #

        # For conversion of internal energy to temperature, see
        # https://www.tng-project.org/data/docs/faq/#gen4
        X_H = 0.76
        gamma = 5./3.
        mu = 4/(1 + 3*X_H + 4*X_H*elec) * m_p
        temp = ( (gamma-1) * inte/k_B * mu * 1e10*u.erg/u.g ).to('K')

        dens = dens * 1e10*u.Msun/littleh * (u.kpc*a0/littleh)**-3
        dens = dens.to(u.g/u.cm**3)

        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc * a0/littleh
        y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc * a0/littleh
        z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc * a0/littleh
        r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)

        mass = mass * 1e10 / littleh * u.Msun

        stat, d_edges, t_edges, binner = \
            binned_statistic_2d(dens, temp, mass, statistic='sum',
                                bins=[np.logspace(-30.5, -23, 64),
                                      np.logspace(3.5, 7, 64)])
        
        p = plt.pcolormesh(d_edges, t_edges, stat.T,
                           norm=LogNorm(vmin=1e6, vmax=1e9))
        c = plt.colorbar(p)
        c.set_label('Total Gas Mass [M$_\odot$]')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Density [g/cm$^3$]')
        plt.ylabel('Temperature [K]')
        plt.savefig(str(sub_id)+'_total_density_temperature_mass.png')
        plt.clf()

        # Only "CGM"
        CGM = r > 2*r_half
        # stat, d_edges, t_edges, binner = \
        #     binned_statistic_2d(dens[CGM], temp[CGM], mass[CGM],
        #                         statistic='sum',
        #                         bins=[np.logspace(-30.5, -23, 64),
        #                               np.logspace(4, 7, 64)])
        
        # p = plt.pcolormesh(d_edges, t_edges, stat.T,
        #                    norm=LogNorm(vmin=1e6, vmax=1e9))
        # c = plt.colorbar(p)
        # c.set_label('Total Gas Mass [M$_\odot$]')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('Density [g/cm$^3$]')
        # plt.ylabel('Temperature [K]')
        # plt.savefig(str(sub_id)+'_CGM_density_temperature_mass.png')
        # plt.clf()

        # Only not CGM
        stat, d_edges, t_edges, binner = \
            binned_statistic_2d(dens[~CGM], temp[~CGM], mass[~CGM],
                                statistic='sum',
                                bins=[np.logspace(-30.5, -23, 64),
                                      np.logspace(3.5, 7, 64)])
        
        p = plt.pcolormesh(d_edges, t_edges, stat.T,
                           norm=LogNorm(vmin=1e6, vmax=1e9))
        c = plt.colorbar(p)
        c.set_label('Total Gas Mass [M$_\odot$]')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Density [g/cm$^3$]')
        plt.ylabel('Temperature [K]')
        plt.title(r"$\log(2r_{\rm half}/r_{200}) =$" + "{}".format(np.log10(2*r_half/r200)))
        plt.savefig(str(sub_id)+'_galaxy_density_temperature_mass.png')
        plt.clf()

        # Only within 1 r_half
        inner = r < r_half
        stat, d_edges, t_edges, binner = \
            binned_statistic_2d(dens[inner], temp[inner], mass[inner],
                                statistic='sum',
                                bins=[np.logspace(-30.5, -23, 64),
                                      np.logspace(3.5, 7, 64)])
        
        p = plt.pcolormesh(d_edges, t_edges, stat.T,
                           norm=LogNorm(vmin=1e6, vmax=1e9))
        c = plt.colorbar(p)
        c.set_label('Total Gas Mass [M$_\odot$]')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Density [g/cm$^3$]')
        plt.ylabel('Temperature [K]')
        plt.title(r"$\log(r_{\rm half}/r_{200}) =$" + "{}".format(np.log10(r_half/r200)))
        plt.savefig(str(sub_id)+'_1rhalf_density_temperature_mass.png')
        plt.clf()
