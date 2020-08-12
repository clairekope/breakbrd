import sys
import h5py
import readsubfHDF5
import readhaloHDF5
import snapHDF5
import numpy as np
import astropy.units as u
from astropy.constants import m_p, k_B
from scipy.stats import binned_statistic
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder, snapnum, littleh, omegaL/M
from utilities import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


nbins = 51


if rank==0:

    part_data = np.genfromtxt(folder+"parent_particle_data.csv", names=True)
    sub_list = part_data['id'].astype(np.int32)

    del part_data
else:
    sub_list = None
                                   
my_subs = scatter_work(sub_list, rank, size)
sub_list = comm.bcast(sub_list, root=0)

boxsize = get(url_dset)['boxsize']
z = args.z
a0 = 1/(1+z)

radial_bins = np.linspace(0,2,nbins)
radii = radial_bins[:-1] + np.diff(radial_bins)

good_ids = np.where(my_subs > -1)[0]
my_profiles = {}

for sub_id in my_subs[good_ids]:
    sub = get(url_sbhalos + str(sub_id))
    vmax = sub['vmax'] * u.km/u.s
    vmax_rad = sub['vmaxrad'] * u.kpc * a0/littleh

    gas = True
    if not args.local:
        # Read particle data
        gas_file = folder+"gas_cutouts/cutout_{}.hdf5".format(sub_id)

        # Gas
        try:
            with h5py.File(gas_file) as f:
                coords = f['PartType0']['Coordinates'][:,:]
                dens = f['PartType0']['Density'][:]
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
            inte = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                         "U   ", 0, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")
            elec = readhaloHDF5.readhalo(args.local, "snap", snapnum,
                                         "NE  ", 0, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")

        except AttributeError:
            gas = False


    if gas:
        # For conversion of internal energy to temperature, see
        # https://www.tng-project.org/data/docs/faq/#gen4
        X_H = 0.76
        gamma = 5./3.
        mu = 4/(1 + 3*X_H + 4*X_H*elec) * m_p
        temp = ( (gamma-1) * inte/k_B * mu * 1e10*u.erg/u.g ).to('K')#u.eV, equivalencies=u.temperature_energy())

        dens = dens * 1e10*u.Msun/littleh * (u.kpc*a0/littleh)**-3
        ne = elec * X_H*dens/m_p
        ent = k_B * temp/ne**(gamma-1)

        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc * a0/littleh
        y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc * a0/littleh
        z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc * a0/littleh
        r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    
        # assuming modified NFW used in Voit 2019 so I can use his scalings
        v200 = vmax / (200*u.km/u.s)
        r200 = 237*u.kpc * v200

        if r200 <= vmax_rad:
            print(sub_id, "bad assumption for virial rad"); sys.stdout.flush()

        r_scale = r/r200

        prof = binned_statistic(r_scale, 
                                ent.to('eV cm^2', 
                                       equivalencies=u.temperature_energy()), 
                                'mean', 
                                radial_bins)

        my_profiles[sub_id] = prof[0]

    else: # no gas
        my_profiles[sub_id] = np.nan*np.ones(nbins-1)

profile_list = comm.gather(my_profiles, root=0)

if rank==0:

    all_profiles = np.zeros( (len(sub_list), nbins) ) # 1 id col, nbins-1 data col
    i=0
    for dic in profile_list:
        for k,v in dic.items():
            all_profiles[i,0] = k
            all_profiles[i,1:] = v
            i+=1

    sort = np.argsort(all_profiles[:,0])

    header = "SubID"
    for r in radii:
        header += " {:.2f}".format(r)

    np.savetxt(folder+'entropy_profiles.csv', all_profiles[sort], 
               delimiter=',', header=header)
