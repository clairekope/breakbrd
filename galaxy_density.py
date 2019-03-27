import os
import readsubfHDF5
import numpy as np
import astropy.units as u
from operator import itemgetter
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder, snapnum, littleh, omegaL/M
from utilities import * 

z = args.z
a0 = 1/(1+z)

if rank==0:
    if os.path.isfile(folder+'subhalo_mass_positions.csv'):
        sub_dat = np.genfromtxt(folder+'subhalo_mass_positions.csv', skip_header=1, delimiter=',')

    else:
        min_mass = littleh # 1e10 Msun in 1/1e10 Msun / h
        max_mass = 100 * littleh # 1e12 Msun
        search_query = "?mass_stars__gt=" + str(min_mass)

        cut1 = get(url_sbhalos + search_query)
        cut1['count']
        cut1 = get(url_sbhalos + search_query, {'limit':cut1['count'], 'order_by':'id'})
        
        if args.local:
            subs = np.array([sub['id'] for sub in cut1['results']], dtype='i')
            cat = readsubfHDF5.subfind_catalog(args.local, snapnum,
                                               keysel=['SubhaloPos','SubhaloMass'])
            sub_dat = np.hstack(( subs.reshape(subs.size,1), 
                                  cat.SubhaloMass[subs].reshape(subs.size,1),
                                  cat.SubhaloPos[subs] ))
            del cat

        else:
            keys = ('id','mass','pos_x','pos_y','pos_z')
            sub_dat = np.array([ itemgetter(*keys)(get(sub['url'])) \
                                 for sub in cut1['results'] ])

        np.savetxt(folder+'subhalo_mass_positions.csv', sub_dat, delimiter=',',
                   header='Sub Id, Total Mass, Sub X Pos, Sub Y Pos, Sub Z Pos')

    # Make copies so arrays are contiguous in memory
    sub_ids = sub_dat[:,0].astype(np.int32).copy()
    mass = sub_dat[:,1].copy()
    pos = sub_dat[:,2:].copy()
    del sub_dat
    
else:
    sub_ids = None
    mass = None
    pos = None

my_subs = scatter_work(sub_ids, rank, size)
sub_ids = comm.bcast(sub_ids, root=0)
mass = comm.bcast(mass, root=0)
pos = comm.bcast(pos, root=0)

boxsize = 75000

good_ids = np.where(my_subs > -1)[0]
my_densities = {}

for sub_id in my_subs[good_ids]:
    try:       
        me = np.argwhere(sub_ids==sub_id)[0,0]
    except:
        print(sub_id, np.argwhere(sub_ids==sub_id))
#        continue
    my_x, my_y, my_z = pos[me]
    other_pos = np.delete(pos, me, axis=0) # delete self; use only positions
    other_mass = np.delete(mass, me, axis=0)
    
    assert other_pos.shape[0] == pos.shape[0]-1
    assert other_pos.shape[1] == pos.shape[1]
    
    other_mass = other_mass * u.Msun * 1e10/littleh 

    x_rel = periodic_centering(other_pos[:,0], my_x, boxsize) * u.kpc * a0/littleh
    y_rel = periodic_centering(other_pos[:,1], my_y, boxsize) * u.kpc * a0/littleh
    z_rel = periodic_centering(other_pos[:,2], my_z, boxsize) * u.kpc * a0/littleh
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2).to(u.Mpc)

    neighbors = r < 2*u.Mpc

    my_densities[sub_id] = (np.sum(neighbors), np.sum(other_mass[neighbors]))

densities_lst = comm.gather(my_densities, root=0)

if rank == 0:
    dens = -1*np.ones( (sub_ids.size, 3) )
    i = 0
    for dic in densities_lst:
        for k, v in dic.items():
            dens[i] = (k, v[0], v[1].value)
            i += 1

    sort = np.argsort(dens[:,0])

    np.savetxt(folder+'local_densities.csv', dens[sort], delimiter=',',
               header='Sub ID, n(r < 2 Mpc), m(r < 2 Mpc) [Msun]')
