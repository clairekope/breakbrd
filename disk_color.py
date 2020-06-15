from __future__ import print_function, division
import numpy as np
# Functions for getting r-band mag of the halo and g-r color in the disk
from get_magnitudes import *
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder, snapnum, littleh, omegaL/M
from utilities import *

z = args.z
a0 = 1/(1+z)

if rank == 0:
    subhalo_ids = np.genfromtxt(folder+'parent_particle_data.csv', usecols=0).astype(np.int32)

else:
    subhalo_ids = None

halo_subset = scatter_work(subhalo_ids, rank, size)
good_ids = np.where(halo_subset > -1)[0]
    
my_gr = {}

for sub_id in halo_subset[good_ids]:

    subhalo = get(url_sbhalos + str(sub_id))

    if args.mock:
        try:
            gr_color = gr_from_spectra(sub_id)
        except OSError:
            gr_color = np.nan
        my_gr[sub_id] = gr_color

    else:
        try:
            view = np.argmin(rmag_from_fits(sub_id))
            rhalf = subhalo["halfmassrad_stars"]*a0/littleh
            gr_color = gr_from_fits(sub_id, view, rhalf)
            my_gr[sub_id] = gr_color 
        except OSError:
            print("Subhalo {} not found".format(sub_id)); sys.stdout.flush()
            my_gr[sub_id] = np.nan

gr_lst = comm.gather(my_gr, root=0)

if rank==0:
    gr = np.zeros( (subhalo_ids.size, 2) )
    i = 0
    for dic in gr_lst:
        for k, v in dic.items():
            gr[i] = (k, v)
            i += 1

    sort = np.argsort(gr[:,0])
            
    np.savetxt(folder+'disk_color.csv', gr[sort], delimiter=',',
               header='Sub ID, g-r disk color')




