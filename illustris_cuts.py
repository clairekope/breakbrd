from __future__ import print_function, division
import os
import sys
import pickle
import numpy as np
from copy import deepcopy

# Functions for getting r-band mag of the halo and g-r color in the disk
from get_magnitudes import *
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder, snapnum, littleh, omegaL/M
from utilities import *


if rank == 0:
    z = args.z
    a = 1/(1+z)
else:
    a = None
a = comm.bcast(a, root=0)

#
# Get galaxies with 1e10 Msun < stellar mass < 1e12 Msun
#
if rank == 0:

    total = get(url_sbhalos)
    print("Total subhalos:",total['count'])

    # Only subhalos (galaxies) with $M_*>10^{10}$ have stellar mocks 
    #([source](http://www.illustris-project.org/data/docs/specifications/#sec4a))
    
    # convert log solar masses into group catalog units
    min_mass = littleh # 1e10 Msun in 1/1e10 Msun / h
    max_mass = 100 * littleh # 1e12 Msun 
    search_query = "?mass_stars__gt=" + str(min_mass) \
                 + "&mass_stars__lt=" + str(max_mass) \
                 + "&halfmassrad_stars__gt=" + str(2 / a * 0.704) # 2 kpc
    
    cut1 = get(url_sbhalos + search_query)
    scatter_total = cut1['count']
    print("Galaxies from mass cut:", cut1['count'])


#
# Cut on $M_R < -19$
#
if not os.path.isfile(folder+"cut2_M_r_cut2_M_r_parent.pkl"):
   
    if rank == 0:
        # Re-get `cut1` with all desired subhalos so I don't have to paginate
        cut1 = get(url_sbhalos + search_query, {'limit':cut1['count']})
        subhalo_ids = np.array([sub['id'] for sub in cut1['results']], dtype='i')
    else:
        subhalo_ids = None

    halo_subset = scatter_work(subhalo_ids, rank, size)
    
    # will gather my_cut2_M_r into cut2_M_r
    my_cut2_M_r = {}
    
    # ignore padding added for scattering
    good_ids = np.where(halo_subset > -1)[0]

    for sub_id in halo_subset[good_ids]:

        if args.tng:
            pass

        else:
            try:
                my_cut2_M_r[sub_id] = load_individual(sub_id)
            except OSError:
                print("Subhalo {} not found".format(sub_id)); sys.stdout.flush()
                continue

    cut2_M_r_lst = comm.gather(my_cut2_M_r, root=0)
    if rank==0:
        cut2_M_r = {}
        for dic in cut2_M_r_lst:
            for k, v in dic.items():
                cut2_M_r[k] = v
        with open(folder+"cut2_M_r_cut2_M_r_parent.pkl", "wb") as f:
            pickle.dump(cut2_M_r, f)
    else:
        cut2_M_r = None
else: # cut2_M_r dict already generated
    if rank == 0:
        with open(folder+"cut2_M_r_cut2_M_r_parent.pkl","rb") as f:
            cut2_M_r = pickle.load(f)
        print(folder + "cut2_M_r_cut2_M_r_parent.pkl exists")
    else:
        cut2_M_r = None

# broadcast dict (either created or read in)
cut2_M_r = comm.bcast(cut2_M_r, root=0)
if rank == 0:
    print("Galaxies from M_r cut:",len(cut2_M_r))
    cut2_M_r_subhalos = np.array([k for k in cut2_M_r.keys()])
else:
    cut2_M_r_subhalos = None  


#
# Cut on g-r in disk
#
if not os.path.isfile(folder+"cut3_g-r.pkl"):

    halo_subset2 = scatter_work(cut2_M_r_subhalos, rank, size)
    good_ids = np.where(halo_subset2 > -1)

    my_cut3_gr = {}

    for sub_id in halo_subset2[good_ids]:

        if args.tng:
            pass
        else:
            my_cut3_gr[sub_id] = gr_from_fits(sub_id, cut2_M_r)

    cut3_gr_lst = comm.gather(my_cut3_gr, root=0)
    if rank==0:
        cut3_gr = {}
        for dic in cut3_gr_lst:
            for k, v in dic.items():
                cut3_gr[k] = v
            
        with open(folder+"cut3_g-r.pkl", "wb") as f:
            pickle.dump(cut3_gr, f)
    else:
        cut3_gr = None
else: # cut3_gr dict already generated                                                                
    if rank == 0:
        with open(folder+"cut3_g-r.pkl","rb") as f:
            cut3 = pickle.load(f)
        print(folder+"cut3_g-r.pkl exists")
    else:
        cut3_gr = None


