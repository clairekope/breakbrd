import os
import sys
import numpy as np
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder
from utilities import *

a = 1/(1+args.z)

if args.local:
    print("Using local snapshot data; do not download cutouts. Exiting...")
    sys.exit()

star_cutout = {"stars":
 "Coordinates,GFM_StellarFormationTime,GFM_InitialMass,GFM_Metallicity,Masses,Velocities"}
gas_cutout = {"gas":
 "Coordinates,Density,Masses,NeutralHydrogenAbundance,StarFormationRate,InternalEnergy"}

if rank==0:
    # Get the halos to loop over. It is now "all" of them.
    min_mass = littleh # 1e10 Msun in 1/1e10 Msun / h
    max_mass = 100 * littleh # 1e12 Msun
    search_query = "?mass_stars__gt=" + str(min_mass) \
                 + "&mass_stars__lt=" + str(max_mass) \
                 + "&halfmassrad_stars__gt=" + str(2 / a * littleh) # 2 kpc

    cut1 = get(url_sbhalos + search_query)
    cut1['count']
    cut1 = get(url_sbhalos + search_query, {'limit':cut1['count']})

    sub_list = np.array([sub['id'] for sub in cut1['results']], dtype='i')

else:
    sub_list = None

my_subs = scatter_work(sub_list, rank, size)
good_ids = np.where(my_subs > -1)[0]

if not os.path.isdir(folder+"gas_cutouts"):
    os.mkdir(folder+"gas_cutouts")
if not os.path.isdir(folder+"stellar_cutouts"):
    os.mkdir(folder+"stellar_cutouts")

for sub_id in my_subs[good_ids]:
    gas_file = folder+"gas_cutouts/cutout_{}.hdf5".format(sub_id)
    if not os.path.isfile(gas_file):
        print("Rank", rank, "downloading gas",sub_id); sys.stdout.flush()
        try:
            get(url_sbhalos + str(sub_id) + "/cutout.hdf5", gas_cutout, 
                folder+'gas_cutouts/')
        except Exception as e:
            print(e)
        
    star_file = folder+"stellar_cutouts/cutout_{}.hdf5".format(sub_id)
    if not os.path.isfile(star_file):
        print("Rank", rank, "downloading stellar",sub_id); sys.stdout.flush()
        try:
            get(url_sbhalos + str(sub_id) + "/cutout.hdf5", star_cutout,
                folder+'stellar_cutouts/')
        except:
            pass
