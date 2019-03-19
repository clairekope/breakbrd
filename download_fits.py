import os
import sys
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder, snapnum, littleh, omegaL/M
from utilities import *

if args.mock:
    print("Generating own mock images; do not download fits. Exiting...")
    sys.exit()

a = 1/(1+args.z)

# convert log solar masses into group catalog units
min_mass = littleh # 1e10 Msun in 1/1e10 Msun / h
max_mass = 100 * littleh # 1e12 Msun 
search_query = "?mass_stars__gt=" + str(min_mass) \
             + "&mass_stars__lt=" + str(max_mass) \
             + "&halfmassrad_stars__gt=" + str(2 / a * littleh) # 2 kpc

# get galaxies with min_mass < mass < max_mass
cut1 = get(url_sbhalos + search_query)
print(cut1['count'])
cut1 = get(url_sbhalos + search_query, {'limit':cut1['count']})

if not os.path.isdir(folder+"illustris_fits"):
    os.mkdir(folder+"illustris_fits")

for subhalo in cut1['results']:

    sub_id = subhalo['id']

    if not os.path.isfile(folder + "illustris_fits/broadband_{}{}.fits".format(
                                    'rest_' if args.z!=0.0 else '', sub_id) ):

        if args.tng:
            rband_url = url_sbhalos + str(sub_id) + "/skirt/broadband_sdss.fits"
        else:
            rband_url = url_sbhalos + str(sub_id) + "/stellar_mocks/broadband.fits"

        print("Downloading subhalo {}".format(sub_id))
    
        try:
            get(rband_url, fpath = folder+"illustris_fits/")
        except requests.HTTPError:
            print("Subhalo {} not found".format(sub_id))

    else:
        print("Subhalo {} exists".format(sub_id))
