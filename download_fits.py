import os
import sys
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder
from utilities import *

if args.mock:
    print("Generating own mock images; do not download fits. Exiting...")
    sys.exit()

# convert log solar masses into group catalog units
min_mass = 0.704 # 1 * 10^10 Msun/h 

# form query
search_query = "?mass_stars__gt=" + str(min_mass)

# get galaxies with mass > min_mass
cut1 = get(url_sbhalos + search_query)
cut1['count']
cut1 = get(url_sbhalos + search_query, {'limit':cut1['count']})

if not os.path.isdir(folder+"illustris_fits"):
    os.mkdir(folder+"illustris_fits")

for subhalo in cut1['results']:

  sub_id = subhalo['id']

  if not os.path.isfile(folder + "illustris_fits/broadband_{}{}.fits".format(
                                    'rest_' if args.z!=0.0 else '', sub_id) ):
    rband_url = url_sbhalos + str(sub_id) + "/stellar_mocks/broadband.fits"
    print("Downloading subhalo {}".format(sub_id))
    try:
        get(rband_url, fpath = folder+"illustris_fits/")
    except requests.HTTPError:
        print("Subhalo {} not found".format(sub_id))
  else:
    print("Subhalo {} exits".format(sub_id))
