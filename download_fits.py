import os
from utilities import *

snap_url = "http://www.illustris-project.org/api/Illustris-1/snapshots/103/subhalos/"

# convert log solar masses into group catalog units
min_mass = 0.704

# form query
search_query = "?mass_stars__gt=" + str(min_mass)

# get galaxies with mass > min_mass
cut1 = get(snap_url + search_query)
cut1['count']
cut1 = get(snap_url + search_query, {'limit':cut1['count']})

for subhalo in cut1['results']:
  sub_id = subhalo['id']
  if not os.path.isfile("illustris_fits/broadband_{}.fits".format(sub_id)):
    rband_url = snap_url + str(sub_id) + "/stellar_mocks/broadband.fits"
    try:
        get(rband_url)
    except requests.HTTPError:
        print("Subhalo {} not found".format(sub_id))
