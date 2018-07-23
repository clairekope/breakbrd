import os
import requests

headers = {"api-key":"5309619565f744f9248320a886c59bec"}

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
       
    # for binary data (FITS or HDF5)
    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

snap_url = "http://www.illustris-project.org/api/Illustris-1/snapshots/135/subhalos/"

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
