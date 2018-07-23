# coding: utf-8
import requests
import pickle
import numpy as np
import matplotlib.pyplot as plt

def get(path, params=None):
    # make HTTP GET request to path
    headers = {"api-key":"5309619565f744f9248320a886c59bec"}
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

with open("cut3.pkl","rb") as f:
    d = pickle.load(f)
    
url_base = "http://www.illustris-project.org/api/Illustris-1/snapshots/135/subhalos/"

masses = np.empty(len(d))
for i, k in enumerate(d.keys()):
    masses[i] = get(url_base+str(k))['mass_stars']    
masses *= 1e10
masses /= 0.704

plt.hist(np.log10(masses), 25, log=True)
plt.xlabel("$\log_{10}(M_*)\ (M_\odot)$")
plt.ylabel("Subhalos with $g-r>0.655$")
plt.savefig("subhalo_masses_25.png")
