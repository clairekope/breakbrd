
# coding: utf-8

# In[1]:


import pickle
import requests
import os
import sys
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# In[2]:


def get(path, params=None, type="stellar"):
    # make HTTP GET request to path
    headers = {"api-key":"5309619565f744f9248320a886c59bec"}
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = "{}_cutouts/".format(type) + r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r


# In[3]:


def scatter_work(array, mpi_rank, mpi_size, root=0):
    if mpi_rank == root:
        scatter_total = array.size
        mod = scatter_total % mpi_size
        if mod != 0:
            print("Padding array for scattering...")
            pad = -1 * np.ones(mpi_size - mod, dtype='i')
            array = np.concatenate((array, pad))
            scatter_total += mpi_size - mod
            assert scatter_total % mpi_size == 0
            assert scatter_total == array.size
    else:
        scatter_total = None
        #array = None

    scatter_total = comm.bcast(scatter_total, root=root)
    subset = np.empty(scatter_total//mpi_size, dtype='i')
    comm.Scatter(array, subset, root=root)

    return subset


# In[ ]:


url = "http://www.illustris-project.org/api/Illustris-1/snapshots/135/subhalos/"
star_cutout = {"stars":
        "Coordinates,GFM_StellarFormationTime,GFM_InitialMass,GFM_Metallicity,Masses,Velocities"}
gas_cutout = {"gas":
              "Coordinates,Density,Masses,NeutralHydrogenAbundance,StarFormationRate,InternalEnergy"}

if rank==0:
    with open("cut2.5.pkl","rb") as f:
        subs = pickle.load(f)
    sub_list = np.array([k for k in subs.keys()])
else:
    subs = {}
    sub_list = None
subs = comm.bcast(subs,root=0)
my_subs = scatter_work(sub_list, rank, size)
good_ids = np.where(my_subs > -1)[0]

for sub_id in my_subs[good_ids]:
    gas_file = "gas_cutouts/cutout_{}.hdf5".format(sub_id)
    if not os.path.isfile(gas_file):
        print("Rank", rank, "downloading gas",sub_id); sys.stdout.flush()
        try:
            get(url + str(sub_id) + "/cutout.hdf5", gas_cutout, 'gas')
        except:
            pass
        
    star_file = "stellar_cutouts/cutout_{}.hdf5".format(sub_id)
    if not os.path.isfile(star_file):
        print("Rank", rank, "downloading gas",sub_id); sys.stdout.flush()
        try:
            get(url + str(sub_id) + "/cutout.hdf5", star_cutout,'stellar')
        except:
            pass
