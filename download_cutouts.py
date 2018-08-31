
# coding: utf-8

# In[1]:


import pickle
import os
import sys
import numpy as np
from mpi4py import MPI
from utilities import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

url = "http://www.illustris-project.org/api/Illustris-1/snapshots/135/subhalos/"
star_cutout = {"stars":
        "Coordinates,GFM_StellarFormationTime,GFM_InitialMass,GFM_Metallicity,Masses,Velocities"}
gas_cutout = {"gas":
              "Coordinates,Density,Masses,NeutralHydrogenAbundance,StarFormationRate,InternalEnergy"}

if rank==0:
    with open("parent.pkl","rb") as f:
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
