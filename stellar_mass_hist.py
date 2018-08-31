# coding: utf-8
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utilities import *

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
