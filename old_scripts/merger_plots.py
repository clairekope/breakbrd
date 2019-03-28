import pickle
import h5py
import os
import glob
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from matplotlib.colors import LogNorm

#######
# Load samples
#######

with open("cut_final_dusty.pkl","rb") as f:
    final_cut = pickle.load(f)
    
with open("parent.pkl","rb") as f:
    parent = pickle.load(f)
    
#######
# Load lists of centrals
#######

central = np.genfromtxt("ClaireDn4000SubhaloIDsCentrals_new.txt", dtype=np.int32)
parent_central = np.genfromtxt("ClaireParentSubhaloIDsCentrals.txt")

satellite = [k for k in final_cut.keys() if k not in central]
parent_satellite = [k for k in parent.keys() if k not in parent_central]

assert len(satellite)+len(central) == len(final_cut)


# File template for merger histories
# Everyone can be found in the "parent" set of files
fparent = "parent-M_r-mergers/forClaireParent_{:d}mergerlistallz.txt"

#######
# Recent mergers for the *full* parent sample
#######
p_mstar = np.zeros(len(parent))
p_mr4_t1 = np.zeros_like(p_mstar)
p_mr4_t6 = np.zeros_like(p_mstar)
p_mr10_t1 = np.zeros_like(p_mstar)
p_mr10_t6 = np.zeros_like(p_mstar)
p_mr100_t1 = np.zeros_like(p_mstar)
p_mr100_t6 = np.zeros_like(p_mstar)

for i, sub_id in enumerate(parent.keys()):
    try:
        data = np.genfromtxt(fparent.format(sub_id), names=True)
    except OSError:
        p_mstar[i] = parent[sub_id]['stellar_mass']
        p_mr4_t1[i] = 0
        p_mr4_t6[i] = 0
        p_mr10_t1[i] = 0
        p_mr10_t6[i] = 0
        p_mr100_t1[i] = 0
        p_mr100_t6[i] = 0
        continue
    mr4  = data['StarsMassratio'] > 0.25 # 1:4 merger
    mr10 = data['StarsMassratio'] > 0.1  # 1:10 merger
    mr100 = data['StarsMassratio'] > 0.01  # 1:100 merger
    t1 = data['SnapshotMerging'] > 128 # snapshot 1.1 Gyr ago
    t6 = data['SnapshotMerging'] > 97  # snapshot 6 Gyr ago
    
    # Sum the number of mergers that meet both 
    mr4_t1 = np.sum(np.logical_and(mr4, t1))
    mr4_t6 = np.sum(np.logical_and(mr4, t6))
    mr10_t1 = np.sum(np.logical_and(mr10, t1))
    mr10_t6 = np.sum(np.logical_and(mr10, t6))
    mr100_t1 = np.sum(np.logical_and(mr100, t1))
    mr100_t6 = np.sum(np.logical_and(mr100, t6))

    p_mstar[i] = parent[sub_id]['stellar_mass']
    p_mr4_t1[i] = mr4_t1
    p_mr4_t6[i] = mr4_t6
    p_mr10_t1[i] = mr10_t1
    p_mr10_t6[i] = mr10_t6
    p_mr100_t1[i] = mr100_t1
    p_mr100_t6[i] = mr100_t6

#######
# Recent mergers for the D4000 satellites
#######
s_mstar = np.zeros(len(final_cut))
s_mr4_t1 = np.zeros_like(s_mstar)
s_mr4_t6 = np.zeros_like(s_mstar)
s_mr10_t1 = np.zeros_like(s_mstar)
s_mr10_t6 = np.zeros_like(s_mstar)
s_mr100_t1 = np.zeros_like(s_mstar)
s_mr100_t6 = np.zeros_like(s_mstar)

for i, sub_id in enumerate(satellite):
    try:
        data = np.genfromtxt(fparent.format(sub_id), names=True)
    except OSError:
        print("File {} not found".format(sub_id))
        s_mstar[i] = parent[sub_id]['stellar_mass']
        s_mr4_t1[i] = 0
        s_mr4_t6[i] = 0
        s_mr10_t1[i] = 0
        s_mr10_t6[i] = 0
        s_mr100_t1[i] = 0
        s_mr100_t6[i] = 0
        continue
    mr4   = data['StarsMassratio'] > 0.25 # 1:4 merger
    mr10  = data['StarsMassratio'] > 0.1  # 1:10 merger
    mr100 = data['StarsMassratio'] > 0.01 # 1:100 merger
    t1 = data['SnapshotMerging'] > 128 # snapshot 1.1 Gyr ago
    t6 = data['SnapshotMerging'] > 97  # snapshot 6 Gyr ago
        
    # Sum the number of mergers that meet both 
    mr4_t1 = np.sum(np.logical_and(mr4, t1))
    mr4_t6 = np.sum(np.logical_and(mr4, t6))
    mr10_t1 = np.sum(np.logical_and(mr10, t1))
    mr10_t6 = np.sum(np.logical_and(mr10, t6))
    mr100_t1 = np.sum(np.logical_and(mr100, t1))
    mr100_t6 = np.sum(np.logical_and(mr100, t6))

    s_mstar[i] = parent[sub_id]['stellar_mass']
    s_mr4_t1[i] = mr4_t1
    s_mr4_t6[i] = mr4_t6
    s_mr10_t1[i] = mr10_t1
    s_mr10_t6[i] = mr10_t6
    s_mr100_t1[i] = mr100_t1
    s_mr100_t6[i] = mr100_t6

#######
# Recent mergers for the D4000 centrals
#######
c_mstar = np.zeros(len(central))
c_mr4_t1 = np.zeros_like(c_mstar)
c_mr4_t6 = np.zeros_like(c_mstar)
c_mr10_t1 = np.zeros_like(c_mstar)
c_mr10_t6 = np.zeros_like(c_mstar)
c_mr100_t1 = np.zeros_like(c_mstar)
c_mr100_t6 = np.zeros_like(c_mstar)

for i, sub_id in enumerate(central):
    try:
        data = np.genfromtxt(fparent.format(sub_id), names=True)
    except OSError:
        print("File {} not found".format(sub_id))
        c_mstar[i] = parent[sub_id]['stellar_mass']
        c_mr4_t1[i] = 0
        c_mr4_t6[i] = 0
        c_mr10_t1[i] = 0
        c_mr10_t6[i] = 0
        c_mr100_t1[i] = 0
        c_mr100_t6[i] = 0
        continue
    mr4   = data['StarsMassratio'] > 0.25 # 1:4 merger
    mr10  = data['StarsMassratio'] > 0.1  # 1:10 merger
    mr100 = data['StarsMassratio'] > 0.01 # 1:100 merger
    t1 = data['SnapshotMerging'] > 128 # snapshot 1.1 Gyr ago
    t6 = data['SnapshotMerging'] > 97  # snapshot 6 Gyr ago
        
    # Sum the number of mergers that meet both 
    mr4_t1 = np.sum(np.logical_and(mr4, t1))
    mr4_t6 = np.sum(np.logical_and(mr4, t6))
    mr10_t1 = np.sum(np.logical_and(mr10, t1))
    mr10_t6 = np.sum(np.logical_and(mr10, t6))
    mr100_t1 = np.sum(np.logical_and(mr100, t1))
    mr100_t6 = np.sum(np.logical_and(mr100, t6))

    c_mstar[i] = parent[sub_id]['stellar_mass']
    c_mr4_t1[i] = mr4_t1
    c_mr4_t6[i] = mr4_t6
    c_mr10_t1[i] = mr10_t1
    c_mr10_t6[i] = mr10_t6
    c_mr100_t1[i] = mr100_t1
    c_mr100_t6[i] = mr100_t6

#######
# Plot merger ratio 1:4
#######

fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(14,4))

counts, binsx, binsy = np.histogram2d(np.log10(p_mstar), p_mr4_t1, range=[[10,12],[0,10]])

p = ax[0].pcolormesh(binsx, binsy, counts.T,  norm=LogNorm(), cmap='gray_r',vmin=1, vmax=683)
a = ptch.Patch(color='lightgray')
b = ax[0].scatter(np.log10(s_mstar), s_mr4_t1, marker='o', color='C1', label='Satellite d4000 < 1.4')
c = ax[0].scatter(np.log10(c_mstar), c_mr4_t1, marker='o', color='C2', label='Central d4000 < 1.4')
ax[0].set_xlim(10,12)
ax[0].set_ylim(0,10)
ax[0].set_title(">1:4 in 1.1 Gyr")
ax[0].legend((a,b,c), ('Parent','Satellite D4000','Central D4000'), loc='upper left')
ax[0].xaxis.set_ticklabels([])

counts, binsx, binsy = np.histogram2d(np.log10(p_mstar), p_mr4_t6, (30,10), range=[[10,12],[0,10]])

ax[1].pcolormesh(binsx, binsy, counts.T,  norm=LogNorm(), cmap='gray_r', vmin=1, vmax=683)
ax[1].scatter(np.log10(s_mstar), s_mr4_t6, marker='o', color='C1')
ax[1].scatter(np.log10(c_mstar), c_mr4_t6, marker='o', color='C2')
ax[1].set_title(">1:4 in 6 Gyr")
ax[1].xaxis.set_ticklabels([])

ax[0].set_xlabel("log($M_*$) [$M_\odot$]")
ax[1].set_xlabel("log($M_*$) [$M_\odot$]")
ax[0].set_ylabel("Number of Mergers")

fig.savefig("mergers_z0_1-4.png")
plt.close(fig)


#######
# Plot merger ratio 1:10
#######

fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(14,4))

counts, binsx, binsy = np.histogram2d(np.log10(p_mstar), p_mr10_t1, (30,10), range=[[10,12],[0,10]])
#print(counts.min(), counts.max())
p = ax[0].pcolormesh(binsx, binsy, counts.T,  norm=LogNorm(), cmap='gray_r',vmin=1, vmax=683)
a = ptch.Patch(color='lightgray')
b = ax[0].scatter(np.log10(s_mstar), s_mr10_t1, marker='o', color='C1', label='Satellite d4000 < 1.4')
c = ax[0].scatter(np.log10(c_mstar), c_mr10_t1, marker='o', color='C2', label='Central d4000 < 1.4')
ax[0].set_xlim(10,12)
ax[0].set_ylim(0,10)
ax[0].set_title(">1:10 in 1.1 Gyr")
ax[0].legend((a,b,c), ('Parent','Satellite D4000','Central D4000'), loc='upper left')

counts, binsx, binsy = np.histogram2d(np.log10(p_mstar), p_mr10_t6, (30,10), range=[[10,12],[0,10]])
#print(counts.min(), counts.max())
ax[1].pcolormesh(binsx, binsy, counts.T,  norm=LogNorm(), cmap='gray_r', vmin=1, vmax=683)
ax[1].scatter(np.log10(s_mstar), s_mr10_t6, marker='o', color='C1')
ax[1].scatter(np.log10(c_mstar), c_mr10_t6, marker='o', color='C2')
ax[1].set_title(">1:10 in 6 Gyr")

ax[0].set_xlabel("log($M_*$) [$M_\odot$]")
ax[1].set_xlabel("log($M_*$) [$M_\odot$]")
ax[0].set_ylabel("Number of Mergers")

fig.savefig("mergers_z0_1-10.png")
plt.close(fig)


#######
# Plot merger ratio 1:100
#######

fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(14,4))

counts, binsx, binsy = np.histogram2d(np.log10(p_mstar), p_mr100_t1, (30,10), range=[[10,12],[0,10]])
#print(counts.min(), counts.max())
p = ax[0].pcolormesh(binsx, binsy, counts.T,  norm=LogNorm(), cmap='gray_r',vmin=1, vmax=683)
a = ptch.Patch(color='lightgray')
b = ax[0].scatter(np.log10(s_mstar), s_mr100_t1, marker='o', color='C1', label='Satellite d4000 < 1.4')
c = ax[0].scatter(np.log10(c_mstar), c_mr100_t1, marker='o', color='C2', label='Central d4000 < 1.4')
ax[0].set_xlim(10,12)
ax[0].set_ylim(0,10)
ax[0].set_title(">1:100 in 1.1 Gyr")
ax[0].legend((a,b,c), ('Parent','Satellite D4000','Central D4000'), loc='upper left')

counts, binsx, binsy = np.histogram2d(np.log10(p_mstar), p_mr100_t6, (30,10), range=[[10,12],[0,10]])
#print(counts.min(), counts.max())
ax[1].pcolormesh(binsx, binsy, counts.T,  norm=LogNorm(), cmap='gray_r', vmin=1, vmax=683)
ax[1].scatter(np.log10(s_mstar), s_mr100_t6, marker='o', color='C1')
ax[1].scatter(np.log10(c_mstar), c_mr100_t6, marker='o', color='C2')
ax[1].set_title(">1:100 in 6 Gyr")

ax[0].set_xlabel("log($M_*$) [$M_\odot$]")
ax[1].set_xlabel("log($M_*$) [$M_\odot$]")
ax[0].set_ylabel("Number of Mergers")

fig.savefig("mergers_z0_1-100.png")
plt.close(fig)


#######
# Mass histograms
#######

# histogram of stellar mass for the parent sample and d4000 halos
# with the centrals and satellites seperately
p_cen_mstar = [parent[sub]['stellar_mass'] for sub in parent_central]
p_gen_mstar = [parent[sub]['stellar_mass'] for sub in parent_satellite]
s_cen_mstar = [final_cut[sub]['stellar_mass'] for sub in central]
s_gen_mstar = [final_cut[sub]['stellar_mass'] for sub in satellite]


fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
n, bins, p = ax[0].hist(np.log10(p_cen_mstar), label="$M_r$ Parent", log=True)
ax[0].hist(np.log10(s_cen_mstar), bins=bins, label="d4000 < 1.4", log=True)
ax[0].set_title("Centrals")
ax[0].set_xlabel("$M_*$ [$M_\odot$]")

n, bins, p = ax[1].hist(np.log10(p_gen_mstar), label="$M_r$ Parent", log=True)
ax[1].hist(np.log10(s_gen_mstar), bins=bins, label="d4000 < 1.4", log=True)
ax[1].set_title("Satellites")
ax[1].set_xlabel("$M_*$ [$M_\odot$]")
ax[1].legend()
fig.savefig("mass_hist_z0_centrals_satellite.png")
plt.close(fig)

