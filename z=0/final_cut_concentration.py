###########################################################
#
# Make plots about SFR and gas concentration, and gas mass vs stellar mass
#
###########################################################

import pickle
import requests
import h5py
import pdb
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import astropy.units as u
import seaborn as sns
from matplotlib.ticker import LinearLocator, LogLocator, FixedFormatter, NullFormatter
from matplotlib.legend_handler import HandlerTuple
from utilities import *

with open("parent.pkl", "rb") as f:
    parent = pickle.load(f)
with open("parent_gas_info.pkl", "rb") as f:  
    all_inner_gas_info = pickle.load(f) # within r = 2 kpc
with open("cut_final_dusty.pkl","rb") as f:
    final_cut = pickle.load(f)

# Ascertain which are centrals
first_in_group = []
for i in range(8):
    file = "group_data/groups_135.{}.hdf5".format(i)
    with h5py.File(file) as f:
        gfs = f['Group']['GroupFirstSub'][:]
        for sub in final_cut.keys():
            if sub in gfs:
                first_in_group.append(sub)

len(first_in_group)

##################
# Make data arrays
##################

# Parent sample
full_inner_inst_sfr = np.empty(len(all_inner_gas_info))
full_total_inst_sfr = np.empty_like(full_inner_inst_sfr)

full_inner_gas = np.empty_like(full_inner_inst_sfr)
full_total_gas = np.empty_like(full_inner_gas)

full_mstar = np.empty_like(full_inner_inst_sfr)
full_sfe = np.empty((full_mstar.size, 4))
full_total_sfe = np.empty_like(full_inner_inst_sfr)

# D4000 satellites
inner_inst_sfr = np.empty(len(final_cut)-len(first_in_group))
total_inst_sfr = np.empty_like(inner_inst_sfr)

inner_inst_ssfr = np.empty_like(inner_inst_sfr)

inner_gas = np.empty_like(inner_inst_sfr)
total_gas = np.empty_like(inner_gas)

mstar = np.empty_like(inner_inst_sfr)
sfe = np.empty((mstar.size, 4))
total_sfe = np.empty_like(inner_inst_sfr)

# D4000 centrals
cntrl_inner_inst_sfr = np.empty(len(first_in_group))
cntrl_total_inst_sfr = np.empty_like(cntrl_inner_inst_sfr)

cntrl_inner_inst_ssfr = np.empty_like(cntrl_inner_inst_sfr)

cntrl_inner_gas = np.empty_like(cntrl_inner_inst_sfr)
cntrl_total_gas = np.empty_like(cntrl_inner_gas)

cntrl_mstar = np.empty_like(cntrl_inner_inst_sfr)
cntrl_sfe = np.empty((cntrl_mstar.size, 4))
cntrl_total_sfe = np.empty_like(cntrl_inner_inst_sfr)

# Fill arrays
for i, k in enumerate(all_inner_gas_info.keys()):
    full_inner_inst_sfr[i] = all_inner_gas_info[k]['inner_SFR'].value
    full_total_inst_sfr[i] = all_inner_gas_info[k]['total_SFR'].value
    
    full_inner_gas[i] = all_inner_gas_info[k]['inner_gas'].value
    full_total_gas[i] = all_inner_gas_info[k]['total_gas'].value
    
    full_mstar[i] = parent[k]['stellar_mass']   
    
    full_sfe[i, 0] = all_inner_gas_info[k]['inner_sfe'].value
    full_sfe[i, 1] = all_inner_gas_info[k]['mid_sfe'].value
    full_sfe[i, 2] = all_inner_gas_info[k]['outer_sfe'].value
    full_sfe[i, 3] = all_inner_gas_info[k]['far_sfe'].value
    full_total_sfe[i] = all_inner_gas_info[k]['total_sfe'].value
    
i = 0
j = 0
for k in final_cut.keys():
    if k not in first_in_group:
        inner_inst_sfr[i] = all_inner_gas_info[k]['inner_SFR'].value
        total_inst_sfr[i] = all_inner_gas_info[k]['total_SFR'].value

        inner_inst_ssfr[i] = all_inner_gas_info[k]['inner_sSFR'].value

        inner_gas[i] = all_inner_gas_info[k]['inner_gas'].value
        total_gas[i] = all_inner_gas_info[k]['total_gas'].value

        mstar[i] = final_cut[k]['stellar_mass']

        sfe[i, 0] = all_inner_gas_info[k]['inner_sfe'].value
        sfe[i, 1] = all_inner_gas_info[k]['mid_sfe'].value
        sfe[i, 2] = all_inner_gas_info[k]['outer_sfe'].value
        sfe[i, 3] = all_inner_gas_info[k]['far_sfe'].value
        total_sfe[i] = all_inner_gas_info[k]['total_sfe'].value
        i += 1
    else:
        cntrl_inner_inst_sfr[j] = all_inner_gas_info[k]['inner_SFR'].value
        cntrl_total_inst_sfr[j] = all_inner_gas_info[k]['total_SFR'].value

        cntrl_inner_inst_ssfr[j] = all_inner_gas_info[k]['inner_sSFR'].value

        cntrl_inner_gas[j] = all_inner_gas_info[k]['inner_gas'].value
        cntrl_total_gas[j] = all_inner_gas_info[k]['total_gas'].value

        cntrl_mstar[j] = final_cut[k]['stellar_mass']

        cntrl_sfe[j, 0] = all_inner_gas_info[k]['inner_sfe'].value
        cntrl_sfe[j, 1] = all_inner_gas_info[k]['mid_sfe'].value
        cntrl_sfe[j, 2] = all_inner_gas_info[k]['outer_sfe'].value
        cntrl_sfe[j, 3] = all_inner_gas_info[k]['far_sfe'].value
        cntrl_total_sfe[j] = all_inner_gas_info[k]['total_sfe'].value
        j+=1

# Calculate ratios        
full_sfr_ratio = full_inner_inst_sfr/full_total_inst_sfr
sfr_ratio = inner_inst_sfr/total_inst_sfr
cntrl_sfr_ratio = cntrl_inner_inst_sfr/cntrl_total_inst_sfr

log_full_mstar = np.log10(full_mstar)
log_mstar = np.log10(mstar)
log_cntrl_mstar = np.log10(cntrl_mstar)

full_total_inst_ssfr = full_total_inst_sfr/full_mstar
total_inst_ssfr = total_inst_sfr/mstar
cntrl_total_inst_ssfr = cntrl_total_inst_sfr/cntrl_mstar
    
full_gas_ratio = full_inner_gas/full_total_gas
gas_ratio = inner_gas/total_gas
cntrl_gas_ratio = cntrl_inner_gas/cntrl_total_gas

full_sfe_ratio = full_sfe[:,0]/full_total_sfe
sfe_ratio = sfe[:,0]/total_sfe
cntrl_sfe_ratio = cntrl_sfe[:,0]/cntrl_total_sfe

########################
# Inst SFR Concentration
########################

# Some SFR values are garbage
good = np.nonzero(full_total_inst_sfr)

# For contours
n_lvls = 50
gy_colors = sns.light_palette('black', n_lvls)
gy_colors[0] = (1,1,1)

# Generic patch for legend
a = ptch.Patch(color='lightgrey')

# gridspec_kw controls relative widths and heights of subplots
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15,10), 
                       gridspec_kw={'height_ratios':[1, 2],
                                    'width_ratios':[3,3,3,2],
                                    'hspace':0.1, 'wspace':0.15})

# Stellar mass vs SFR ratio
counts, binsx, binsy = np.histogram2d(log_full_mstar[good], full_sfr_ratio[good], 30)
ax[0,0].hist(log_full_mstar[good], density=True, color='lightgray')
ax[0,0].hist(log_mstar, density=True, histtype='step', lw=2)
ax[0,0].hist(log_cntrl_mstar, density=True, histtype='step', lw=2, ls="--")
ax[0,0].xaxis.set_visible(False)
ax[0,0].set_xlim(10,12)
ax[0,0].set_ylim(0,3)

ax[1,0].contourf(counts.T, n_lvls-1, colors=gy_colors,
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
b = ax[1,0].scatter(log_mstar, sfr_ratio, marker='.')
c = ax[1,0].scatter(log_cntrl_mstar, cntrl_sfr_ratio, marker='.')
ax[1,0].set_xlabel("$\mathrm{\log_{10}(M_*)\ [M_\odot]}$")
ax[1,0].set_xlim(10,12)
ax[1,0].set_ylim(0,1)

# Inst SFR vs SFR ratio
counts, binsx, binsy = np.histogram2d(np.log10(full_total_inst_sfr[good]),
                                      full_sfr_ratio[good], 30)
ax[0,1].hist(np.log10(full_total_inst_sfr[good]), density=True, color='lightgray')
ax[0,1].hist(np.log10(total_inst_sfr), density=True, histtype='step', lw=2)
ax[0,1].hist(np.log10(cntrl_total_inst_sfr), density=True, histtype='step', lw=2, ls="--")
ax[0,1].xaxis.set_visible(False)
ax[0,1].yaxis.set_visible(False)
ax[0,1].set_xlim(-1.5,1.5)
ax[0,1].set_ylim(0,3)

ax[1,1].contourf(counts.T, n_lvls-1, colors=gy_colors,
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[1,1].scatter(np.log10(total_inst_sfr), sfr_ratio, marker='.')
ax[1,1].scatter(np.log10(cntrl_total_inst_sfr), cntrl_sfr_ratio, marker='.')
ax[1,1].set_xlabel("$\log_{10}$ Instantaneous SFR(total)")
ax[1,1].set_xlim(-1.5,1.5)
ax[1,1].set_ylim(0,1)
ax[1,1].yaxis.set_visible(False)

# Inst sSFR vs SFR ratio
counts, binsx, binsy = np.histogram2d(np.log10(full_total_inst_ssfr[good]),
                                      full_sfr_ratio[good], 30)
ax[0,2].hist(np.log10(full_total_inst_ssfr[good]), density=True, color='lightgray')
ax[0,2].hist(np.log10(total_inst_ssfr), density=True, histtype='step', lw=2)
ax[0,2].hist(np.log10(cntrl_total_inst_ssfr), density=True, histtype='step', lw=2, ls="--")
ax[0,2].xaxis.set_visible(False)
ax[0,2].yaxis.set_visible(False)
ax[0,2].set_xlim(-12.5,-9)
ax[0,2].set_ylim(0,3)

ax[1,2].contourf(counts.T, n_lvls-1, colors=gy_colors,
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[1,2].scatter(np.log10(total_inst_ssfr), sfr_ratio, marker='.')
ax[1,2].scatter(np.log10(cntrl_total_inst_ssfr), cntrl_sfr_ratio, marker='.')
ax[1,2].set_xlabel("$\log_{10}$ Instantaneous sSFR(total)")
ax[1,2].set_xlim(-12.5,-9)
ax[1,2].set_ylim(0,1)
ax[1,2].yaxis.set_visible(False)

ax[0,3].legend((a,b,c), ('Parent','D4000 Satellites','D4000 Centrals'), loc='lower left')
ax[0,3].set_axis_off()

ax[1,3].hist(full_sfr_ratio[good], density=True, orientation='horizontal', color='lightgray')
ax[1,3].hist(sfr_ratio, density=True, orientation='horizontal', histtype='step', lw=2)
ax[1,3].hist(cntrl_sfr_ratio, density=True, orientation='horizontal', 
             histtype='step', lw=2, ls='--')
ax[1,3].set_ylim(0,1)
ax[1,3].yaxis.set_visible(False)

ax[1,0].set_ylabel("Instantaneous SFR(fiber) / SFR(total)")
fig.savefig("sfr_ratio_z00.png")

###################
# Gas Concentration
###################

# Some SFR values are garbage
good = np.nonzero(full_total_inst_sfr)

# For contours
n_lvls = 50
gy_colors = sns.light_palette('black', n_lvls)
gy_colors[0] = (1,1,1)

# Generic patch for legend
a = ptch.Patch(color='lightgrey')

# gridspec_kw controls relative widths and heights of subplots
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15,10), 
                       gridspec_kw={'height_ratios':[1, 2],
                                    'width_ratios':[3,3,3,2],
                                    'hspace':0.1, 'wspace':0.15})

# Stellar mass vs gas ratio
counts, binsx, binsy = np.histogram2d(log_full_mstar[good], full_gas_ratio[good], 30)
ax[0,0].hist(log_full_mstar[good], density=True, color='lightgray')
ax[0,0].hist(log_mstar, density=True, histtype='step', lw=2)
ax[0,0].hist(log_cntrl_mstar, density=True, histtype='step', lw=2, ls="--")
ax[0,0].xaxis.set_visible(False)
ax[0,0].set_xlim(10,12)
ax[0,0].set_ylim(0,3)

ax[1,0].contourf(counts.T, n_lvls-1, colors=gy_colors,
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
b = ax[1,0].scatter(log_mstar, gas_ratio, marker='.')
c = ax[1,0].scatter(log_cntrl_mstar, cntrl_gas_ratio, marker='.')
ax[1,0].set_xlabel("$\mathrm{\log_{10}(M_*)\ [M_\odot]}$")
ax[1,0].set_xlim(10,12)
ax[1,0].set_ylim(0,0.6)

# Inst SFR vs gas ratio
counts, binsx, binsy = np.histogram2d(np.log10(full_total_inst_sfr[good]),
                                      full_gas_ratio[good], 30)
ax[0,1].hist(np.log10(full_total_inst_sfr[good]), density=True, color='lightgray')
ax[0,1].hist(np.log10(total_inst_sfr), density=True, histtype='step', lw=2)
ax[0,1].hist(np.log10(cntrl_total_inst_sfr), density=True, histtype='step', lw=2, ls="--")
ax[0,1].xaxis.set_visible(False)
ax[0,1].yaxis.set_visible(False)
ax[0,1].set_xlim(-1.5,1.5)
ax[0,1].set_ylim(0,3)

ax[1,1].contourf(counts.T, n_lvls-1, colors=gy_colors,
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[1,1].scatter(np.log10(total_inst_sfr), gas_ratio, marker='.')
ax[1,1].scatter(np.log10(cntrl_total_inst_sfr), cntrl_gas_ratio, marker='.')
ax[1,1].set_xlabel("$\log_{10}$ Instantaneous SFR(total)")
ax[1,1].set_xlim(-1.5,1.5)
ax[1,1].set_ylim(0,0.6)
ax[1,1].yaxis.set_visible(False)

# Inst sSFR vs gas ratio
counts, binsx, binsy = np.histogram2d(np.log10(full_total_inst_ssfr[good]),
                                      full_gas_ratio[good], 30)
ax[0,2].hist(np.log10(full_total_inst_ssfr[good]), density=True, color='lightgray')
ax[0,2].hist(np.log10(total_inst_ssfr), density=True, histtype='step', lw=2)
ax[0,2].hist(np.log10(cntrl_total_inst_ssfr), density=True, histtype='step', lw=2, ls="--")
ax[0,2].xaxis.set_visible(False)
ax[0,2].yaxis.set_visible(False)
ax[0,2].set_xlim(-12.5,-9)
ax[0,2].set_ylim(0,3)

ax[1,2].contourf(counts.T, n_lvls-1, colors=gy_colors,
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[1,2].scatter(np.log10(total_inst_ssfr), gas_ratio, marker='.')
ax[1,2].scatter(np.log10(cntrl_total_inst_ssfr), cntrl_gas_ratio, marker='.')
ax[1,2].set_xlabel("$\log_{10}$ Instantaneous sSFR(total)")
ax[1,2].set_xlim(-12.5,-9)
ax[1,2].set_ylim(0,0.6)
ax[1,2].yaxis.set_visible(False)

ax[0,3].legend((a,b,c), ('Parent','D4000 Satellites','D4000 Centrals'), loc='lower left')
ax[0,3].set_axis_off()

ax[1,3].hist(full_gas_ratio[good], density=True, orientation='horizontal', color='lightgray')
ax[1,3].hist(gas_ratio, density=True, orientation='horizontal', histtype='step', lw=2)
ax[1,3].hist(cntrl_gas_ratio, density=True, orientation='horizontal', 
             histtype='step', lw=2, ls='--')
ax[1,3].set_ylim(0,0.6)
ax[1,3].yaxis.set_visible(False)

ax[1,0].set_ylabel("Gas(fiber) / Gas(total)")
fig.savefig("gas_ratio_z00.png")

##########################
# Gas mass vs stellar mass
##########################

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,5))
fig.tight_layout()
x = np.linspace(10,12)
y1 = x
y2 = np.log10(0.1*10**x)
y3 = np.log10(0.01*10**x)

n_lvls = 50
gy_colors = sns.light_palette('black', n_lvls)
gy_colors[0] = (1,1,1)

a = ptch.Patch(color='lightgrey')
counts, binsx, binsy = np.histogram2d(log_full_mstar, np.log10(full_total_gas), 30)
ax[0].contourf(counts.T, n_lvls-1, colors=gy_colors, extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
b = ax[0].scatter(log_mstar, np.log10(total_gas), marker='.')
b2 = ax[0].scatter(log_cntrl_mstar, np.log10(cntrl_total_gas), marker='.')
c, = ax[0].plot(x,y1,'k--')
d, = ax[0].plot(x,y2,'k-.')
e, = ax[0].plot(x,y3,'k:')
ax[0].set_title("Total")
ax[0].set_xlabel("$\mathrm{M_*\ [M_\odot]}$")
ax[0].set_ylabel("$\mathrm{M_{gas}\ [M_\odot]}$")
ax[0].legend((a,b,b2,c,d,e),['Parent','D4000 Satellites',"D4000 Centrals",'100%','10%','1%'])

good = full_inner_gas!=0
counts, binsx, binsy = np.histogram2d(log_full_mstar[good], np.log10(full_inner_gas[good]), 30)
ax[1].contourf(counts.T, n_lvls-1, colors=gy_colors, extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[1].scatter(log_mstar, np.log10(inner_gas), marker='.')
ax[1].scatter(log_cntrl_mstar, np.log10(cntrl_inner_gas), marker='.')
ax[1].plot(x,y1,'k--')
ax[1].plot(x,y2,'k-.')
ax[1].plot(x,y3,'k:')
ax[1].set_title("Inner $r<2$ kpc")
ax[1].set_xlabel("$\mathrm{M_*\ [M_\odot]}$")

ax[0].set_xlim(10,12)
ax[0].set_ylim(7, 12)
fig.savefig("mass_comparison_z00.png")
