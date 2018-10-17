
# coding: utf-8

# In[1]:


import pickle
import requests
import h5py
import pdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
#from utilities import *
import astropy.units as u
#import seaborn as sns
#sns.set_palette(sns.color_palette(['steelblue','mediumpurple','yellowgreen']))
#sns.set_palette(sns.color_palette('colorblind'))
from operator import itemgetter
from matplotlib.ticker import LinearLocator, LogLocator, FixedFormatter, NullFormatter
from matplotlib.legend_handler import HandlerTuple
#plt.style.use('seaborn-talk')


# In[2]:


with open("misfired/z=0/cut_final_dusty.pkl","rb") as f:
    final_cut = pickle.load(f)


# In[3]:


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


# In[4]:


def periodic_centering(x, center, boxsixe):
    quarter = boxsize/4
    upper_qrt = boxsize-quarter
    lower_qrt = quarter
    
    if center > upper_qrt:
        # some of our particles may have wrapped around to the left half 
        x[x < lower_qrt] += boxsize
    elif center < lower_qrt:
        # some of our particles may have wrapped around to the right half
        x[x > upper_qrt] -= boxsize
    
    return x - center


# # Download Mock Images

# In[5]:


# for k in final_cut.keys():
#     #try:
#     os.system("wget -O fof_{0:06d}.png --header 'api-key:5309619565f744f9248320a886c59bec' http://www.illustris-project.org/api/Illustris-1/snapshots/103/subhalos/{0}/stellar_mocks/image_fof.png".format(k))
#     os.system("wget -O gz_{0:06d}.png --header 'api-key:5309619565f744f9248320a886c59bec' http://www.illustris-project.org/api/Illustris-1/snapshots/103/subhalos/{0}/stellar_mocks/image_gz.png".format(k))
#     #except requests.HTTPError:
#     #    continue


# # Environment

# In[6]:


first_in_group = []
for i in range(8):
    file = "groups_135.{}.hdf5".format(i)
    with h5py.File(file) as f:
        gfs = f['Group']['GroupFirstSub'][:]
        for sub in final_cut.keys():
            if sub in gfs:
                first_in_group.append(sub)


# In[7]:


print ('len of first_in_group', len(first_in_group))


# ## 10^10 Halos in 2 Mpc

# In[8]:


# min_mass = 1 * 0.704 # 1e10/1e10 * 0.704
# max_mass = 100 * 0.704 # 1e12
# search_query = "?mass_stars__gt=" + str(min_mass)# + "&mass_stars__lt=" + str(max_mass)

# snap_url = "http://www.illustris-project.org/api/Illustris-1/snapshots/103/subhalos/"
# cut1 = get(snap_url + search_query)
# cut1 = get(snap_url + search_query, {'limit':cut1['count']})
with open("misfired/z=0/parent.pkl", "rb") as f:
    parent = pickle.load(f)


first_in_group_parents = []
for i in range(8):
    file = "groups_135.{}.hdf5".format(i)
    with h5py.File(file) as f:
        gfs = f['Group']['GroupFirstSub'][:]
        for sub in parent.keys():
            if sub in gfs:
                first_in_group_parents.append(sub)


# In[7]:


print ('len of first_in_group parents', len(first_in_group_parents))

# In[9]:


#keys = ('id','pos_x','pos_y','pos_z')
#pos = np.array([itemgetter(*keys)(get(sub['url'])) for sub in cut1['results']]) # some fancy shit right here
#pos = np.array([itemgetter(*keys)(get(snap_url+str(sub))) for sub in parent.keys()]) 


# In[10]:


#np.savetxt("all_10^10_subhalo_pos",pos)
pos = np.genfromtxt("all_10^10_subhalo_pos")


# In[11]:


all_id, all_pos = np.split(pos, [1], axis=1)


# In[12]:


boxsize = 75000
#ndens = np.empty((len(final_cut), 2))
ndens_all = np.empty((len(parent)))
#ndens_all = np.empty((len(cut1['results'])))

#print (parent.keys())
#print (all_id)
#print (len(parent.keys()))
#print (len(all_id))

for i, sub_id in enumerate(parent.keys()):
#for i, sub in enumerate(cut1['results']):
#    sub_id = sub['id']
    #print ('where they are the same:',np.argwhere(all_id==sub_id))
    me = np.argwhere(all_id==sub_id)[0,0]
    my_x, my_y, my_z = all_pos[me]
    other_pos = np.delete(all_pos, me, axis=0) # don't count yourself!
    
    assert other_pos.shape[0] == all_pos.shape[0]-1
    assert other_pos.shape[1] == all_pos.shape[1]
    
    x_rel = periodic_centering(other_pos[:,0], my_x, boxsize) * u.kpc / 0.704
    y_rel = periodic_centering(other_pos[:,1], my_y, boxsize) * u.kpc / 0.704
    z_rel = periodic_centering(other_pos[:,2], my_z, boxsize) * u.kpc / 0.704
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2).to(u.Mpc)

    count = np.sum(r < 2*u.Mpc)
    
    #ndens[i] = (sub_id, count)
    ndens_all[i] = count
    parent[sub_id]['N 10^10 2 Mpc'] = count


ndens_central_parent = np.empty(len(first_in_group_parents))
for i, sub in enumerate(first_in_group_parents):
    ndens_central_parent[i] = parent[sub]['N 10^10 2 Mpc']

# In[13]:


boxsize = 75000
#ndens = np.empty((len(final_cut), 2))
ndens = np.empty((len(final_cut)))
                 
for i, sub_id in enumerate(final_cut.keys()):
    try:
        
        me = np.argwhere(all_id==sub_id)[0,0]
    except:
        print(sub_id, np.argwhere(all_id==sub_id))
    my_x, my_y, my_z = all_pos[me]
    other_pos = np.delete(all_pos, me, axis=0) # don't count yourself!
    
    assert other_pos.shape[0] == all_pos.shape[0]-1
    assert other_pos.shape[1] == all_pos.shape[1]
    
    x_rel = periodic_centering(other_pos[:,0], my_x, boxsize) * u.kpc / 0.704
    y_rel = periodic_centering(other_pos[:,1], my_y, boxsize) * u.kpc / 0.704
    z_rel = periodic_centering(other_pos[:,2], my_z, boxsize) * u.kpc / 0.704
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2).to(u.Mpc)

    count = np.sum(r < 2*u.Mpc)
    
    #ndens[i] = (sub_id, count)
    ndens[i] = count
    final_cut[sub_id]['N 10^10 2 Mpc'] = count


# In[14]:


ndens_central = np.empty(len(first_in_group))
for i, sub in enumerate(first_in_group):
    ndens_central[i] = final_cut[sub]['N 10^10 2 Mpc']


# In[15]:


n,bins,p = plt.hist(ndens_all, 15,  label="All $M_r < -19$", log=True)
plt.hist(ndens_central_parent,bins,label="Central $M_r < -19$",log=True)
# also add g-r
plt.hist(ndens, bins, label="All d4000 < 1.4", log=True)
plt.hist(ndens_central, bins, label="Central d4000 < 1.4", log=True)
plt.legend()
plt.xlabel("Number of $M_* > 10^{10}\ M_\odot$ Subhalo Centers w/in 2 Mpc")
plt.savefig("Hist_z0_numsubhalos2mpc_incparentcentrals.png")
plt.clf()
# In[16]:


parent_dens  = np.array([(np.log10(v['stellar_mass']), v['N 10^10 2 Mpc']) for k,v in parent.items()])#  if k not in first_in_group])
parent_central_dens  = np.array([(np.log10(v['stellar_mass']), v['N 10^10 2 Mpc']) for k,v in parent.items() if k in first_in_group_parents])
final_dens   = np.array([(np.log10(v['stellar_mass']), v['N 10^10 2 Mpc']) for k,v in final_cut.items()])# if k not in first_in_group])
central_dens = np.array([(np.log10(v['stellar_mass']), v['N 10^10 2 Mpc']) for k,v in final_cut.items() if k in first_in_group])


# In[20]:


left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

nullfmt = NullFormatter() # no labels

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
fig = plt.figure(1, figsize=(7, 7))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

binsx = np.linspace(10,12,20)
#binsy = np.logspace(0,np.log10(70),70)
binsy = np.arange(71)

x_p = parent_dens[:,0][np.isfinite(parent_dens[:,1])]
y_p = parent_dens[:,1][np.isfinite(parent_dens[:,1])]
x_pc = parent_central_dens[:,0][np.isfinite(parent_central_dens[:,1])]
y_pc = parent_central_dens[:,1][np.isfinite(parent_central_dens[:,1])]

counts, binsx, binsy = np.histogram2d(x_p, y_p, 20)#[binsx,binsy])
n_lvls = 500
#gy_colors = sns.light_palette('black', n_lvls)
#gy_colors[0] = (1,1,1)
#axScatter.contourf(counts.T, n_lvls-1, colors=gy_colors,
#                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()], )
axScatter.contourf(counts.T, n_lvls-1, cmap='binary',
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()], )

x_f = final_dens[:,0][np.isfinite(final_dens[:,1])]
y_f = final_dens[:,1][np.isfinite(final_dens[:,1])]
counts_f, binsx_f, binsy_f = np.histogram2d(x_f, y_f, [binsx,binsy])
axScatter.scatter(x_f, y_f,color='limegreen')# color='C1')

x_c = central_dens[:,0][np.isfinite(central_dens[:,1])]
y_c = central_dens[:,1][np.isfinite(central_dens[:,1])]

counts_c, binsx_c, binsy_c = np.histogram2d(x_f, y_f, [binsx,binsy])
axScatter.scatter(x_c, y_c, marker='x', color='b')#color='C2')


a = axHistx.hist(x_p, bins=binsx, density=True, color='lightgray')[-1][0]
ac = axHistx.hist(x_pc, bins=binsx, density=True, histtype='step',color='k',lw=2,zorder=1)[-1][0]
b = axHistx.hist(x_f, bins=binsx, density=True, histtype='step', color='limegreen', lw=2, zorder=3)[-1][0]
c = axHistx.hist(x_c, bins=binsx, density=True, histtype='step', color='b', lw=2, zorder=2)[-1][0]
axHisty.hist(y_p, bins=binsy, density=True, orientation='horizontal', color='lightgray')
axHisty.hist(y_pc, bins=binsy, density=True, orientation='horizontal', color='k', lw=2, histtype='step', zorder=1)
axHisty.hist(y_f, bins=binsy, density=True, orientation='horizontal', color='limegreen', lw=2, histtype='step', zorder=3)
axHisty.hist(y_c, bins=binsy, density=True, orientation='horizontal', color='b', lw=2, histtype='step', zorder=2)

# a = axHistx.hist(x_p, bins=binsx, density=False, color='lightgray')[-1][0]
# b = axHistx.hist(x_f, bins=binsx, density=False, histtype='step', color='C1', lw=2, zorder=2)[-1][0]
# c = axHistx.hist(x_c, bins=binsx, density=False, histtype='step', color='C2', lw=2, zorder=1)[-1][0]
# axHisty.hist(y_p, bins=binsy, density=False, orientation='horizontal', color='lightgray')
# axHisty.hist(y_f, bins=binsy, density=False, orientation='horizontal', color='C1', lw=2, histtype='step', zorder=2)
# axHisty.hist(y_c, bins=binsy, density=False, orientation='horizontal', color='C2', lw=2, histtype='step', zorder=1)
# axHistx.set_yscale('log')
# axHisty.set_xscale('log')

axScatter.set_xlabel("$M_* \ [M_\odot]$")
axScatter.set_ylabel("N")

axScatter.set_xlim(10,12)
axScatter.set_ylim(0,70)
#axScatter.set_yscale('log')
axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

fig.legend((a,b,c), ("$M_r$ Parent", "All d4000 < 1.4", "Central d4000 < 1.4"), 
           bbox_to_anchor=(left_h, bottom_h), bbox_transform=fig.transFigure, loc="lower left",
           scatteryoffsets=[0.6, 0.5, 0.3125])
fig.suptitle("Number of $M_* > 10^{10}$ Halos w/in 2 Mpc", x=0.5, y=1)

plt.savefig("Hist_z0_2d_galdens_mstar_incparentcentrals.png")
plt.clf()
# ## 10^8 Halos in 1 Mpc

# In[19]:


# min_mass = 0.01 * 0.704 # 1e8/1e10 * 0.704
# search_query = "?mass_stars__gt=" + str(min_mass)

# snap_url = "http://www.illustris-project.org/api/Illustris-1/snapshots/103/subhalos/"
# cut1 = get(snap_url + search_query)
# cut1 = get(snap_url + search_query, {'limit':cut1['count']})
# with open("cut2.5.pkl", "rb") as f:
#     parent = pickle.load(f)


# In[20]:


# keys = ('id','pos_x','pos_y','pos_z')
# pos = np.array([itemgetter(*keys)(get(sub['url'])) for sub in cut1['results']]) # some fancy shit right here
# #pos = np.array([itemgetter(*keys)(get(snap_url+str(sub))) for sub in parent.keys()]) 


# In[21]:


# all_id, all_pos = np.split(pos, [1], axis=1)


# In[22]:


# np.savetxt("all_10^8_subhalo_pos",pos)
# #pos = np.genfromtxt("all_10^8_subhalo_pos")


# In[23]:


# boxsize = 75000
# #ndens = np.empty((len(final_cut), 2))
# ndens_all = np.empty((len(parent)))
# #ndens_all = np.empty((len(cut1['results'])))

# for i, sub_id in enumerate(parent.keys()):
# #for i, sub in enumerate(cut1['results']):
# #    sub_id = sub['id']
#     me = np.argwhere(all_id==sub_id)[0,0]
#     my_x, my_y, my_z = all_pos[me]
#     other_pos = np.delete(all_pos, me, axis=0) # don't count yourself!
    
#     assert other_pos.shape[0] == all_pos.shape[0]-1
#     assert other_pos.shape[1] == all_pos.shape[1]
    
#     x_rel = periodic_centering(other_pos[:,0], my_x, boxsize) * u.kpc / 0.704
#     y_rel = periodic_centering(other_pos[:,1], my_y, boxsize) * u.kpc / 0.704
#     z_rel = periodic_centering(other_pos[:,2], my_z, boxsize) * u.kpc / 0.704
#     r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2).to(u.Mpc)

#     count = np.sum(r < 1*u.Mpc)
    
#     #ndens[i] = (sub_id, count)
#     ndens_all[i] = count
#     parent[sub_id]['N 10^8 1 Mpc'] = count


# In[24]:


# boxsize = 75000
# #ndens = np.empty((len(final_cut), 2))
# ndens = np.empty((len(final_cut)))
                 
# for i, sub_id in enumerate(final_cut.keys()):
#     try:
#         me = np.argwhere(all_id==sub_id)[0,0]
#     except:
#         print(sub_id, np.argwhere(all_id==sub_id))
#     my_x, my_y, my_z = all_pos[me]
#     other_pos = np.delete(all_pos, me, axis=0) # don't count yourself!
    
#     assert other_pos.shape[0] == all_pos.shape[0]-1
#     assert other_pos.shape[1] == all_pos.shape[1]
    
#     x_rel = periodic_centering(other_pos[:,0], my_x, boxsize) * u.kpc / 0.704
#     y_rel = periodic_centering(other_pos[:,1], my_y, boxsize) * u.kpc / 0.704
#     z_rel = periodic_centering(other_pos[:,2], my_z, boxsize) * u.kpc / 0.704
#     r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2).to(u.Mpc)

#     count = np.sum(r < 1*u.Mpc)
    
#     #ndens[i] = (sub_id, count)
#     ndens[i] = count
#     final_cut[sub_id]['N 10^8 1 Mpc'] = count


# In[25]:


# ndens_central = np.empty(len(first_in_group))
# for i, sub in enumerate(first_in_group):
#     ndens_central[i] = final_cut[sub]['N 10^8 1 Mpc']


# In[26]:


# n,bins,p = plt.hist(ndens_all, 15, align='left', label="All $M_r < -19$", log=True)
# # also add g-r
# plt.hist(ndens, bins, align='left', label="All d4000 < 1.4", log=True)
# plt.hist(ndens_central, bins, align='left', label="Central d4000 < 1.4", log=True)
# plt.legend()
# #plt.xlabel("Number of $M_* > 10^{10}\ M_\odot$ Subhalo Centers w/in 2 Mpc")


# In[27]:


# left, width = 0.1, 0.65
# bottom, height = 0.1, 0.65
# bottom_h = left_h = left + width + 0.02

# nullfmt = NullFormatter() # no labels

# rect_scatter = [left, bottom, width, height]
# rect_histx = [left, bottom_h, width, 0.2]
# rect_histy = [left_h, bottom, 0.2, height]

# # start with a rectangular Figure
# fig = plt.figure(1, figsize=(7, 7))

# axScatter = plt.axes(rect_scatter)
# axHistx = plt.axes(rect_histx)
# axHisty = plt.axes(rect_histy)

# # no labels
# axHistx.xaxis.set_major_formatter(nullfmt)
# axHisty.yaxis.set_major_formatter(nullfmt)

# x_p = parent_dens[:,0][np.isfinite(parent_dens[:,1])]
# y_p = parent_dens[:,1][np.isfinite(parent_dens[:,1])]
# counts, binsx, binsy = np.histogram2d(x_p, y_p, 50)
# n_lvls = 500
# gy_colors = sns.light_palette('black', n_lvls)
# gy_colors[0] = (1,1,1)
# axScatter.contourf(counts.T, n_lvls-1, colors=gy_colors,
#                    extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()], )

# x_f = final_dens[:,0][np.isfinite(final_dens[:,1])]
# y_f = final_dens[:,1][np.isfinite(final_dens[:,1])]
# counts, binsx_f, binsy_f = np.histogram2d(x_f, y_f, [binsx,binsy])
# axScatter.scatter(x_f, y_f, marker='.')

# x_c = central_dens[:,0][np.isfinite(central_dens[:,1])]
# y_c = central_dens[:,1][np.isfinite(central_dens[:,1])]
# counts, binsx_c, binsy_c = np.histogram2d(x_f, y_f, [binsx,binsy])
# axScatter.scatter(x_c, y_c, marker='.')

# a = axHistx.hist(x_p, bins=binsx, normed=False, color='lightgray')[-1][0]
# b = axHistx.hist(x_f, bins=binsx, normed=False)[-1][0]
# c = axHistx.hist(x_c, bins=binsx, normed=False)[-1][0]
# axHisty.hist(y_p, bins=binsy, normed=False, orientation='horizontal', color='lightgray')
# axHisty.hist(y_f, bins=binsy, normed=False, orientation='horizontal')
# axHisty.hist(y_c, bins=binsy, normed=False, orientation='horizontal')

# axScatter.set_xlim(10,12)
# axScatter.set_ylim(0,70)
# axHistx.set_xlim(axScatter.get_xlim())
# axHisty.set_ylim(axScatter.get_ylim())
# axHistx.set_yscale('log')
# axHisty.set_xscale('log')

# fig.legend((a,b,c), ("$M_r$ Parent", "All d4000 < 1.4", "Central d4000 < 1.4"), 
#            bbox_to_anchor=(left_h, bottom_h), bbox_transform=fig.transFigure, loc="lower left",
#            scatteryoffsets=[0.6, 0.5, 0.3125])
# fig.suptitle("Number Density of $M_* > 10^{10}$ Halos w/in 2 Mpc", x=0.5, y=1)


# # Make data arrays

# In[5]:


with open("misfired/z=0/parent_gas_info.pkl", "rb") as f:  
    all_inner_gas_info = pickle.load(f) # within r = 2 kpc

with open("misfired/z=0/parent.pkl", "rb") as f:
    parent = pickle.load(f)

# with open("cut4_gas_info.pkl", "rb") as f:  
#     all_inner_gas_info = pickle.load(f) # within r = 2 kpc

# with open("cut4.pkl", "rb") as f:
#     parent = pickle.load(f)


# In[6]:


full_inner_inst_sfr = np.empty(len(all_inner_gas_info))
full_total_inst_sfr = np.empty_like(full_inner_inst_sfr)
#full_total_inst_ssfr = np.empty_like(full_inner_inst_sfr)

full_inner_gas = np.empty_like(full_inner_inst_sfr)
full_total_gas = np.empty_like(full_inner_gas)

full_mstar = np.empty_like(full_inner_inst_sfr)
full_sfe = np.empty((full_mstar.size, 4))
full_total_sfe = np.empty_like(full_inner_inst_sfr)

    
inner_inst_sfr = np.empty(len(final_cut))
total_inst_sfr = np.empty_like(inner_inst_sfr)
#total_inst_ssfr = np.empty_like(inner_inst_sfr)

inner_inst_ssfr = np.empty_like(inner_inst_sfr)
inner_avg_ssfr = np.empty_like(inner_inst_sfr)

inner_gas = np.empty(len(final_cut))
total_gas = np.empty_like(inner_gas)

mstar = np.empty_like(inner_inst_sfr)
sfe = np.empty((mstar.size, 4))
total_sfe = np.empty_like(inner_inst_sfr)


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
    

for i, k in enumerate(final_cut.keys()):
    inner_inst_sfr[i] = all_inner_gas_info[k]['inner_SFR'].value
    total_inst_sfr[i] = all_inner_gas_info[k]['total_SFR'].value
    
    inner_inst_ssfr[i] = all_inner_gas_info[k]['inner_sSFR'].value
    #inner_avg_ssfr[i] = final_cut[k]['inner_sSFR_1Gyr'].value
    
    inner_gas[i] = all_inner_gas_info[k]['inner_gas'].value
    total_gas[i] = all_inner_gas_info[k]['total_gas'].value
    
    mstar[i] = final_cut[k]['stellar_mass']
    
    sfe[i, 0] = all_inner_gas_info[k]['inner_sfe'].value
    sfe[i, 1] = all_inner_gas_info[k]['mid_sfe'].value
    sfe[i, 2] = all_inner_gas_info[k]['outer_sfe'].value
    sfe[i, 3] = all_inner_gas_info[k]['far_sfe'].value
    total_sfe[i] = all_inner_gas_info[k]['total_sfe'].value

full_sfr_ratio = full_inner_inst_sfr/full_total_inst_sfr
log_full_mstar = np.log10(full_mstar)
    
sfr_ratio = inner_inst_sfr/total_inst_sfr
log_mstar = np.log10(mstar)

full_total_inst_ssfr = full_total_inst_sfr/full_mstar
total_inst_ssfr = total_inst_sfr/mstar
    
full_gas_ratio = full_inner_gas/full_total_gas
gas_ratio = inner_gas/total_gas

full_sfe_ratio = full_sfe[:,0]/full_total_sfe
sfe_ratio = sfe[:,0]/total_sfe


# # Instantaneous SFR

# In[30]:


good = np.nonzero(full_total_inst_sfr)
n_lvls = 50
#gy_colors = sns.light_palette('black', n_lvls)
#gy_colors[0] = (1,1,1)

fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(14,5))

#counts, binsx, binsy = np.histogram2d(log_full_mstar[good], full_sfr_ratio[good], 30)
#ax[0].contourf(counts.T, n_lvls-1, colors=gy_colors,
#                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
counts, binsx, binsy = np.histogram2d(log_full_mstar[good], full_sfr_ratio[good], 30)
ax[0].contourf(counts.T, n_lvls-1, cmap='binary',
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
b = ax[0].scatter(log_mstar, sfr_ratio, c='g', marker='.')
ax[0].set_xlabel("$\mathrm{\log_{10}(M_*)\ [M_\odot]}$")
ax[0].set_xlim(10,12)

counts, binsx, binsy = np.histogram2d(np.log10(full_total_inst_sfr[good]), full_sfr_ratio[good], 30)
#ax[1].contourf(counts.T, n_lvls-1, colors=gy_colors,
#                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[1].contourf(counts.T, n_lvls-1, cmap='binary',
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[1].scatter(np.log10(total_inst_sfr), sfr_ratio, c='g', marker='.')
ax[1].set_xlabel("log Instantaneous SFR(total)")
ax[1].set_xlim(-1.5,1.5)

counts, binsx, binsy = np.histogram2d(np.log10(full_total_inst_ssfr[good]), full_sfr_ratio[good], 30)
#ax[2].contourf(counts.T, n_lvls-1, colors=gy_colors,
#                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[2].contourf(counts.T, n_lvls-1, cmap='binary',
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[2].scatter(np.log10(total_inst_ssfr), sfr_ratio, c='g', marker='.')
ax[2].set_xlabel("log Instantaneous sSFR(total)")
ax[2].set_xlim(-12.5,-9)

ax[0].set_ylabel("Instantaneous SFR(fiber) / SFR(total)")
fig.suptitle("Ratio of Instantaneous SFR in Fiber to Total for $M_r$ Selected Subhalos")
a = ptch.Patch(color='lightgrey')
fig.legend((a,b), ('Parent w/ SFR > 0','dusty d4000 < 1.4'))
plt.savefig('Hist_Figure1.png')
plt.clf()

# In[31]:


# plt.scatter(np.log10(inner_avg_ssfr), np.log10(inner_inst_ssfr), c='g')
# x = np.arange(-11,-8)
# plt.plot(x,x,'m')
# plt.xlim(-11,-9)
# plt.ylim(-11,-9)
# plt.xlabel("log 1 Gyr Averaged sSFR")
# plt.ylabel("log Instantanous sSFR")
# plt.title("Inner sSFR for All g-r Selected Subhalos w/ d4000 < 1.4")


# # Gas Content

# In[32]:


good = np.nonzero(full_total_inst_sfr)
n_lvls = 50
#gy_colors = sns.light_palette('black', n_lvls)
#gy_colors[0] = (1,1,1)

fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(14,5))

counts, binsx, binsy = np.histogram2d(np.log10(full_mstar[good]), full_gas_ratio[good], 30)
ax[0].contourf(counts.T, n_lvls-1, cmap='binary',
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
b = ax[0].scatter(np.log10(mstar), gas_ratio, c='r', marker='.')
ax[0].set_xlabel("$\mathrm{\log_{10}(M_*)\ [M_\odot]}$")
ax[0].set_xlim(10,12)

counts, binsx, binsy = np.histogram2d(np.log10(full_total_inst_sfr[good]), full_gas_ratio[good], 30)
ax[1].contourf(counts.T, n_lvls-1, cmap='binary',
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[1].scatter(np.log10(total_inst_sfr), gas_ratio, c='r', marker='.')
ax[1].set_xlabel("log Instantaneous SFR(total)")
ax[1].set_xlim(-1.5,1.5)

counts, binsx, binsy = np.histogram2d(np.log10(full_total_inst_ssfr[good]), full_gas_ratio[good], 30)
ax[2].contourf(counts.T, n_lvls-1, cmap='binary',
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[2].scatter(np.log10(total_inst_ssfr), gas_ratio, c='r', marker='.')
ax[2].set_xlabel("log Instantaneous sSFR(total)")
ax[2].set_xlim(-12.5,-9)

ax[0].set_ylabel("Gas(fiber) / Gas(total)")
ax[0].set_ylim(0,0.3)
fig.suptitle("Ratio of Instantaneous Gas Mass in Fiber to Total for $M_r$ Selected Subhalos")
a = ptch.Patch(color='lightgrey')
fig.legend((a,b), ('Parent w/ SFR > 0','dusty d4000 < 1.4'))
# gas density cut
plt.savefig('Hist_z0_2d_sSFR_gasmass.png')
plt.clf()

# In[9]:


fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,5))
fig.tight_layout()
x = np.linspace(10,12)
y1 = x
y2 = np.log10(0.1*10**x)
y3 = np.log10(0.01*10**x)

n_lvls = 50
#gy_colors = sns.light_palette('black', n_lvls)
#gy_colors[0] = (1,1,1)

a = ptch.Patch(color='lightgrey')
counts, binsx, binsy = np.histogram2d(log_full_mstar, np.log10(full_total_gas), 30)
ax[0].contourf(counts.T, n_lvls-1, cmap='binary', extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
b = ax[0].scatter(log_mstar, np.log10(total_gas), color='r', marker='.')
c, = ax[0].plot(x,y1,'k--')
d, = ax[0].plot(x,y2,'k-.')
e, = ax[0].plot(x,y3,'k:')
ax[0].plot(x,np.log10(0.4*10**x))
ax[0].set_title("Total")
ax[0].set_xlabel("$\mathrm{M_*\ [M_\odot]}$")
ax[0].set_ylabel("$\mathrm{M_{gas}\ [M_\odot]}$")
ax[0].legend((a,b,c,d,e),['$M_r$ Parent','Dusty d4000 < 1.4','100%','10%','1%'])

good = full_inner_gas!=0
counts, binsx, binsy = np.histogram2d(log_full_mstar[good], np.log10(full_inner_gas[good]), 30)
ax[1].contourf(counts.T, n_lvls-1, cmap='binary', extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[1].scatter(log_mstar, np.log10(inner_gas), color='r', marker='.')
ax[1].plot(x,y1,'k--')
ax[1].plot(x,y2,'k-.')
ax[1].plot(x,y3,'k:')
ax[1].set_title("Inner $r<2$ kpc")
ax[1].set_xlabel("$\mathrm{M_*\ [M_\odot]}$")

ax[0].set_xlim(10,12)
ax[0].set_ylim(7, 12)

plt.savefig('Hist_z0_2d_gasinfo.png')
plt.clf()
# # SFE

# In[34]:


good = np.logical_and(full_sfe_ratio!=0, np.isfinite(full_sfe_ratio))
n_lvls = 50
#gy_colors = sns.light_palette('black', n_lvls)
#gy_colors[0] = (1,1,1)

fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(14,5))

counts, binsx, binsy = np.histogram2d(np.log10(full_mstar[good]), full_sfe_ratio[good], 30)
ax[0].contourf(counts.T, n_lvls-1, cmap='binary',extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
b = ax[0].scatter(np.log10(mstar), sfe_ratio, c='b', marker='.')
ax[0].set_xlabel("$\mathrm{\log_{10}(M_*)\ [M_\odot]}$")
ax[0].set_xlim(10,12)

counts, binsx, binsy = np.histogram2d(np.log10(full_total_inst_sfr[good]), full_sfe_ratio[good], 30)
ax[1].contourf(counts.T, n_lvls-1, cmap='binary',
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[1].scatter(np.log10(total_inst_sfr), sfe_ratio, c='b', marker='.')
ax[1].set_xlabel("log Instantaneous SFR(total)")
#ax[1].set_xlim(-1.5,1.5)

counts, binsx, binsy = np.histogram2d(np.log10(full_total_inst_ssfr[good]), full_sfe_ratio[good], 30)
ax[2].contourf(counts.T, n_lvls-1, cmap='binary',
                   extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
ax[2].scatter(np.log10(total_inst_ssfr), sfe_ratio, c='b', marker='.')
ax[2].set_xlabel("log Instantaneous sSFR(total)")
ax[2].set_xlim(-13.5,-9)

ax[0].set_ylabel("SFE(fiber) / SFE(total)")
#ax[0].set_ylim(0,0.3)
fig.suptitle("Ratio of Instantaneous SFE in Fiber to Total for $M_r$ Selected Subhalos")
a = ptch.Patch(color='lightgrey')
fig.legend((a,b), ('Parent w/ SFR > 0','dusty d4000 < 1.4'))

plt.savefig('Hist_z0_2d_SFinfo.png')
plt.clf()
# In[35]:


# bins = np.linspace(-11,-8.5,25,endpoint=True)
# #print(bins)
# good2 = np.isfinite(np.log10(sfe[:,2]))
# good3 = np.isfinite(np.log10(sfe[:,3]))

# plt.hist(np.log10(sfe[:,0]), bins=bins, histtype='step', label="$\mathrm{r<2\ kpc}$")
# plt.hist(np.log10(sfe[:,1]), bins=bins, ls="--", histtype='step', label="$\mathrm{2\ kpc < r < R_{M_{1/2}}}}$")
# plt.hist(np.log10(sfe[:,2])[good2], bins=bins, ls="-.", histtype='step', label="$\mathrm{R_{M_{1/2}} < r < 2 R_{M_{1/2}}}$")
# plt.hist(np.log10(sfe[:,3])[good3], bins=bins, ls=":", histtype='step', label="$\mathrm{r > 2 R_{M_{1/2}}}$")
# plt.legend(loc='upper left')


# In[69]:


plt.plot(np.log10(sfe).T, ls=':', color='lightgrey', marker='.', mfc='m') # lines to see where values thin
plt.plot(np.log10(np.nanmedian(sfe, axis=0)), ls='none', marker='*', ms=20, label="Median", zorder=3)
plt.plot(np.log10(np.nanpercentile(sfe, 75, axis=0)), ls='none', marker='^', ms=10, label="Upper Q")
plt.plot(np.log10(np.nanpercentile(sfe, 25, axis=0)), ls='none', marker='v', ms=10, label="Lower Q")
plt.legend(loc='lower left')
plt.ylabel("log SFE(r)")
ax=plt.gca()
ax.xaxis.set_major_locator(LinearLocator(4))
ax.xaxis.set_major_formatter(FixedFormatter(["$\mathrm{r<2\ kpc}$", "$\mathrm{2\ kpc < r < R_{M_{1/2}}}}$",
                                             "$\mathrm{R_{M_{1/2}} < r < 2 R_{M_{1/2}}}$",
                                             "$\mathrm{r > 2 R_{M_{1/2}}}$"]))
plt.title("SFE in Radial Bins for Subhalos with Dusty d4000 < 1.4")


# In[70]:


plt.plot(np.log10(full_sfe).T, ls=':', color='lightgrey', marker='.', mfc='m')
pa, = plt.plot([0.25,1.25,2.25,3.25], np.log10(np.nanmedian(full_sfe, axis=0)),
              ls='none', marker='*', ms=20, mfc='none', zorder=3)
pb, = plt.plot([0.25,1.25,2.25,3.25], np.log10(np.nanpercentile(full_sfe, 75, axis=0)),
              ls='none', marker='^', ms=10, mfc='none')
pc, = plt.plot([0.25,1.25,2.25,3.25], np.log10(np.nanpercentile(full_sfe, 25, axis=0)),
              ls='none', marker='v', ms=10, mfc='none')
a, = plt.plot(np.log10(np.nanmedian(sfe, axis=0)), ls='none', marker='*', ms=20, mfc='C0',mec='C0', zorder=3)
b, = plt.plot(np.log10(np.nanpercentile(sfe, 75, axis=0)), ls='none', marker='^', ms=10, mfc='C1',mec='C1')
c, = plt.plot(np.log10(np.nanpercentile(sfe, 25, axis=0)), ls='none', marker='v', ms=10, mfc='C2',mec='C2')

#plt.legend([(a,pa),(b,pb),(c,pc)], ['Median','Upper Q','Lower Q'], scatterpoints=1,
#               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})
plt.legend([a,b,c,pa,pb,pc],['Sample Med', 'Sample LQ', 'Sample UQ', 'Parent Med', 'Parent LQ', 'Parent UQ'], ncol=2)
plt.ylabel("log SFE(r)")
ax=plt.gca()
ax.xaxis.set_major_locator(LinearLocator(4))
ax.xaxis.set_major_formatter(FixedFormatter(["$\mathrm{r<2\ kpc}$", "$\mathrm{2\ kpc < r < R_{M_{1/2}}}}$",
                                             "$\mathrm{R_{M_{1/2}} < r < 2 R_{M_{1/2}}}$",
                                             "$\mathrm{r > 2 R_{M_{1/2}}}$"]))
plt.title("SFE in Radial Bins for $M_r$ Subhalos")
plt.savefig('Hist_z0_SFEradialbins.png')
plt.clf()

# 1e8 and 1 Mpc density as well
# 
# X radially inverted for whole M_r not just g-r
# 
# X median drop in mass over last ~few Gyr vs full M_r sample
# 
# X 4 panel plot

# In[72]:


good1 = np.logical_and(full_sfe[:,0]!=0, np.isfinite(full_sfe[:,0]))
good2 = np.logical_and(full_total_sfe!=0, np.isfinite(full_total_sfe))

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,5))
c,b,p = ax[1].hist(np.log10(1/full_sfe[:,0][good1]), bins=25, density=True, lw=2, histtype='step')
ax[1].hist(np.log10(1/sfe[:,0]), bins=b, density=True, lw=2, histtype='step')
ax[1].set_title("Inner $r<2$ kpc")
ax[1].set_xlabel("log Time [yr]")

c,b,p = ax[0].hist(np.log10(1/full_total_sfe[good2]), bins=25, density=True, histtype='step', lw=2, label="$M_r$ Parent")
ax[0].hist(np.log10(1/total_sfe), bins=b, density=True, histtype='step', lw=2, label="d4000 < 1.4")
ax[0].set_title("Total")
ax[0].set_xlabel("log Time [yr]")
ax[0].set_ylabel("Probability Density")
ax[0].legend()

fig.suptitle("Gas Depeletion Time (1/SFE)")
plt.savefig('Hist_z0_gasdeptime.png')

# In[53]:


#sns.palplot(sns.color_palette(['mediumpurple','steelblue','yellowgreen']))


# In[40]:


sns.__version__

