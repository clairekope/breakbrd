
# coding: utf-8

import pickle
import requests
import h5py
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from astropy import units as u

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

def periodic_centering(x, center, boxsixe):
    # stack two periodic boxes next to each other
    xx = np.concatenate((x, boxsize+x))
    # if the center is on the left side of the box,
    # move it over one boxlength to put it near the center of xx
    if center < boxsize/2:
        center +=  boxsize
    crit = np.logical_and(xx >= center-boxsize/2,
                          xx <= center+boxsize/2)
    #try:
    assert x.size == xx[crit].size
    #except AssertionError:
    #    pdb.set_trace()
    return xx[crit] - center

# with open("cut_sample.pkl","rb") as f:
#     sample = pickle.load(f)

# with open("cut_radii.pkl","rb") as f:
#     radii = pickle.load(f)

# cut5 = {}
# for k in sample.keys():
#     if k in radii:
#         cut5[k] = sample[k]

# with open("cut5.pkl","wb") as f:
#     pickle.dump(cut5,f)

with open("cut5.pkl","rb") as f:
    sample = pickle.load(f)

# with h5py.File('nonparametric_morphologies.hdf5') as f:
#     cam0_id = f['Snapshot_135']['SubfindID_cam0'][:]
#     cam1_id = f['Snapshot_135']['SubfindID_cam1'][:]
#     cam2_id = f['Snapshot_135']['SubfindID_cam2'][:]
#     cam3_id = f['Snapshot_135']['SubfindID_cam3'][:]

#     cam0 = f['Snapshot_135']['gSDSS']['RE_cam0'][:]
#     cam1 = f['Snapshot_135']['gSDSS']['RE_cam1'][:]
#     cam2 = f['Snapshot_135']['gSDSS']['RE_cam2'][:]
#     cam3 = f['Snapshot_135']['gSDSS']['RE_cam3'][:]

sub_ids = [k for k in sample.keys()]
mstar = np.array([sample[k]['stellar_mass'] for k in sub_ids]) # ensure same order
mgas = np.empty_like(mstar)
ssfr = np.array([sample[k]['sSFR_1Gyr'].value for k in sub_ids])
# half_mass = np.array([sample[k]['half_mass_rad'] for k in sub_ids])
# half_light = np.empty((4, half_mass.size))

url_base = "http://www.illustris-project.org/api/Illustris-1/snapshots/135/subhalos/"
for i, sub_id in enumerate(sub_ids):
    sub = get(url_base+str(sub_id))
    mgas[i] = sub['mass_gas']*1e10/0.704
    
#     half_light[0,i] = cam0[np.argwhere(cam0_id==sub_id)].flatten() if (cam0_id==sub_id).any() else np.nan
#     half_light[1,i] = cam1[np.argwhere(cam1_id==sub_id)].flatten() if (cam1_id==sub_id).any() else np.nan
#     half_light[2,i] = cam2[np.argwhere(cam2_id==sub_id)].flatten() if (cam2_id==sub_id).any() else np.nan
#     half_light[3,i] = cam3[np.argwhere(cam3_id==sub_id)].flatten() if (cam3_id==sub_id).any() else np.nan

# half_light_mean = np.nanmean(half_light, axis=0)

log_mstar = np.log10(mstar)
log_mgas = np.where(mgas!=0, np.log10(mgas), np.nan)

plt.hist([log_mstar, log_mgas], 20, range=(6.2,11.91), histtype='step', label=["stars",'gas'])
plt.xlabel("$\mathrm{\log_{10}\ M\ [M_\odot]}$")
#plt.title("Subhalos with $\mathrm{\log_{10}(sSFR) > -11\ yr^{-1}}$")
plt.title("Growth-Inverted Subhalos with $\mathrm{\log_{10}\ sSFR > -11\ yr^{-1}}$")
plt.legend(loc="upper left")
plt.savefig("star_+_gas_mass_dist.png"); plt.clf()

plt.scatter(log_mstar, log_mgas, marker='.')
plt.xlabel("$\mathrm{\log_{10}\ M_*\ [M_\odot]}$")
plt.ylabel("$\mathrm{\log_{10}\ M_{gas}\ [M_\odot]}$")
#plt.title("Subhalos with $\mathrm{\log_{10}(sSFR) > -11\ yr^{-1}}$")
plt.title("Growth-Inverted Subhalos with $\mathrm{\log_{10}\ sSFR > -11\ yr^{-1}}$")
plt.savefig("star_v_gas_mass.png"); plt.clf()

plt.hist(np.log10(ssfr),15)
plt.xlabel("$\mathrm{\log_{10}\ sSFR(1\ Gyr)\ [M_\odot]}$")
plt.title("Growth-Inverted Subhalos with $\mathrm{\log_{10}\ sSFR > -11\ yr^{-1}}$")
plt.savefig("ssfr_dist.png"); plt.clf()

# plt.scatter(log_mstar, half_mass, marker='.', label='Half Mass')
# plt.scatter(log_mstar, half_light_mean, marker='.', label='<Half Light>')
# plt.ylabel("Radius (kpc)")
# plt.xlabel("$\mathrm{\log_{10}(M_*)\ (M_\odot)}$")
# plt.legend(loc="upper left")

boxsize = 75000
H0 = 0.704 * 100
omegaM = 0.2726
omegaL = 0.7274
timenow = 2.0/(3.0*H0) * 1./np.sqrt(omegaL) \
    * np.log(np.sqrt(omegaL*1./omegaM) \
    + np.sqrt(omegaL*1./omegaM+1)) \
    * 3.08568e19/3.15576e16 * u.Gyr

# plt.margins(tight=True)
# fig, ax = plt.subplots(2,3,sharey=True)
# ax = ax.flatten()

# subset = np.random.choice(sub_ids, 6)
# for i, s in enumerate(subset):
#     sub = get(url_base+str(s))
#     r_half = sample[s]['half_mass_rad']
#     file = "stellar_cutouts/cutout_{}.hdf5".format(s)
#     with h5py.File(file) as f:
#         coords = f['PartType4']['Coordinates'][:,:]
#         a = f['PartType4']['GFM_StellarFormationTime'][:] # as scale factor
#         init_mass = f['PartType4']['GFM_InitialMass'][:]

#     stars = [a > 0] # throw out wind particles (a < 0)
#     x = coords[:,0][stars]
#     y = coords[:,1][stars]
#     z = coords[:,2][stars]
#     x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc / 0.704
#     y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc / 0.704
#     z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc / 0.704
#     r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    
#     init_mass = init_mass[stars] * 1e10 / 0.704 * u.Msun
#     a = a[stars]

#     form_time = 2.0/(3.0*H0) * 1./np.sqrt(omegaL) \
#                 * np.log(np.sqrt(omegaL*1./omegaM*(a)**3) \
#                 + np.sqrt(omegaL*1./omegaM*(a)**3+1)) \
#                 * 3.08568e19/3.15576e16  \
#                 * u.Gyr
#     age = timenow-form_time

#     bins = [0, 2, 1*r_half, 2*r_half] * u.kpc
#     binner = np.digitize(r, bins) # index len(bins) is overflow
    
#     for r_bin in range(1, bins.size+1):
#         form_history = np.sort(form_time[binner==r_bin])
#         sort = np.argsort(form_time[binner==r_bin])
#         age_progrssn = age[binner==r_bin][sort]       # sort ages by formation time; oldest first
#         mass_history = init_mass[binner==r_bin][sort] # sort initial mass by formation time; 
#                                                       #     early mass first
#         mass_frac = np.cumsum(mass_history)/np.sum(mass_history)
    
#         assert np.all(mass_frac[1:] >= mass_frac[:-1])       # monotonically increasing
#         assert np.all(age_progrssn[1:] <= age_progrssn[:-1]) # monotonically decreasing

#         if r_bin==1:
#             ax[i].plot(age_progrssn.value, mass_frac, c='pink', # X11 colors
#                      label="$<2\mathrm{\ kpc}$")
#         elif r_bin==2:
#             ax[i].plot(age_progrssn.value, mass_frac, c='plum',
#                      label="$2\mathrm{\ kpc} - 1R_{M_{1/2}}$")
#         elif r_bin==3:
#             ax[i].plot(age_progrssn.value, mass_frac, c='orchid',
#                      label="$1R_{M_{1/2}} - 2R_{M_{1/2}}$")
#         elif r_bin==4:
#             ax[i].plot(age_progrssn.value, mass_frac, c='purple',
#                      label="$>2R_{M_{1/2}}$")
            
#     #lim = (0, np.max(np.log10(age.value)))
#     lim = ax[i].set_xlim((0,15))
#     ax[i].set_ylim(0,1)
#     ax[i].hlines([0.5,0.8], *lim, linestyle=":", color="gray")
# ax[3].set_xlabel("Age (Gyr)")
# ax[3].set_ylabel("Stellar Mass Fraction")
# ax[3].legend(loc="lower left", fontsize='xx-small')
# fig.savefig("random_sample.png"); plt.clf()
# print(subset)


nbins = 50
bin_masses = np.empty((nbins, log_mstar.size))
bin_masses2 = np.empty((4, log_mstar.size))
inner_mass = np.empty_like(log_mstar)
w_inst_ssfr = np.empty_like(log_mstar)
w_avg_ssfr = np.empty_like(log_mstar)
h_inst_ssfr = np.empty_like(log_mstar)
h_avg_ssfr = np.empty_like(log_mstar)

time_bins = np.arange(0,14.01,0.01) # 0 to 14 Gyr in 10 Myr bins
dt = time_bins[1:] - time_bins[:-1] # if we change to unequal bins this supports that

for i, s in enumerate(sub_ids):
    sub = get(url_base+str(s))
    w_inst_ssfr[i] = sub["sfr"]/mstar[i] # 1/yr
    h_inst_ssfr[i] = sub["sfrinhalfrad"]/mstar[i] # 1/yr
    
    r_half = sample[s]['half_mass_rad']
    file = "stellar_cutouts/cutout_{}.hdf5".format(s)
    with h5py.File(file) as f:
        coords = f['PartType4']['Coordinates'][:,:]
        a = f['PartType4']['GFM_StellarFormationTime'][:] # as scale factor
        init_mass = f['PartType4']['GFM_InitialMass'][:]
        curr_mass = f['PartType4']['Masses'][:]

    stars = [a > 0] # throw out wind particles (a < 0)
    x = coords[:,0][stars]
    y = coords[:,1][stars]
    z = coords[:,2][stars]
    x_rel = periodic_centering(x, sub['pos_x'], boxsize) * u.kpc / 0.704
    y_rel = periodic_centering(y, sub['pos_y'], boxsize) * u.kpc / 0.704
    z_rel = periodic_centering(z, sub['pos_z'], boxsize) * u.kpc / 0.704
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    
    init_mass = init_mass[stars] * 1e10 / 0.704 * u.Msun
    curr_mass = curr_mass[stars] * 1e10 / 0.704 * u.Msun
    a = a[stars]
    
    form_time = 2.0/(3.0*H0) * 1./np.sqrt(omegaL) \
                * np.log(np.sqrt(omegaL*1./omegaM*(a)**3) \
                + np.sqrt(omegaL*1./omegaM*(a)**3+1)) \
                * 3.08568e19/3.15576e16  \
                * u.Gyr
    
    bins = np.linspace(0, 2, nbins, endpoint=True) # r_half
    binner = np.digitize(r/r_half, bins) # index len(bins) is overflow   
    
    for r_bin in range(0, bins.size):
        bin_masses[r_bin, i] = (np.sum(curr_mass[binner==r_bin])/mstar[i]).value

    bins2 = np.array([0,2/r_half,1,2]) # std bins in r_half
    binner2 = np.digitize(r/r_half, bins2)
    for r_bin in range(1, bins2.size+1):
        if r_bin==1:
            inner_mass[i] = np.sum(curr_mass[binner2==r_bin]).value            
        bin_masses2[r_bin-1, i] = (np.sum(curr_mass[binner2==r_bin])/mstar[i]).value
        

    # sSFR(1 Gyr) inside 1 half mass radius
    inner = r/(r_half*u.kpc) < 1
    form_history = np.sort(form_time[inner])
    sort = np.argsort(form_time[inner])
    mass_history = init_mass[inner][sort]

    t_binner = np.digitize(form_history, bins=time_bins)
    sfr = np.array([ mass_history.value[t_binner==j].sum()/dt[j] for j in range(1,dt.size) ])
    sfr *= u.Msun/u.Gyr; sfr = sfr.to(u.Msun/u.yr) # divide by 1e9
    #print(np.average(sfr[-101:])/np.sum(curr_mass[inner]))
    h_avg_ssfr[i] = (np.average(sfr[-101:])/np.sum(curr_mass[inner])).value
    
    # sSFR(1 Gyr) for whole subhalo
    form_history = np.sort(form_time)
    sort = np.argsort(form_time)
    mass_history = init_mass[sort]

    t_binner = np.digitize(form_history, bins=time_bins)
    sfr = np.array([ mass_history.value[t_binner==j].sum()/dt[j] for j in range(1,dt.size) ])
    sfr *= u.Msun/u.Gyr; sfr = sfr.to(u.Msun/u.yr) # divide by 1e9

    w_avg_ssfr[i] = (np.average(sfr[-101:])/np.sum(curr_mass)).value


plt.scatter(log_mstar, np.log10(inner_mass))
plt.xlabel("$\mathrm{\log_{10}\ M_*\ [M_\odot]}$")
plt.ylabel("$\mathrm{\log_{10}\ M_*(r<2\ kpc)\ [M_\odot]}$")
plt.title("Growth-Inverted Subhalos with $\mathrm{\log_{10}\ sSFR > -11\ yr^{-1}}$")
plt.savefig("ssfr_mstar_scatter.png"); plt.clf()

plt.scatter(log_mstar, inner_mass/mstar)
plt.xlabel("$\mathrm{\log_{10}\ M_*\ [M_\odot]}$")
plt.ylabel("Fraction of $M_*$ with $r<2$ kpc")
plt.title("Growth-Inverted Subhalos with $\mathrm{\log_{10}\ sSFR > -11\ yr^{-1}}$")
plt.savefig("ssfr_mstar_ratio.png"); plt.clf()


avg = np.average(bin_masses, axis=1)
wavg = np.average(bin_masses, axis=1, weights=mstar/mstar.sum())

bavg = np.average(bin_masses2, axis=1)
width = np.empty(4)
width[:-1] = bins2[1:]- bins2[:-1]
width[-1] = .5

plt.plot(bins, avg, label="unweighted", c='C2')
#plt.plot(bins, wavg, label='weighted')
plt.bar(bins2[:-1], bavg[:-1], width=width[:-1], align='edge',
        fill=False, zorder=-1)
plt.bar(bins2[-1], bavg[-1], width=width[-1], align="edge",
        fill=False, zorder=-1, hatch='/')
plt.xlabel("$r/R_{M_{1/2}}$")
plt.ylabel("Current Stellar Mass Fraction")
plt.title("Growth-Inverted Subhalos with $\mathrm{\log_{10}\ sSFR > -11\ yr\^{-1}}$")
plt.legend()
plt.savefig("stellar_mass_frac.png"); plt.clf()

plt.scatter(np.log10(w_avg_ssfr), np.log10(w_inst_ssfr), marker='.', label="Whole Subhalo")
plt.scatter(np.log10(h_avg_ssfr), np.log10(h_inst_ssfr), marker='.', label="$r < R_{M_{1/2}}$")
x=np.linspace(-12,-10)
plt.plot(x,x, c='m', label='1:1 eye guide')
plt.legend()
plt.xlabel("log(sSFR) over 1 Gyr")
plt.ylabel("Instantaneous log(sSFR)")
plt.savefig("inst_v_avg_ssfr.png"); plt.clf()
