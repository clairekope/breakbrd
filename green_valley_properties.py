import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import os
import gc
import h5py
import readsubfHDF5
import readhaloHDF5
import snapHDF5
import numpy as np
import astropy.units as u
from astropy.constants import m_p, k_B, G
from scipy.stats import binned_statistic_2d
# prep MPI environnment and import scatter_work(), get(), periodic_centering(),
# CLI args container, url_dset, url_sbhalos, folder, snapnum, littleh, omegaL/M
from utilities import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

do_profiles = False

nbins = 100
r_edges = np.logspace(-1, 0, nbins+1)
binned_r = r_edges[:-1] + np.diff(r_edges)


if rank==0:
    # Get the halos to loop over. It is now "all" of them.
    min_mass = littleh # 1e10 Msun in 1/1e10 Msun / h
    max_mass = 1000 * littleh # 1e13 Msun
    search_query = "?mass_dm__gt=" + str(min_mass) \
                 + "&mass_dm__lt=" + str(max_mass)
                 
    cut1 = get(url_sbhalos + search_query)
    cut1 = get(url_sbhalos + search_query, {'limit':cut1['count'], 'order_by':'id'})

    sub_ids = np.array([sub['id'] for sub in cut1['results']], dtype='i')

    if args.local:
        cat = readsubfHDF5.subfind_catalog(args.local, snapnum, #grpcat=False, subcat=False,
                                           keysel=['GroupFirstSub','SubhaloGrNr'])
        sat = np.zeros(cat.SubhaloGrNr.size, dtype=bool)
        sat[sub_ids] = (sub_ids != cat.GroupFirstSub[cat.SubhaloGrNr[sub_ids]])

        del cat
        gc.collect()

        # sat is bigger than sub_ids
        sub_ids = sub_ids[~sat[sub_ids]]

    print("List acquired")

else:
    sub_ids = None
    if args.local:
        sat = None
                                   
my_subs = scatter_work(sub_ids, rank, size)
#sub_ids = comm.bcast(sub_ids, root=0)
if args.local:
    sat = comm.bcast(sat, root=0)

boxsize = get(url_dset)['boxsize']
z = args.z
a0 = 1/(1+z)

H0 = littleh * 100 * u.km/u.s/u.Mpc

good_ids = np.where(my_subs > -1)[0]
my_profiles = {}

for sub_id in my_subs[good_ids]:

    print(sub_id, flush=True)

    my_profiles[sub_id] = {}

    #
    # Pull API quantities
    #
    
    sub = get(url_sbhalos + str(sub_id))
    dm_halo = sub["mass_dm"] * 1e10 / littleh * u.Msun
    star_mass = sub["mass_stars"] * 1e10 / littleh * u.Msun
    sfr = sub["sfr"] * u.Msun / u.yr
    r_half = sub["halfmassrad_stars"] * u.kpc * a0/littleh
    r_half_gas = sub["halfmassrad_gas"] * u.kpc * a0/littleh
    x0 = sub["pos_x"]
    y0 = sub["pos_y"]
    z0 = sub["pos_z"]
    
    my_profiles[sub_id]['dm_mass'] = dm_halo
    my_profiles[sub_id]['star_mass'] = star_mass
    my_profiles[sub_id]['ssfr'] = sfr/star_mass
    my_profiles[sub_id]['sat'] = sat[sub_id]

    #
    # Load particle data
    #
    
    gas = True
    stars = True
    if not args.local:
        raise NotImplementedError("Cutouts not updated to include required info")
    
    else: # use local dataset instead of api-downloaded cutouts
        readhaloHDF5.reset()

        # dm_coords = readhaloHDF5.readhalo(args.local, "snap", snapnum,
        #                                   "POS ", 1, -1, sub_id, long_ids=True,
        #                                   double_output=False).astype("float32")
        # dm_mass = readhaloHDF5.readhalo(args.local, "snap", snapnum,
        #                                 "MASS", 1, -1, sub_id, long_ids=True,
        #                                 double_output=False).astype("float32")
        
        try:
            # Gas
            coords = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                           "POS ", 0, -1, sub_id, long_ids=True,
                                           double_output=False).astype("float32")
            
            vel = readhaloHDF5.readhalo(args.local, "snap", snapnum,
                                        "VEL ", 0, -1, sub_id, long_ids=True,
                                        double_output=False).astype("float32")
            
            dens = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                         "RHO ", 0, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")
            
            mass = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                         "MASS", 0, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")
            
            inte = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                         "U   ", 0, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")
            
            elec = readhaloHDF5.readhalo(args.local, "snap", snapnum,
                                         "NE  ", 0, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")

            # cool_rate = readhaloHDF5.readhalo(args.local, "snap", snapnum,
            #                                   "GCOL", 0, -1, sub_id, long_ids=True,
            #                                   double_output=False).astype("float32")
            
        except AttributeError:
            gas = False


        # Stars
        try:
            scoords = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                            "POS ", 4, -1, sub_id, long_ids=True,
                                            double_output=False).astype("float32")
            
            svel = readhaloHDF5.readhalo(args.local, "snap", snapnum,
                                         "VEL ", 4, -1, sub_id, long_ids=True,
                                         double_output=False).astype("float32")
            
            smass = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                          "MASS", 4, -1, sub_id, long_ids=True,
                                          double_output=False).astype("float32")
            
            a_form = readhaloHDF5.readhalo(args.local, "snap", snapnum, 
                                           "GAGE", 4, -1, sub_id, long_ids=True,
                                           double_output=False).astype("float32")

            # filter out wind particles
            star_filter = a_form > 0
            scoords = scoords[star_filter]
            svel = svel[star_filter]
            smass = smass[star_filter]
            
        except AttributeError:
            stars = False

    #
    # Calculate r200 and other virial quantities
    #

    r200 = (G*dm_halo/(100*H0**2))**(1/3)
    r200 = r200.to('kpc')

    disp_200 = G*dm_halo/(2*r200)
    disp_200 = np.sqrt(disp_200).to('km/s')
        
    T200 = 0.6*m_p/(2*k_B) * disp_200**2
    T200 = T200.to('K')

    # save virial quantities
    my_profiles[sub_id]['disp_200'] = disp_200
    my_profiles[sub_id]['T_200'] = T200
  
    #
    # Calculate gas information
    #
    
    if gas:
        
        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        x_rel = periodic_centering(x, x0, boxsize) * u.kpc * a0/littleh
        y_rel = periodic_centering(y, y0, boxsize) * u.kpc * a0/littleh
        z_rel = periodic_centering(z, z0, boxsize) * u.kpc * a0/littleh
        r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        r_vec = np.column_stack([x_rel,y_rel,z_rel]).value * u.kpc # unit borks
        
        mass = mass * 1e10 / littleh * u.Msun
        vel = vel * np.sqrt(a0) * u.km/u.s

        # subtracting off bulk velocity
        vel[:,0] -= sub['vel_x'] * u.km/u.s
        vel[:,1] -= sub['vel_y'] * u.km/u.s
        vel[:,2] -= sub['vel_z'] * u.km/u.s

        #
        # Calculate Entropy
        #

        # For conversion of internal energy to temperature, see
        # https://www.tng-project.org/data/docs/faq/#gen4
        inte *= u.erg/u.g
        X_H = 0.76
        gamma = 5./3.
        mu = 4/(1 + 3*X_H + 4*X_H*elec)
        temp = ( (gamma-1) * inte/k_B * mu*m_p * 1e10 ).to('K')

        dens = dens * 1e10*u.Msun/littleh * (u.kpc*a0/littleh)**-3
        ne = elec * X_H*dens/m_p # elec frac defined as n_e/n_H
        ent = k_B * temp/ne**(gamma-1)
        ent = ent.to('eV cm^2', equivalencies=u.temperature_energy())

        pres = dens/m_p * k_B * temp

        cool_rate *= u.erg * u.cm**3 / u.s
        intE = (inte*mu*m_p).to('erg')
        n = (dens/(mu*m_p)).to('cm**-3')
        tcool = intE/(cool_rate*n)
        assert tcool.unit == u.s

        #
        # Calculate some CGM gas properties
        #

        I will probably change how I distinguish the CGM
        so that lower radii can be probed
        r_CGM = 2*r_half # DeFelippis+20, Zinger+20
        CGM = r > r_CGM
        M_gas_CGM = np.sum(mass[CGM])

        # Temperatures
        if ( temp[CGM] > 1e5*u.K ).any():
            hot_CGM = np.logical_and( temp > 1e5*u.K, CGM )
            mass_CGM_hot  = np.sum(mass[hot_CGM])
            T_hot_avg = np.average(temp[hot_CGM], weights = mass[hot_CGM])
        else:
            mass_CGM_hot  = np.nan * u.Msun
            T_hot_avg = np.nan * u.K

        if ( temp[CGM] < 1e5*u.K ).any():
            cool_CGM = np.logical_and( temp < 1e5*u.K, CGM )
            mass_CGM_cool = np.sum(mass[cool_CGM])
        else:
            mass_CGM_cool = np.nan * u.Msun

        # Gas angular momentum
        # vel subtracts mass-weighted pec vel; same as COM vel?
        j_vec_gas = mass[:,np.newaxis]*np.cross(r_vec,vel)
        j_vec_gas = j_vec_gas.value * u.Msun * u.kpc * u.km/u.s # units got lost again
        j_vec_gas_CGM = np.sum(j_vec_gas[CGM], axis=0)/M_gas_CGM
        j_gas_CGM = np.sqrt(np.sum(j_vec_gas_CGM**2)).to('kpc*km/s')
   
        # Mass-weighted avg of all entropy (excluding SF gas)
        # Zinger et al 2020
        dthresh = 0.1 * u.cm**-3 * m_p # cannonical star formation density threshold

        dense_disk = np.logical_and(dens > dthresh, r < 2*r_half)
        if dense_disk.sum() > 0:
            disk_avg_ent = np.average(ent[dense_disk], weights=mass[dense_disk])
        else:
            disk_avg_ent = np.nan * u.eV*u.cm**2

        inner_CGM = np.logical_and(r > 2*r_half, r < r_half_gas)
        dense_inner = np.logical_and(dens > dthresh, inner_CGM)
        if dense_inner.sum() > 0:
            inner_avg_ent = np.average(ent[dense_inner], weights=mass[dense_inner])
        else:
            inner_avg_ent = np.nan * u.eV*u.cm**2

        my_profiles[sub_id]['mass_CGM'] = M_gas_CGM
        my_profiles[sub_id]['mass_CGM_hot']  = mass_CGM_hot
        my_profiles[sub_id]['mass_CGM_cool'] = mass_CGM_cool
        my_profiles[sub_id]['T_hot_avg'] = T_hot_avg
        my_profiles[sub_id]['j_gas_CGM'] = j_gas_CGM
        my_profiles[sub_id]['disk_ent_avg'] = disk_avg_ent
        my_profiles[sub_id]['inner_ent_avg'] = inner_ent_avg

        #
        # Calculate & store radial profiles
        #

        if do_profiles:
                        
            # bin K in scaled radial bins
            r_scale = (r/r200).value
            rbinner = np.digitize(r_scale, r_edges)
            
            binned_ent_avg = np.ones_like(binned_r)*np.nan * u.eV*u.cm**2
            binned_ent_med = np.ones_like(binned_r)*np.nan * u.eV*u.cm**2
            
            binned_pres_avg = np.ones_like(binned_r)*np.nan * u.dyn/u.cm**2
            binned_pres_med = np.ones_like(binned_r)*np.nan * u.dyn/u.cm**2
            
            binned_tcool_avg = np.ones_like(binned_r)*np.nan * u.s
            binned_tcool_med = np.ones_like(binned_r)*np.nan * u.s
            
            binned_mass_cool = np.ones_like(binned_r)*np.nan * u.Msun
            binned_mass_hot  = np.ones_like(binned_r)*np.nan * u.Msun
            
            binned_mass_loK = np.zeros_like(binned_r) * u.Msun
            binned_mass_hiK = np.zeros_like(binned_r) * u.Msun
            
            #find central tendency for each radial bin
            for i in range(1, r_edges.size):
                this_bin = rbinner==i
                if np.sum(mass[this_bin]) != 0: # are there particles in this bin

                    binned_ent_avg[i-1] = np.average(ent[this_bin],
                                                 weights = mass[this_bin])
                    binned_ent_med[i-1] = np.median(ent[this_bin])

                    binned_pres_avg[i-1] = np.average(pres[this_bin],
                                                  weights = mass[this_bin])
                    binned_pres_med[i-1] = np.median(pres[this_bin])

                    binned_tcool_avg[i-1] = np.average(tcool[this_bin],
                                                   weights = mass[this_bin])
                    binned_tcool_med[i-1] = np.median(tcool[this_bin])

                    this_temp = temp[this_bin]
                    binned_mass_cool[i-1] = np.sum(mass[this_bin][this_temp < 1e5*u.K])
                    binned_mass_hot[i-1]  = np.sum(mass[this_bin][this_temp > 1e5*u.K])

                    this_ent = ent[this_bin]
                    binned_mass_loK[i-1] = np.sum(mass[this_bin][this_ent < 1e3*u.eV*u.cm**2])
                    binned_mass_hiK[i-1] = np.sum(mass[this_bin][this_ent > 1e3*u.eV*u.cm**2])
                
            my_profiles[sub_id]['ent_avg'] = binned_ent_avg
            my_profiles[sub_id]['ent_med'] = binned_ent_med
            my_profiles[sub_id]['pres_avg'] = binned_pres_avg
            my_profiles[sub_id]['pres_med'] = binned_pres_med
            my_profiles[sub_id]['tcool_avg'] = binned_tcool_avg
            my_profiles[sub_id]['tcool_med'] = binned_tcool_med
            my_profiles[sub_id]['mass_cool'] = binned_mass_cool
            my_profiles[sub_id]['mass_hot']  = binned_mass_hot
            my_profiles[sub_id]['mass_loK'] = binned_mass_loK
            my_profiles[sub_id]['mass_hiK'] = binned_mass_hiK

    else: # no gas
        my_profiles[sub_id]['mass_CGM'] = np.nan * u.Msun
        my_profiles[sub_id]['mass_CGM_hot']  = np.nan * u.Msun
        my_profiles[sub_id]['mass_CGM_cool'] = np.nan * u.Msun
        my_profiles[sub_id]['T_hot_avg'] = np.nan * u.K
        my_profiles[sub_id]['j_gas_CGM'] = np.nan * u.km/u.s * u.kpc
        my_profiles[sub_id]['disk_ent_avg'] = np.nan * u.eV*u.cm**2
        my_profiles[sub_id]['inner_ent_avg'] = np.nan * u.eV*u.cm**2

        if do_profiles:
            my_profiles[sub_id]['ent_avg'] = np.nan
            my_profiles[sub_id]['ent_med'] = np.nan
            my_profiles[sub_id]['pres_avg'] = np.nan
            my_profiles[sub_id]['pres_med'] = np.nan
            my_profiles[sub_id]['tcool_avg'] = np.nan
            my_profiles[sub_id]['tcool_med'] = np.nan
            my_profiles[sub_id]['mass_cool'] = 0.0
            my_profiles[sub_id]['mass_hot']  = 0.0
            my_profiles[sub_id]['mass_loK'] = 0.0
            my_profiles[sub_id]['mass_hiK'] = 0.0
        
    #
    # Calculate stellar information
    #
    
    if stars:

        sx = scoords[:,0]
        sy = scoords[:,1]
        sz = scoords[:,2]
        sx_rel = periodic_centering(sx, x0, boxsize) * u.kpc * a0/littleh
        sy_rel = periodic_centering(sy, y0, boxsize) * u.kpc * a0/littleh
        sz_rel = periodic_centering(sz, z0, boxsize) * u.kpc * a0/littleh
        sr = np.sqrt(sx_rel**2 + sy_rel**2 + sz_rel**2)    

        smass = smass * 1e10 / littleh * u.Msun
        svel = svel * np.sqrt(a0) * u.km/u.s

        # subtracting off bulk velocity
        svel[:,0] -= sub['vel_x'] * u.km/u.s
        svel[:,1] -= sub['vel_y'] * u.km/u.s
        svel[:,2] -= sub['vel_z'] * u.km/u.s

        disp_star = np.sqrt(np.sum(np.var(svel[sr < r_half], axis=0)))

        my_profiles[sub_id]['disp_star'] = disp_star
        
    else: # no stars
        my_profiles[sub_id]['disp_star'] = np.nan * u.km/u.s


# Collect data from MPI ranks & write to files


profile_list = comm.gather(my_profiles, root=0)

if rank==0:

    all_galprop = np.zeros( (len(sub_ids), 15) )
    if do_profiles:
        all_entprof = np.zeros( (len(sub_ids), 2*nbins+1) )
        all_presprof = np.zeros( (len(sub_ids), 2*nbins+1) )
        all_tcoolprof = np.zeros( (len(sub_ids), 2*nbins+1) )
        all_massprof = np.zeros( (len(sub_ids), 4*nbins+1) )
    
    i=0
    for dic in profile_list:
        for k,v in dic.items():
            all_galprop[i,0] = k
            all_galprop[i,1] = v['sat']
            all_galprop[i,2] = v['dm_mass'].value
            all_galprop[i,3] = v['star_mass'].value
            all_galprop[i,4] = v['mass_CGM'].value
            all_galprop[i,5] = v['mass_CGM_hot'].value
            all_galprop[i,6] = v['mass_CGM_cool'].value
            all_galprop[i,7] = v['disp_200'].value
            all_galprop[i,8] = v['disp_star'].value
            all_galprop[i,9] = v['ssfr'].value
            all_galprop[i,10] = v['j_gas_CGM'].value
            all_galprop[i,11] = v['T_200'].value
            all_galprop[i,12] = v['T_hot_avg'].value
            all_galprop[i,13] = v['disk_ent_avg'].value
            all_galprop[i,14] = v['inner_ent_avg'].value

            if do_profiles:
                all_entprof[i,0] = k
                all_entprof[i,1::2] = v['ent_avg']
                all_entprof[i,2::2] = v['ent_med']
                
                all_presprof[i,0] = k
                all_presprof[i,1::2] = v['pres_avg']
                all_presprof[i,2::2] = v['pres_med']
                
                all_tcoolprof[i,0] = k
                all_tcoolprof[i,1::2] = v['tcool_avg']
                all_tcoolprof[i,2::2] = v['tcool_med']
                
                all_massprof[i,0] = k
                all_massprof[i,1::4] = v['mass_cool']
                all_massprof[i,2::4] = v['mass_hot']
                all_massprof[i,3::4] = v['mass_loK']
                all_massprof[i,4::4] = v['mass_hiK']
            
            i+=1

    sort = np.argsort(all_galprop[:,0])

    prop_header = "SubID,Sat,MassDark,MassStar,MassCGM,MassCGMHot,MassCGMCool,Disp200,DispStar,sSFR,jGas,T200,THot,AvgEnt"

    header = "SubID"
    for r in binned_r:
        header += "   {:.4f} avg med".format(r)

    mass_header = "SubID"
    for r in binned_r:
        header += "   {:.4f} cool hot lowK highK".format(r)

    np.savetxt(folder+'GV_halo_properties_mini.csv', all_galprop[sort],
               delimiter=',', header=prop_header)
    if do_profiles:
        np.savetxt(folder+'GV_entropy_profiles.csv', all_entprof[sort], 
                   delimiter=',', header=header)
        np.savetxt(folder+'GV_pressure_profiles.csv', all_presprof[sort],
                   delimiter=',', header=header)
        np.savetxt(folder+'GV_tcool_profiles.csv', all_tcoolprof[sort],
                   delimiter=',', header=header)
        np.savetxt(folder+'GV_mass_profiles.csv', all_massprof[sort],
                   delimiter=',', header=mass_header)
