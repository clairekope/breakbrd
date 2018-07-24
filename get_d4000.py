# coding: utf-8
import pickle
import numpy as np
from scipy.interpolate import interp1d
import glob

dust = False

def get_dn4000(wave,spec):
    interp_spec = interp1d(wave,spec)
    blue_wav = np.linspace(3850,3950,100)
    red_wav = np.linspace(4000,4100,100)
    d4000 = np.sum(interp_spec(red_wav)) / np.sum(interp_spec(blue_wav))

    return d4000

files = glob.glob("spectra/{}dust/spectra_*.txt".format("" if dust else "no_"))
d4000 = {}

for f in files:

    sub_id = int(f[-10:-4])
    dat = np.genfromtxt(f)

    wave = dat[:int(dat.size/2)]
    spec = dat[int(dat.size/2):]

    d4000[sub_id] = get_dn4000(wave, spec)

with open("d4000_{}dust.pkl".format("" if dust else "no_"), "wb") as pkl:
    pickle.dump(d4000, pkl)