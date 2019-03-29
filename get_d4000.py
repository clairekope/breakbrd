import pickle
import numpy as np
from scipy.interpolate import interp1d
import glob
from utilities import folder, args 

inst = args.inst_sfr
dust = args.dusty

def get_dn4000(wave,spec):
    interp_spec = interp1d(wave,spec)
    blue_wav = np.linspace(3850,3950,100)
    red_wav = np.linspace(4000,4100,100)
    d4000 = np.sum(interp_spec(red_wav)) / np.sum(interp_spec(blue_wav))

    return d4000

files = glob.glob(folder+"spectra/{}inst/{}dust/inner/spectra_*.txt".format(
                             "" if inst else "no_",
                             "" if dust else "no_"))
d4000 = np.zeros( (len(files),2) )

for i, f in enumerate(files):

    sub_id = int(f[-10:-4])
    dat = np.genfromtxt(f)

    wave = dat[0,:]
    spec = dat[1,:]

    d4000[i] = sub_id, get_dn4000(wave, spec)

sort = np.argsort(d4000[:,0])

np.savetxt(folder+"D4000_{}{}dust.csv".format(
                             "" if inst else "no_inst_",
                             "" if dust else "no_"),
           d4000[sort], fmt="%d %g", header='Sub ID, D4000', delimiter=',')

