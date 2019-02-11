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

files = glob.glob(folder+"spectra/{}inst/{}dust/spectra_*.txt".format(
                            "" if inst else "no_", "" if dust else "no_"))
d4000 = {}

for f in files:

    sub_id = int(f[-10:-4])
    dat = np.genfromtxt(f)

    wave = dat[0,:]
    spec = dat[1,:]

    d4000[sub_id] = get_dn4000(wave, spec)

with open(folder+"d4000_{}dust{}.pkl".format(
                        "" if dust else "no_", "" if inst else "_no_inst"
                                             ), "wb") as pkl:
    pickle.dump(d4000, pkl)

# Do D4000 cut
with open(folder+"parent.pkl","rb") as f:
    parent = pickle.load(f)
with open(folder+"parent_gas_info.pkl","rb") as f:
    parent_gas = pickle.load(f)

final = {k:{**parent_gas[k], **parent[k]} for k in d4000.keys() if d4000[k]<1.4}
with open(folder+"cut_final_{}{}.pkl".format("dusty" if dust else "dustless", \
                                   "_no_inst" if not inst else ""), "wb") as f:
    pickle.dump(final, f)
