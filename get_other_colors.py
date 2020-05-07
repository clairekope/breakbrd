import numpy as np
import glob
from utilities import folder, args 
from get_magnitudes import band_mag

inst = args.inst_sfr
dust = args.dusty

# adjustq for local (non-rusty) dir
folder = "../work_dir/" + folder

files = glob.glob(folder+"spectra/{}inst/{}dust/full/spectra_*.txt".format(
                             "" if inst else "no_",
                             "" if dust else "no_"))

bands = ['SDSS_g','SDSS_r','SDSS_u','WISE_W2','WISE_W3','WISE_W4']
colors = np.zeros( (len(files), len(bands)+1) )

for i, f in enumerate(files):

    sub_id = int(f[-10:-4])
    dat = np.genfromtxt(f)

    print(sub_id, flush=True)
    
    wave = dat[0,:]
    spec = dat[1,:]

    colors[i,0] = sub_id

    for j, band in enumerate(bands):
        colors[i,j+1] = band_mag(wave, spec, band+'_transmission.txt')

sort = np.argsort(colors[:,0])

np.savetxt(folder+"colors_{}{}dust.csv".format(
                             "" if inst else "no_inst_",
                             "" if dust else "no_"),
           colors[sort], fmt="%d %g %g %g %g %g %g", delimiter=',',
           header='Sub ID, g, r, u, W2, W3, W4')

