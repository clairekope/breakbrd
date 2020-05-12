import requests
import numpy as np
import argparse
import warnings

# Are we using MPI? If yes, we must set up MPI runtime first
try:
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    def scatter_work(array, mpi_rank, mpi_size, root=0, dtype=np.int32):
        """ will only work if MPI has been initialized by calling script.
            array should only exist on root & be None elsewhere"""
        if mpi_rank == root:
            scatter_total = array.size
            mod = scatter_total % mpi_size
            if mod != 0:
                print("Padding array for scattering...")
                pad = -1 * np.ones(mpi_size - mod, dtype=dtype)
                array = np.concatenate((array, pad))
                scatter_total += mpi_size - mod
                assert scatter_total % mpi_size == 0
                assert scatter_total == array.size
        else:
            scatter_total = None

        scatter_total = comm.bcast(scatter_total, root=root)
        subset = np.empty(scatter_total//mpi_size, dtype=dtype)
        comm.Scatter(array, subset, root=root)

        return subset
except ImportError: # no mpi4py
    warnings.warn("mpi4py not imported", RuntimeWarning)


# Helper functions
def get(path, params=None, fpath=""):
    
    attempt = 1

    # make HTTP GET request to path
    headers = {"api-key":"5309619565f744f9248320a886c59bec"}
    r = requests.get(path, params=params, headers=headers)
    
    while r.status_code==503 or r.status_code==502: # Server Error; try again
        attempt += 1
        print(f"Error 503 for {path}; attempt {attempt}", flush=True)
        r = requests.get(path, params=params, headers=headers)

    # raise exception for other response codes that aren't HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = fpath + r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

def periodic_centering(x, center, boxsize):
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

# Parse CLI arguments
parser = argparse.ArgumentParser(
    description="Specify Illustris version, redshift, and analysis specifications."
                                )
parser.add_argument('z', type=float, choices=[0.0, 0.03, 0.1, 0.5], action="store",
                    help='Redshift; only 0.0 or 0.5 (or 0.03 or 0.1 for TNG) are currently supported')

#parser.add_argument('-p','--parent', action='store_true', dest='parent',
#                    help='Process parent sample')

parser.add_argument('--no-inst', action='store_false', dest='inst_sfr',
                    help='Exclude instantaneous SFR')

parser.add_argument('--no-dust', action='store_false', dest='dusty',
                    help='Exclude dust from spectra')

parser.add_argument('--tng', action='store_true', dest='tng',
                    help='Use Illustris TNG instead of original')

parser.add_argument('-l','--local', nargs='?', action='store', dest='local',
                    metavar='DIR',
                    help='Use a local copy of the full snapshot, stored in the specified directory. Default depends on "--tng": /mnt/xfs1/home/sgenel/myceph/PUBLIC/[Illustris-1, IllustrisTNG100]',
                    const='/mnt/xfs1/home/sgenel/myceph/PUBLIC/',
                    default=None)

parser.add_argument('-m','--gen-mocks', action='store_true', dest='mock',
                    help='Generate mock magnitudes using FSPS spectra instead of using FITS from the Illustris team')

args = parser.parse_args()

if not args.tng:
    
    littleh = 0.704
    omegaL = 0.7274
    omegaM = 0.2726

    if args.local == '/mnt/xfs1/home/sgenel/myceph/PUBLIC/':
        args.local += 'Illustris-1/'
        
    url_dset = "http://www.illustris-project.org/api/Illustris-1/"

    if args.z==0.0:
        snapnum = 135
        folder = 'z00/'

    elif args.z==0.5:
        snapnum = 103
        folder = 'z05/'

else:

    littleh = 0.6774
    omegaM = 0.2726
    omegaL = 0.6911
    
    if args.local == '/mnt/xfs1/home/sgenel/myceph/PUBLIC/':
        args.local += 'IllustrisTNG100/'
        
    url_dset = "http://www.tng-project.org/api/TNG100-1/"

    if args.z==0.0:
        snapnum = 99
        folder = 'z00_TNG/'

    elif args.z==0.5:
        snapnum = 67
        folder = 'z05_TNG/'

    elif args.z==0.1:
        snapnum = 91
        folder = 'z01_TNG/'

    elif args.z==0.03:
        snapnum = 96
        folder = 'z003_TNG/'
        
url_sbhalos = url_dset + "snapshots/" + str(snapnum) + "/subhalos/"
