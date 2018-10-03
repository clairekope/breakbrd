import requests
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def scatter_work(array, mpi_rank, mpi_size, root=0):
    """ will only work if MPI has been initialized by calling script.
        array should only exist on root & be None elsewhere"""
    if mpi_rank == root:
        scatter_total = array.size
        mod = scatter_total % mpi_size
        if mod != 0:
            print("Padding array for scattering...")
            pad = -1 * np.ones(mpi_size - mod, dtype='i')
            array = np.concatenate((array, pad))
            scatter_total += mpi_size - mod
            assert scatter_total % mpi_size == 0
            assert scatter_total == array.size
    else:
        scatter_total = None
        #array = None                                                                        

    scatter_total = comm.bcast(scatter_total, root=root)
    subset = np.empty(scatter_total//mpi_size, dtype='i')
    comm.Scatter(array, subset, root=root)

    return subset

def get(path, params=None, fpath=None):
    # make HTTP GET request to path
    headers = {"api-key":"5309619565f744f9248320a886c59bec"}
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
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
