# README

## Pickle Files

Every `.pkl` file contains a dictionary of dictionaries. For the first level, the keys are subhalo IDs. For the second level, the keys are described in the **File Contents** section.

### Reading a File

As an example, let's read `cut2_M_r.pkl` into **Python 3**. Because these files were created using Python 3, they must be read using Python 3.

```python3
import pickle

with open('cut2_M_r.pkl', 'rb') as f:
  cut_M_r = pickle.load(f)
```

### Prefixes

cut1 would be the mass > 10^10 Msun cut but it is not saved to file

File Prefix | Cut
------------|----
`cut2_M_r`    | M_r < -19
`parent`      | M_r sans M_* > 10^12 Msun and half mass radius < 2 kpc
`cut3_g-r`    | g-r > 0.655
`cut4`        | "Investigative" cuts; sSFR, radial inversion, or both
`cut_final`   | D4000 < 1.4 for dusty or dustless spectra (inst SFR included)

The `d4000` files contain the D4000 measures for different types of spectra (those with and without dust, and those made with or without the inclusion of the instantaneous star formation rate)

`cut_final.pkl.orig` is used by my (untracked) Jupyter Notebooks to make sure any changes to the `.pkl` files don't change the final dusty D4000 cut.

### File Contents
#### Cut Files
- `M_r`:  Array of M_r for all camera views
- `view`: Camera view with brightest M_r
- `half_mass_rad`: Stellar half mass radius in kpc
- `stellar_mass`: Stellar mass in Msun
- `g-r`: If cut > 2, g-r in the "disk" (Astropy quantity)
- `inner_sSFR_50Myr`/`100Myr`/`1Gyr`: If cut > 3, the inner (r < 2 kpc) time-averaged sSFR. Present in `cut4_radii.pkl` for subhalos that overlap with `cut4_ssfr.pkl`, for some reason?

#### `_gas_info.pkl`
All of these are Astropy quantities, and have units attached

- `total_SFR`: Instantaneous SFR for whole galaxy in Msun/yr
- `total_gas`: Total gas mass in Msun
- `total_sfe`: Instantaneous star formation efficiency for whole galaxy in 1/yr
- `inner_SFR`: Instantaneous SFR for r < 2 kpc in Msun/yr
- `inner_sSFR`: Instantaneous sSFR for r < 2 kpc in 1/yr
- `inner_gas`: Gas mass for r < 2 kpc in Msun
- `inner_sfe`: Instantaneous star formation efficiency for r < 2 kpc in 1/yr
- `mid_sfe`: As above, for 2 kpc < r < 1 half mass radius
- `far_sfe`: As above, for 1-2 half mass radii
- `outer_sfe`: As above, for r > 2 half mass radii

#### `cut3_ssfr.pkl`
This one is *not* a dictionary of dictionaries. Instead, each subhalo ID key returns the 1 Gyr averaged sSFR for that subhalo. Used for plotting sSFR of the g-r sample.


## Python Scripts
- **download_cutouts** and **download_fits** are for bulk downloading particle cutouts and mock FITs files, respectively
- **gas_analysis** produces the `_gas_info.pkl` files by analysing gas particle cutouts. Whether this information is generted for the parent sample or the g-r sample is controlled by a boolean at the top of the file.
- **get_d4000** post-processes FSPS spectra to calculate the D4000 measure (uses Tjitske's function) and saves them in the appropriate `d4000` pickle file (depending on whether or not dust or the instantanous SFR was used in the spectra)
- **illustris_cuts** performs the photometric cuts, and generates `cut2_M_r`, `parent`, and `cut3_g-r.pkl` files
- **selection_investigation** makes a bunch of plots. It's rather obsolete, except that it makes the mass growth history plots.
- **stellar_mass_growth** determines whether a galaxy is radially inverted or has high sSFR in the center. It creates `cut3_ssfr.pkl` and the `cut4` files.
- **stellar_spectra** generates the mock spectra with FSPS, and will either include or disclude dust or the instantaneous SFR
- **utilities** contains helper functions for downloading Illustris API data, splitting work among MPI tasks, and dealing with Illustris domain periodicity

### Script Arguments
All scripts use the same set of command line arguments:

+ `z`: redshift; currently either 0.0 or 0.5
+ `--parent`: run the analysis for the parent sample. Replaces a boolean set at the top of some of the scripts.
+ `--inst`: include instantaneous SFR. Replaces a boolean set at the top of some of the scripts.
+ `--dusty`: include dust in the spectra. Replaces a boolean set at the top of some of the scripts.
+ `--tng`: use TNG instead of the original Illustris.

### Pipeline

To get the information necessary to generate `cut_final`, the scripts should be run as laid out in `python.slurm`. All scripts should be run with the same command line arguments.

## Distributing Work Using MPI and `scatter_work`
```python3
import pickle
from mpi4py import MPI
from utilities import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    # Assemble the subhalos IDs you want to operate on as a numpy array
    # This example assumes they live in a pickled dictionary
    with open("dict.pkl", "rb") as f:
        dictionary = pickle.read(f)
    subhalo_ids = np.array([k for k in dictionary.keys()])
    
    # Any secondary data to be broadcasted should also be read in on the
    # root processor; for instance, supplementary gas data
    with open("gas_data.pkl", "rb") as f:
        secondary_data = pickle.load(f)
    secondary_data = np.arange(5)
else:
    # Variable names need to be declared for any data you want to distribute
    subhalo_ids = None
    secondary_data = None

# This helper function from utilities.py pads and scatters the arrays
halo_subset = scatter_work(subhalo_ids, rank, size)

# Because scattered arrays have to be the same size, they are padded with -1
good_ids = np.where(halo_subset > -1)

# Broadcast the secondary data normally
secondary_data = comm.bcast(secondary_data, root=0)

my_storage = {} # every rank needs their own way of story results, to be combined later
for halo in halo_subset[good_ids]:
    # do stuff

# Gather the individual results onto one process, stitch together, and save
result_lst = comm.gather(my_storage, root=0)
if rank==0:
    storage = {}
    for dic in result_lst:
        for k, v in dic.items():
            storage[k] = v
    with open("these_results.pkl", "wb") as f:
        pickle.dump(storage, f)
        
# If you want to broadcast the compiled data back out to all processes, add this:
else:
    storage = None
storage = comm.bcast(storage, root=0)

```
