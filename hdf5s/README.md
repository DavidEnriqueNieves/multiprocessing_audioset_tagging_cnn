This branch is an improved version of the pack_inefficient branch.


The directory structure is as follows:

`errors` is meant for all errors obtained when packing the data

`index_hdf5s` is meant for all HDF5 files that themselves contain filenames and indices in each rows, which correspond to the filename and index of the data point in another HDF5 file. Any HDF5 file inside of `index_hdf5s` is thus a manifest of all the waveforms and to which HDF5 files they are located in.

`pack_hdf5s` contains all of the HDF5 files with the actual waveforms