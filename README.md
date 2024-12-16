# Multiprocessing for HDF5 packing features from `audioset_tagging_cnn` 

The scripts that pack audio files into [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files from the PANN paper are quite useful. However, one upgrade I sought out was running the packing in parallel since the AudioSet dataset is most usable when separated into a number of HDF5 files, not just one giant file.

I realize now that multiprocessing for this application may have been excessive... However, it has significantly reduced the packing time. 

This is a modified version of the [original repo](https://github.com/qiuqiangkong/audioset_tagging_cnn) which allows for one to pack the HDF5 files in parallel. Here is a screenshot of it in action:

![](imgs/packing_script.png)

For packing some files, use the following command:



