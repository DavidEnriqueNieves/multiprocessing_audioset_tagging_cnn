"""
Script to generate a csv file with the indices of the valid samples in the hdf5
files, be they HDF5 files for packed audio or HDF5 files for packed mel
spectrograms.
"""
from pathlib import Path
import numpy as np
import h5py
import pandas as pd

pack_hdf5s_dir: Path = Path("/datasets/AudioSet/pann_repo/hdf5s/melspec_hdf5s_1.6m/val")
meta_csv_path: Path = Path("/datasets/AudioSet/pann_repo/hdf5s/melspec_hdf5s_1.6m")

def debug():
    PORT: int = 5678
    import debugpy

if __name__ == "__main__":

    meta_df: pd.DataFrame = pd.DataFrame()

    for file in pack_hdf5s_dir.glob("*.h5"):
        print(file)
        h5py_file: h5py.File = h5py.File(file, "r")
        if "valid" in h5py_file.keys():
            valid_mask: np.array = np.array(h5py_file["valid"])
        elif "loaded_successfully" in h5py_file.keys():
            valid_mask: np.array = np.array(h5py_file["loaded_successfully"])

        print(f"{valid_mask.shape=}")
        # count on how many valid samples are in the file
        valid_count: int = np.sum(valid_mask, dtype=int)
        print(f"{valid_count=}")

        indices: np.array =np.where(valid_mask)[0]
        audio_name: np.array = np.array([file.name for x in range(indices.shape[0])])

        meta_df = pd.concat([meta_df, pd.DataFrame({
            "filename": audio_name,
            "audio_index": indices,
            # "human_labels": 
        })])

        # break
    
    print("Writing to csv...")
    meta_df.to_csv(meta_csv_path / "val_hdf5_idxs.csv", index=False)
    print("done")

