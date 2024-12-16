"""
Script to pack audio files into the HDF5 format. Run using a command like so:

```
python3 utils/dataset.py pack_waveforms_to_hdf5
--csv_path="./unbalanced_train_segments.csv"
--audios_dir="/datasets/AudioSet/dvc-audioset/audioset"
--waveforms_hdf5_path="./hdf5s/unbalanced_train_segments.h5" 
--debug

python3 utils/dataset.py pack_waveforms_to_hdf5 \
--csv_path="./unbalanced_train_segments.csv" \
--audios_dir="/datasets/AudioSet/dvc-audioset/audioset" \
--waveforms_hdf5_path="/datasets/AudioSet/pann_repo/hdf5s/pack_1.6m_hdf5s/train" 

date ; time python3 utils/dataset.py pack_waveforms_to_hdf5 --csv_path="./unbalanced_train_segments.csv" --audios_dir="/datasets/AudioSet/dvc-audioset/audioset" --waveforms_hdf5_path="/datasets/AudioSet/pann_repo/hdf5s/pack_1.6m_hdf5s/train" ; date
```
"""
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import numpy as np
from typing import Union, Tuple, List, Optional
import pickle
from pathlib import Path
import re
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa
import json
from tqdm import tqdm
import multiprocessing as mp
from dataclasses import dataclass
from utilities import (create_folder, get_filename, create_logging, 
    float32_to_int16, pad_or_truncate, read_metadata)
import config
import pandas as pd

@dataclass
class PathWMeta:
    file_path: Path
    row_meta_idx: int
    target: list[int]

@dataclass
class PreProcSet:
    nested_dict: dict[str, dict[tuple, PathWMeta]]
    audios_dir: str
    raw_list: List[PathWMeta]
    raw_keys: List[Tuple[str, float, float]]
    flat: bool
    
    def __len__(self) -> int:
        # assert len(self.raw_list) == len(self.raw_keys)
        return len(self.raw_list)

def filepath_to_info(filename) -> Tuple[str, float, float]:
    # youtube id is 11 chars long, followed by an underscore

    ytid: str = filename[:11]
    # assert len(ytid) == 11

    # start and end times are separated by a dash or a -
    rest: str = filename[12:]
    # print(f"{rest=}")
    if "_" in rest:
        start, end = [float(x) for x in rest.split("_")]
    elif "-" in filename:
        start, end = [float(x) for x in rest.split("-")]
    else:
        raise ValueError(f"Invalid filename {filename}")

    return ytid, start, end

def pack_split(preproc_set: PreProcSet, split_idx : int, waveforms_hdf5_path : str, log_mod : int, lock : mp.Lock, meta_df: pd.DataFrame):
        summary : dict = {
            "successes" : {},
            "failures" : {}
        }

        if preproc_set.flat and preproc_set.audios_dir == None:
            raise ValueError("Invalid arguments provided since need to specify flat directory")

        clip_samples = config.clip_samples
        classes_num = config.classes_num
        sample_rate = config.sample_rate

        audios_num : int = len(preproc_set.raw_keys)
        print(f"{audios_num=}")

        pbar = tqdm(preproc_set.raw_keys, position=split_idx, desc=f"Split {split_idx}", leave=True, total=len(preproc_set.raw_keys))
        with h5py.File(waveforms_hdf5_path, 'w') as hf:
            hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
            hf.create_dataset('waveform', shape=((audios_num, clip_samples)), dtype=np.int16)
            hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=bool)
            hf.create_dataset('meta_csv_idx', shape=((audios_num,)), dtype=np.int32)
            hf.create_dataset('valid', shape=((audios_num,)), dtype=bool)

            hf.attrs.create("feature_type", data="waveform", dtype="S20")
            hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

            hf['valid'][:] = np.zeros((audios_num,), dtype=bool)

            last_total : int = 0
            # Pack waveform & target of several audio clips to a single hdf5 file

            # i is used for the split index relative to the start, i.e. (0 - num_split_files),
            #  and n is used for the index relative to the metadata CSV, i.e. (50k - 100k)
            for i, path_info in enumerate(pbar):
                something: int
                # print(f"{n=}")
                # print(f"{meta_dict['cum_hum_targets'][n]=}")

                inner_pack_loop(preproc_set, split_idx, log_mod, lock, meta_df, summary, clip_samples, classes_num, sample_rate, pbar, hf, last_total, i, path_info)
            pbar.close()

def inner_pack_loop(preproc_set, split_idx, log_mod, lock, meta_df, summary, clip_samples, classes_num, sample_rate, pbar, hf, last_total, i, path_info):
    success : bool = False
                # print(f"{human_labels=}")
    ytid, start_s, end_s = path_info        
                # print(f"{flat=}")
    # if not preproc_set.flat:
    if ytid in preproc_set.nested_dict:
        time_tuple : tuple = (float(start_s), float(end_s))

        if time_tuple in preproc_set.nested_dict[ytid]:
            success, exception, audio_path = handle_nested_dirs(path_info, preproc_set, meta_df, summary, clip_samples, classes_num, sample_rate, hf, i, ytid, time_tuple)
        else:
            exception : str = f"Slice {(time_tuple)} {ytid} not even present inside of slice_dict (for tuple {path_info} )"
    else:
        exception : str = f"YTID {ytid} not even present inside of slice_dict (for {path_info} )"

    #             # TODO: fix
    #             # flat directory
    # else:
    #     success, exception, audio_path = handle_flat_dirs(summary, clip_samples, sample_rate, hf, i, path_info, ytid)

    if success:
        summary["successes"][str(path_info)] = (f"{path_info}-{audio_path}")
        hf['valid'][i] = True
    else:
                    # print(f"{exception=}")
        summary["failures"][str(path_info)] = exception
        hf['valid'][i] = False
                
    with lock:
        pbar.set_postfix({'num_successes' : len(summary['successes']), 'num_failures' : len(summary['failures'].keys())})

    errors_path : Path = Path(f"./hdf5s/errors/errors_split_{split_idx}.json")
    errors_path.parent.mkdir(exist_ok=True)
        
    if i % log_mod == 0:
        log_summary_file(split_idx, log_mod, summary, i)

def log_summary_file(split_idx, log_mod, summary, i):
    with open(f"./hdf5s/errors/errors_split_{split_idx}.json", "w") as f:
        summary["num_successes"] = len(summary['successes'])
        summary["num_failures"] = len(summary['failures'].keys())
        json.dump(summary, f, indent=4)

def handle_flat_dirs(summary, clip_samples, sample_rate, hf, i, path_info, ytid):
    audio_name : str = f"Y{ytid}.wav" 
    if audio_name in slice_dict:
        audio_path : str = f"{flat_audio_dir}/{audio_name}"
        try:
            (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            audio = pad_or_truncate(audio, clip_samples)
                            # print(f"{meta_dict['indices'][n]=}, {meta_dict['audio_name'][n]=}")

            hf['meta_csv_idx'][i] = meta_dict['indices'][n]
            hf['audio_name'][i] = str(audio_path)
            hf['waveform'][i] = float32_to_int16(audio)
            hf['target'][i] = meta_dict['target'][n]
            hf['valid'][i] = True
            summary["successes"].append(audio_path)
            success = True
        except Exception as e:
            exception : str = str(e)
            raise RuntimeError(f"Exception while loading audio at path {audio_path}")
    else:
        exception : str = f"Path {(audio_path)}  not even present inside of set (for tuple {path_info} )"
        raise RuntimeError(f"Exception while loading audio at path {audio_path}")
    return success,exception,audio_path

def handle_nested_dirs(path_info, preproc_set, meta_df, summary, clip_samples, classes_num, sample_rate, hf, i, ytid, time_tuple) -> Tuple[bool, str, Path]:
    path_w_meta: PathWMeta = preproc_set.nested_dict[ytid][time_tuple]
    exception: str = None
    success: bool = False
    if isinstance(path_w_meta, PathWMeta):
        audio_path: Path = path_w_meta.file_path
        meta_idx: int = path_w_meta.row_meta_idx
        meta_target: list[int] = path_w_meta.target
        meta: pd.Series = meta_df.iloc[meta_idx]

        try:
            (audio, _) = librosa.core.load(str(audio_path), sr=sample_rate, mono=True)
            audio = pad_or_truncate(audio, clip_samples)
                                    # print(f"{meta_dict['indices'][n]=}, {meta_dict['audio_name'][n]=}")
            hf['meta_csv_idx'][i] = meta["index"]
            hf['audio_name'][i] = str(audio_path)
            hf['waveform'][i] = float32_to_int16(audio)
            target_arr: np.array = np.zeros((classes_num,), dtype=bool)

            assert len(meta_target) > 0, "Metadata has no valid targets despite being valid"
            for idx in meta_target:
                target_arr[idx] = 1

            hf['target'][i] = target_arr
            hf['valid'][i] = True
            success = True
        except Exception as e:
            # print out e 
            exception : str = str(e)
    else:
        exception : str = f"Slice not loaded from metadata properly"
    return success, exception, audio_path

        # logging.info('Write to {}'.format(waveforms_hdf5_path))
        # logging.info('Pack hdf5 time: {:.3f}'.format(time.time() - total_time))


def pack_waveforms_to_hdf5(args):
    """Pack waveform and target of several audio clips to a single hdf5 file. 
    This can speed up loading and training.
    """

    # Arguments & parameters
    audios_dir = args.audios_dir
    csv_path = args.csv_path
    waveforms_hdf5_path = args.waveforms_hdf5_path
    mini_data = args.mini_data
    flat : bool = args.flat

    clip_samples = config.clip_samples
    classes_num = config.classes_num
    sample_rate = config.sample_rate
    id_to_ix = config.id_to_ix

    # Paths
    if mini_data:
        prefix = 'mini_'
        waveforms_hdf5_path += '.mini'
    else:
        prefix = ''

    create_folder(os.path.dirname(waveforms_hdf5_path))

    logs_dir: str = '_logs/pack_waveforms_to_hdf5/{}{}'.format(prefix, get_filename(csv_path))
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))

    # Read csv file
    print("Reading metadata CSV...")
    start : float = time.time()
    meta_dict = read_metadata(csv_path, classes_num, id_to_ix)
    meta_df: pd.DataFrame = pd.read_csv(csv_path, delimiter='|')
    end : float = time.time()

    print(f"Reading of metadata CSV took {end - start} seconds")

    if mini_data:
        mini_num = 10
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][0 : mini_num]

    total_rows : int = len(meta_dict['audio_path_info'])
    assert total_rows == len(meta_dict['target'])
    assert total_rows == len(meta_dict['cum_hum_targets'])


    # NOTE: the object stored here may not necessarily be a dictionary in the case of a flat directory
    audios_dict_path : str = "./eval_nested_audios_list.pkl"

    if(os.path.exists(audios_dict_path)):
        with open(audios_dict_path, "rb") as f:
            print("Loading slice dictionary...")
            start: float = time.time()
            preproc_set : Union[set, dict] = pickle.load(f)
            end: float = time.time()
            print(f"Loaded in {end - start} seconds")
            print(f"Slice dictionary loaded...")
    else:
        with open(audios_dict_path, "wb") as f:
            print("Preprocessing slice dictionary...")

            preproc_set : PreProcSet = get_preproc_dict(Path(audios_dir), meta_df, flat)

            print("Dumping the preproc set to a pickle file")
            start: float = time.time()
            pickle.dump(preproc_set, f)
            end: float = time.time()
            print(f"Dumping the preproc set took {end - start} seconds")
        
    # print(f"{preproc_set=}")

    # we're assuming that you have ALL the data
    num_splits: int = 40

    # split the PreProcSet into num_splits

    print(f"{len(preproc_set.raw_keys)=}")
    preproc_splits: list[PreProcSet] = []

    assert len(preproc_set.raw_list) == len(preproc_set.raw_keys)
    raw_list_splits: list[list[PathWMeta]] = np.array_split(preproc_set.raw_list, num_splits)
    raw_key_splits: list[list[Tuple[str, float, float]]] = np.array_split(preproc_set.raw_keys, num_splits)

    assert len(raw_list_splits) == num_splits
    print(f"True length for a split is: {len(raw_list_splits[0])}")
    print(f"Expected length for a split is: {int(total_rows / num_splits)}")

    total_key_len: int = 0
    total_list_split_len: int = 0
    for split_idx in range(num_splits):
        print(f"{len(raw_list_splits[split_idx])=}")
        split_preproc: PreProcSet = PreProcSet(
            preproc_set.nested_dict,
            preproc_set.audios_dir,
            raw_list_splits[split_idx],
            raw_key_splits[split_idx],
            flat=flat
            )
        total_key_len += len(split_preproc.raw_keys)
        total_list_split_len += len(split_preproc.raw_list)
        preproc_splits.append(split_preproc)
    
    assert total_key_len == total_list_split_len
    print(f"{total_key_len=}, {total_list_split_len=}")
    
    print("done...")

    # log current results every 500 clips?
    log_mod : int = 500
    processes : list[mp.Process] = []

    # start : float = time.time()

    # print(f"{slice_dict=}")
    # # exit()


    lock = mp.Lock()
    # # pack_split(preproc_splits[0], 0, waveforms_hdf5_path, log_mod, lock, meta_df)
    for split_idx in range(num_splits):
        split_filepath: Path = Path(waveforms_hdf5_path) / Path(f"audioset_{split_idx}.h5")    
        # print(f"{(range_start, range_end)=}")
        # print(f"{[len(split_meta_dict[key]) for key in split_meta_dict]= }")
        # print(f"{[split_meta_dict[key][:10] for key in split_meta_dict]= }")
        
        print(f"length of preproc splits is {len(preproc_splits[split_idx].raw_keys)}")
        # print(f"{split_idx=}")
        p = mp.Process(
            target=pack_split,
            args=(
                preproc_splits[split_idx],
                split_idx,
                str(split_filepath),
                log_mod,
                lock,
                meta_df
                )
        )
        p.start()
        processes.append(p)
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt as e:
        for p in processes:
            p.close()
    
    # end : float = time.time()

    # print(f"Actual packing {end - start} seconds")

    print("done")


def get_preproc_dict_serial(paths: list[Path], audios_dir: Path, meta_df: pd.DataFrame, flat: bool) -> PreProcSet:
    """
    for getting a nested directory dictionary of all files, assuming a nested directory structure
    """
    
    start : float = time.time()
    slice_dict : dict[str, dict[tuple, PathWMeta]] = {}
    all_files: list = []
    missing: int = 0

    end : float = time.time()
    print(f"Missing files: {missing}")
    print(f"Duration for preprocessing audios dir dictionary is {end - start}")

    preproc_set: PreProcSet = PreProcSet(slice_dict, str(audios_dir), all_files, flat=flat)
    return preproc_set

# def process_split(file, meta_df):

def get_preproc_dict(audios_dir: Path, meta_df: pd.DataFrame, flat: bool) -> PreProcSet:
    """
    Gives you a PreProcSet object containing the nested dictionary of all files
    and their associated metadata
    """
    start_s = time.time()
    slice_dict = {}
    all_files = []
    missing = 0
    paths = list(audios_dir.glob(f"*/*.wav"))
    all_keys: list[Tuple[str, float, float]] = []
    num_splits: int = 6

    # first, get a nested dictionary to loop through

    duplicates: list = []
    files_nest: dict = {}
    for path in tqdm(paths):    
        ytid, start_s, end_s = filepath_to_info(path.stem)
        if ytid not in files_nest:
            files_nest[ytid] = {}

        if (start_s, end_s) in files_nest[ytid]:
            print(f"Duplicate {(ytid, start_s, end_s)}")
            duplicates.append(path)
            continue
        else:
            files_nest[ytid][(start_s, end_s)] = path
            all_files.append(path)
            all_keys.append((ytid, start_s, end_s))

    print(f"Number of duplicates is {len(duplicates)}")
    all_files: list = list(set(all_files))
    all_keys: list = list(set(all_keys))
    assert len(all_files) == len(all_keys)

    print("Creating subset of metadata CSV...")
    start: float = time.time()

    # add the metadata information to the dictionary
    total_matches: int = 0
    recollected_keys: list[Tuple[str, float, float]] = []
    nested_dict: dict[str, dict[tuple, PathWMeta]] = {}

    errors: list = []
    for i, row in tqdm(meta_df.iterrows(), total=len(meta_df), leave=True):
        ytid: str = row['YTID']
        start_s: float = row['start_seconds']
        end_s: float = row['end_seconds']

        if ytid in files_nest:
            if (start_s, end_s) in files_nest[ytid]:

                machine_labels: str = row['positive_labels']
                machine_labels: list[str] = machine_labels.replace("\"", "").split(",")
                human_labels: list[str] = [config.id_to_lb[x] for x in machine_labels]
                human_labels.sort()
                human_label_idxs: list[int] = [config.labels.index(x) for x in human_labels]
                files_nest[ytid][(start_s, end_s)] = PathWMeta(files_nest[ytid][(start_s, end_s)], i, human_label_idxs)
                recollected_keys.append((ytid, start_s, end_s))
                total_matches+=1
            else:
                errors.append(f"YTID in files but start {start_s} and end {end_s} for that ID not in files")
        else:
            errors.append(f"YTID {ytid} not in files_nest")

        
    end: float = time.time()    

    # get the missing keys
    missing_keys: list[Tuple[str, float, float]] = set(all_keys) - set(recollected_keys)
    print(f"{len(missing_keys)} missing keys")
    print(missing_keys)
    print(f"Duration for creating subset of metadata CSV is {end - start}")
    # print(f"Errors are \n\n{errors}\n")

    print(f"{len(all_keys) - total_matches} files missing from the metadata CSV")
    # assert total_matches == len(all_files), print(f"{total_matches=}, {len(all_files)=} should match!")   
    if total_matches > len(all_files):
        raise ValueError("More matches than files")
    elif total_matches < len(all_files):
        print("Difference in matches and files")
        print(f"{len(all_files) - total_matches} files missing from the metadata CSV")
        # raise ValueError("Less matches than files")

    # NOTE: we mainly care about all_keys, not necessarily all_files since there might be duplicates
    return PreProcSet(files_nest, str(audios_dir), all_files, raw_keys=all_keys, flat=flat)

def get_preproc_set(audios_dir : Path) -> PreProcSet:
    """
    for getting a flat list of all files, assuming a flat directory structure
    """
    start : float = time.time()
    files_set : list = []

    for file in audios_dir.glob(f"*/*.wav"):
        files_set.append(file.name)

    end : float = time.time()

    files_set = set(files_set)

    print(f"Duration for preprocessing audios dir dictionary is {end - start}")
    return files_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_pack_wavs = subparsers.add_parser("pack_waveforms_to_hdf5")
    parser_pack_wavs.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path of csv file containing audio info to be downloaded.",
    )
    parser_pack_wavs.add_argument(
        "--audios_dir",
        type=str,
        required=True,
        help="Directory to save out downloaded audio.",
    )
    parser_pack_wavs.add_argument(
        "--waveforms_hdf5_path",
        type=str,
        required=True,
        help="Path to save out packed hdf5.",
    )
    parser_pack_wavs.add_argument( "--debug", action="store_true", help="Whether to launch things in debug mode")
    parser_pack_wavs.add_argument( "--flat", action="store_true", help="Whether the directory structure with all the wave files is nested", default=False)

    args = parser.parse_args()

    if args.debug:
        print(f"Debugpy launched")
        import debugpy
        PORT : int = 5678
        debugpy.listen(PORT)
        debugpy.wait_for_client()

    # print(f"{args.flat=}")
    if args.mode == "pack_waveforms_to_hdf5":
        pack_waveforms_to_hdf5(args)

    else:
        raise Exception("Incorrect arguments!")
