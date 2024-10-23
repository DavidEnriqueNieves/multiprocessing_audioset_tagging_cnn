import numpy as np
from typing import Union
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

from utilities import (create_folder, get_filename, create_logging, 
    float32_to_int16, pad_or_truncate, read_metadata)
import config


def split_unbalanced_csv_to_partial_csvs(args):
    """Split unbalanced csv to part csvs. Each part csv contains up to 50000 ids. 
    """
    
    unbalanced_csv_path = args.unbalanced_csv
    unbalanced_partial_csvs_dir = args.unbalanced_partial_csvs_dir
    
    create_folder(unbalanced_partial_csvs_dir)
    
    with open(unbalanced_csv_path, 'r') as f:
        lines = f.readlines()

    lines = lines[3:]   # Remove head info
    audios_num_per_file = 50000
    
    files_num = int(np.ceil(len(lines) / float(audios_num_per_file)))
    
    for r in range(files_num):
        lines_per_file = lines[r * audios_num_per_file : 
            (r + 1) * audios_num_per_file]
        
        out_csv_path = os.path.join(unbalanced_partial_csvs_dir, 
            'unbalanced_train_segments_part{:02d}.csv'.format(r))

        with open(out_csv_path, 'w') as f:
            f.write('empty\n')
            f.write('empty\n')
            f.write('empty\n')
            for line in lines_per_file:
                f.write(line)
        
        print('Write out csv to {}'.format(out_csv_path))


def download_wavs(args):
    """Download videos and extract audio in wav format.
    """

    # Paths
    csv_path = args.csv_path
    audios_dir = args.audios_dir
    mini_data = args.mini_data
    
    if mini_data:
        logs_dir = '_logs/download_dataset/{}'.format(get_filename(csv_path))
    else:
        logs_dir = '_logs/download_dataset_minidata/{}'.format(get_filename(csv_path))
    
    create_folder(audios_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Download log is saved to {}'.format(logs_dir))

    # Read csv
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    lines = lines[3:]   # Remove csv head info

    if mini_data:
        lines = lines[0 : 10]   # Download partial data for debug
    
    download_time = time.time()

    # Download
    for (n, line) in enumerate(lines):
        
        items = line.split(', ')
        audio_id = items[0]
        start_time = float(items[1])
        end_time = float(items[2])
        duration = end_time - start_time
        
        logging.info('{} {} start_time: {:.1f}, end_time: {:.1f}'.format(
            n, audio_id, start_time, end_time))
        
        # Download full video of whatever format
        video_name = os.path.join(audios_dir, '_Y{}.%(ext)s'.format(audio_id))
        os.system("youtube-dl --quiet -o '{}' -x https://www.youtube.com/watch?v={}"\
            .format(video_name, audio_id))

        video_paths = glob.glob(os.path.join(audios_dir, '_Y' + audio_id + '.*'))

        # If download successful
        if len(video_paths) > 0:
            video_path = video_paths[0]     # Choose one video

            # Add 'Y' to the head because some video ids are started with '-'
            # which will cause problem
            audio_path = os.path.join(audios_dir, 'Y' + audio_id + '.wav')

            # Extract audio in wav format
            os.system("ffmpeg -loglevel panic -i {} -ac 1 -ar 32000 -ss {} -t 00:00:{} {} "\
                .format(video_path, 
                str(datetime.timedelta(seconds=start_time)), duration, 
                audio_path))
            
            # Remove downloaded video
            os.system("rm {}".format(video_path))
            
            logging.info("Download and convert to {}".format(audio_path))
                
    logging.info('Download finished! Time spent: {:.3f} s'.format(
        time.time() - download_time))

    logging.info('Logs can be viewed in {}'.format(logs_dir))

def filepath_to_info(filename):
    # Existing pattern to extract YTID without changing it
    ytid_pattern = r'^.*/([A-Za-z0-9_-]+?)(?=_[0-9]+(?:\.[0-9]+)?(?:[-_][0-9]+(?:\.[0-9]+)?)?\.wav$)'
    ytid_match = re.match(ytid_pattern, filename)
    
    if ytid_match:
        ytid = ytid_match.group(1)
        
        # Pattern to extract start_seconds
        # Looks for an underscore followed by digits, possibly with a decimal
        start_pattern = r'_(\d+(?:\.\d+)?)'
        start_match = re.search(start_pattern, filename)
        
        if start_match:
            start_seconds_str = start_match.group(1)
        else:
            print(f"Failed to extract start seconds from filename: {filename}")
            return None
        
        # Pattern to extract end_seconds
        # Looks for an underscore or hyphen followed by digits, possibly with a decimal, before '.wav'
        end_pattern = r'[-_](\d+(?:\.\d+)?)\.wav$'
        end_match = re.search(end_pattern, filename)
        
        if end_match:
            end_seconds_str = end_match.group(1)
        else:
            print(f"Failed to extract end seconds from filename: {filename}")
            return None
        
        try:
            # Convert the extracted strings to integers (after converting to float to handle decimals)
            start_seconds = float(start_seconds_str)
            end_seconds = float(end_seconds_str)
            return ytid, start_seconds, end_seconds
        except ValueError:
            print(f"Error converting times in file {filename}")
            return None
    else:
        print(f"Failed to parse YTID from filename: {filename}")
        return None

def pack_split(split_start : int, split_end : int, meta_dict : dict, split_idx : int, waveforms_hdf5_path : str, slice_dict : Union[set, dict], log_mod : int, lock : mp.Lock, flat : bool = False, flat_audio_dir : str = None):
        summary : dict = {
            "successes" : [],
            "failures" : {}
        }

        if flat and flat_audio_dir == None:
            raise ValueError("Invalid arguments provided since need to specify flat directory")

        clip_samples = config.clip_samples
        classes_num = config.classes_num
        sample_rate = config.sample_rate

        audios_num : int = split_end - split_start

        iterator = range(split_start, split_end, 1)
        pbar = tqdm(iterator, position=split_idx, desc=f"Split {split_idx}", leave=True, total=split_end - split_start)
        with h5py.File(waveforms_hdf5_path, 'w') as hf:
            hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
            hf.create_dataset('waveform', shape=((audios_num, clip_samples)), dtype=np.int16)
            hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
            hf.create_dataset('meta_csv_idx', shape=((audios_num,)), dtype=np.int32)
            hf.create_dataset('loaded_successfully', shape=((audios_num,)), dtype=np.bool)
            hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

            hf['loaded_successfully'][:] = np.zeros((audios_num,), dtype=np.bool)

            last_total : int = 0
            # Pack waveform & target of several audio clips to a single hdf5 file

            # i is used for the split index relative to the start, i.e. (0 - num_split_files),
            #  and n is used for the index relative to the metadata CSV, i.e. (50k - 100k)
            for i, n in enumerate(iterator):
                # print(f"{n=}")
                # print(f"{meta_dict['cum_hum_targets'][n]=}")

                success : bool = False
                # print(f"{human_labels=}")

                path_info : tuple = meta_dict['audio_path_info'][n]
                ytid, start_seconds, end_seconds = path_info

                # print(f"{(ytid, start_seconds, end_seconds)=}")

                # print(f"{flat=}")
                if not flat:
                    assert isinstance(slice_dict, dict), f"When using nested dirs, slice_dict should be a dictionary, is instead a {type(slice_dict)}"
                    if ytid in slice_dict:
                        time_tuple : tuple = (float(start_seconds), float(end_seconds))

                        if time_tuple in slice_dict[ytid]:
                            audio_path : str = str(slice_dict[ytid][time_tuple])

                            try:
                                (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                                audio = pad_or_truncate(audio, clip_samples)
                                # print(f"{meta_dict['indices'][n]=}, {meta_dict['audio_name'][n]=}")

                                hf['meta_csv_idx'][i] = meta_dict['indices'][n]
                                hf['audio_name'][i] = str(audio_path)
                                hf['waveform'][i] = float32_to_int16(audio)
                                hf['target'][i] = meta_dict['target'][n]
                                hf['loaded_successfully'][i] = True
                                summary["successes"].append(audio_path)
                                success = True
                            except Exception as e:
                                exception : str = str(e)
                        else:
                            exception : str = f"Slice {(time_tuple)} {ytid} not even present inside of slice_dict (for tuple {path_info} )"
                    else:
                        exception : str = f"YTID {ytid} not even present inside of slice_dict (for tuple {path_info} )"
                else:
                    assert isinstance(slice_dict, set), f"When NOT using nested dirs, slice_dict should be set, is instead a {type(slice_dict)}"

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
                            hf['loaded_successfully'][i] = True
                            summary["successes"].append(audio_path)
                            success = True
                        except Exception as e:
                            exception : str = str(e)
                    else:
                        exception : str = f"Path {(audio_path)}  not even present inside of set (for tuple {path_info} )"

                if success:
                    summary["successes"].append(f"{path_info}-{audio_path}")
                    hf['loaded_successfully'][i] = True
                else:
                    # print(f"{exception=}")
                    summary["failures"][str(path_info)] = exception
                    hf['loaded_successfully'][i] = False
                
                current_total : int = len(summary["successes"]) + len(summary["failures"])
                with lock:
                    pbar.update(current_total - last_total)
                    pbar.set_postfix({'num_successes' : len(summary['successes']), 'num_failures' : len(summary['failures'].keys())})
                    last_total = current_total
                # print(f"{json.dumps(summary, indent=4)}")

                errors_path : Path = Path(f"./hdf5s/errors/errors_split_{split_idx}.json")
                errors_path.parent.mkdir(exist_ok=True)
        
                if n % log_mod == 0:
                    with open(f"./hdf5s/errors/errors_split_{split_idx}.json", "w") as f:
                        summary["num_successes"] = len(summary['successes'])
                        summary["num_failures"] = len(summary['failures'].keys())
                        json.dump(summary, f, indent=4)
            pbar.close()

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

    logs_dir = '_logs/pack_waveforms_to_hdf5/{}{}'.format(prefix, get_filename(csv_path))
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))

    # Read csv file
    start : float = time.time()
    meta_dict = read_metadata(csv_path, classes_num, id_to_ix)
    end : float = time.time()

    print(f"Reading of metadata CSV took {end - start} seconds")

    if mini_data:
        mini_num = 10
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][0 : mini_num]

    total_rows : int = len(meta_dict['audio_path_info'])
    assert total_rows == len(meta_dict['target'])
    assert total_rows == len(meta_dict['cum_hum_targets'])

    num_splits : int = 1
    # num_splits : int = 2000

    rows_per_split : int = round(total_rows/num_splits)
    print(f"{rows_per_split}")

    # log current results every 500 clips?

    log_mod : int = 500
    processes : list[mp.Process] = []


    # split_idx : int = 0
    # with mp.Manager() as manager:
    #     lock = manager.Lock()
    #     pack_split(
    #         0,
    #         len(meta_dict["audio_name"]),
    #         meta_dict,
    #         0,
    #         f"./hdf5s/audioset_{split_idx}.h5",
    #         audios_dir,
    #         log_mod,
    #         lock,
    #     )
    
    # NOTE: the object stored here may not necessarily be a dictionary in the case of a flat directory
    audios_dict_path : str = "./eval_nested_audios_list.pkl"

    if(os.path.exists(audios_dict_path)):
        with open(audios_dict_path, "rb") as f:
            print("Loading slice dictionary...")
            slice_dict : Union[set, dict] = pickle.load(f)
            print(f"Slice dictionary loaded...")

            if isinstance(slice_dict, dict):
                print(f"Number of keys for audios dir dictionary is {len(slice_dict.keys())}")
            elif isinstance(slice_dict, set):
                print(f"Number of keys for audios dir dictionary is {len(slice_dict)}")
    else:
        with open(audios_dict_path, "wb") as f:
            print("Preprocessing slice dictionary...")

            if flat:
                slice_dict : set = get_preproc_set(Path(audios_dir))
            else:
                slice_dict : dict = get_preproc_dict(Path(audios_dir))

            pickle.dump(slice_dict, f)

    start : float = time.time()

    print(f"{slice_dict=}")
    # exit()

    lock = mp.Lock()
    for split_idx in range(num_splits):
        range_start : int = split_idx * rows_per_split
        range_end : int = (split_idx + 1) * rows_per_split
        # print(f"{(range_start, range_end)=}")
        # print(f"{[len(split_meta_dict[key]) for key in split_meta_dict]= }")
        # print(f"{[split_meta_dict[key][:10] for key in split_meta_dict]= }")
        
        # print(f"{split_idx=}")
        p = mp.Process(
            target=pack_split,
            args=(
                range_start ,
                range_end ,
                meta_dict ,
                split_idx,
                f"./eval_set.h5",
                slice_dict,
                log_mod,
                lock,
                args.flat,
                args.audios_dir
                )
        )
        p.start()
        processes.append(p)
        # break
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt as e:
        for p in processes:
            p.close()
    
    end : float = time.time()

    print(f"Actual packing {end - start} seconds")

    print("done")

def get_preproc_dict(audios_dir : Path) -> dict:
    """
    for getting a nested directory dictionary of all files, assuming a nested directory structure
    """
    start : float = time.time()
    slice_dict : dict[str, dict[tuple, str]] = {}
    for file in audios_dir.glob(f"*/*.wav"):
        ytid, start_seconds, end_seconds = filepath_to_info(str(file))

        if ytid not in slice_dict:
            slice_dict[ytid] = {}
        
        # NOTE: start_seconds, and end_seconds are in the float type
        slice_tuple : tuple = (start_seconds, end_seconds)
        
        if slice_tuple not in slice_dict[ytid]:
            slice_dict[ytid][slice_tuple] = file

    end : float = time.time()

    print(f"Duration for preprocessing audios dir dictionary is {end - start}")
    return slice_dict

def get_preproc_set(audios_dir : Path) -> set:
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

    parser_split = subparsers.add_parser("split_unbalanced_csv_to_partial_csvs")
    parser_split.add_argument(
        "--unbalanced_csv",
        type=str,
        required=True,
        help="Path of unbalanced_csv file to read.",
    )
    parser_split.add_argument(
        "--unbalanced_partial_csvs_dir",
        type=str,
        required=True,
        help="Directory to save out split unbalanced partial csv.",
    )

    parser_download_wavs = subparsers.add_parser("download_wavs")
    parser_download_wavs.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path of csv file containing audio info to be downloaded.",
    )
    parser_download_wavs.add_argument(
        "--audios_dir",
        type=str,
        required=True,
        help="Directory to save out downloaded audio.",
    )
    parser_download_wavs.add_argument(
        "--mini_data",
        action="store_true",
        default=True,
        help="Set true to only download 10 audios for debugging.",
    )

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
    parser_pack_wavs.add_argument(
        "--mini_data",
        action="store_true",
        default=False,
        help="Set true to only download 10 audios for debugging.",
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

    if args.mode == "split_unbalanced_csv_to_partial_csvs":
        split_unbalanced_csv_to_partial_csvs(args)

    elif args.mode == "download_wavs":
        download_wavs(args)

    elif args.mode == "pack_waveforms_to_hdf5":
        pack_waveforms_to_hdf5(args)

    else:
        raise Exception("Incorrect arguments!")
