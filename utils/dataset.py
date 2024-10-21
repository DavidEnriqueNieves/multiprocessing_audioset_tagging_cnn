import numpy as np
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

def pack_split(split_start : int, split_end : int, meta_dict : dict, split_idx : int, waveforms_hdf5_path : str, audios_dir : str, log_mod : int, lock : mp.Lock):
        summary : dict = {
            "successes" : [],
            "failures" : {}
        }

        clip_samples = config.clip_samples
        classes_num = config.classes_num
        sample_rate = config.sample_rate

        audios_num : int = len(meta_dict['audio_name'])
        assert audios_num == len(meta_dict['target'])
        assert audios_num == len(meta_dict['cum_hum_targets'])

        iterator = range(split_start, split_end, 1)
        pbar = tqdm(iterator, position=split_idx, desc=f"Split {split_idx}", leave=True)
        with h5py.File(waveforms_hdf5_path, 'w') as hf:
            hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
            hf.create_dataset('waveform', shape=((audios_num, clip_samples)), dtype=np.int16)
            hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
            hf.create_dataset('loaded_successfully', shape=((audios_num,)), dtype=np.bool)
            hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

            hf['loaded_successfully'][:] = np.zeros((audios_num,), dtype=np.bool)

            last_total : int = 0
            # Pack waveform & target of several audio clips to a single hdf5 file
            for n in iterator:
                # print(f"{n=}")
                human_labels : list = meta_dict['cum_hum_targets'][n]
                # print(f"{meta_dict['cum_hum_targets'][n]=}")

                success : bool = False
                # print(f"{human_labels=}")
                for label in human_labels:
                    audio_path : str = os.path.join(audios_dir, label + "/" + meta_dict['audio_name'][n] + ".wav")
                    exceptions : list[str] = []
                    try:

                        if os.path.isfile(audio_path):
                            # logging.info('{} {}'.format(n, audio_path))
                            (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                            audio = pad_or_truncate(audio, clip_samples)

                            hf['audio_name'][n] = meta_dict['audio_name'][n].encode()
                            hf['waveform'][n] = float32_to_int16(audio)
                            hf['target'][n] = meta_dict['target'][n]
                            hf['loaded_successfully'][n] = True
                            summary["successes"].append(audio_path)
                            success = True
                            break
                        else:
                            # logging.info('{} File does not exist! {}'.format(n, audio_path))
                            exceptions.append(f"File {audio_path} does not exist!")
                    except Exception as e:
                        exceptions.append(str(e))
                
                if success:
                    summary["successes"].append(audio_path)
                    hf['loaded_successfully'][n] = True
                else:
                    summary["failures"][audio_path] = exceptions
                    hf['loaded_successfully'][n] = False
                
                current_total : int = len(summary["successes"]) + len(summary["failures"])
                with lock:
                    pbar.update(current_total - last_total)
                    pbar.set_postfix({'num_successes' : len(summary['successes']), 'num_failures' : len(summary['failures'].keys())})
                    last_total = current_total
                # print(f"{json.dumps(summary, indent=4)}")
        
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
    meta_dict = read_metadata(csv_path, classes_num, id_to_ix)

    if mini_data:
        mini_num = 10
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][0 : mini_num]

    audios_num = len(meta_dict['audio_name'])

    # Pack waveform to hdf5
    total_time = time.time()

    total_rows : int = len(meta_dict['audio_name'])
    assert total_rows == len(meta_dict['target'])
    assert total_rows == len(meta_dict['cum_hum_targets'])

    num_splits : int = 40

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

    lock = mp.Lock()
    for split_idx in range(num_splits):
        range_start : int = split_idx * rows_per_split
        range_end : int = (split_idx + 1) * rows_per_split
        print(f"{(range_start, range_end)=}")
        # print(f"{[len(split_meta_dict[key]) for key in split_meta_dict]= }")
        # print(f"{[split_meta_dict[key][:10] for key in split_meta_dict]= }")
        
        print(f"{split_idx=}")
        p = mp.Process(
            target=pack_split,
            args=(
                range_start ,
                range_end ,
                meta_dict ,
                split_idx,
                f"./hdf5s/audioset_{split_idx}.h5",
                audios_dir,
                log_mod,
                lock
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

    print("done")

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

    args = parser.parse_args()

    if args.mode == "split_unbalanced_csv_to_partial_csvs":
        split_unbalanced_csv_to_partial_csvs(args)

    elif args.mode == "download_wavs":
        download_wavs(args)

    elif args.mode == "pack_waveforms_to_hdf5":
        pack_waveforms_to_hdf5(args)

    else:
        raise Exception("Incorrect arguments!")
