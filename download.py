#!/usr/bin/env python

"""
Download script for SHIFT Dataset (Multi-threaded version).

The data is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License.
Homepage: www.vis.xyz/shift/.
(C)2022, VIS Group, ETH Zurich.


Script usage example:
    python download.py --view  "[front, left_stereo]" \     # list of view abbreviation to download
                       --group "[img, semseg]" \            # list of data group abbreviation to download 
                       --split "[train, val, test]" \       # list of split to download 
                       --framerate "[images, videos]" \     # chooses the desired frame rate (images=1fps, videos=10fps)
                       --shift "discrete" \                 # type of domain shifts. Options: discrete, continuous/1x, continuous/10x, continuous/100x 
                       --threads 4 \                        # number of concurrent download threads
                       dataset_root                         # path where to store the downloaded data

You can set the option to "all" to download the entire data from this option. For example,
    python download.py --view "all" --group "[img]" --split "all" --framerate "[images]" --threads 8 .
downloads the entire RGB images from the dataset using 8 threads.  
"""

import argparse
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import tqdm

if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    import urllib.request as urllib
else:
    import urllib


BASE_URL = "https://dl.cv.ethz.ch/shift/"

FRAME_RATES = [("images", "images (1 fps)"), ("videos", "videos (10 fps)")]

SPLITS = [
    ("train", "training set"),
    ("val", "validation set"),
    ("minival", "mini validation set (for online evaluation)"),
    ("test", "testing set"),
    ("minitest", "mini testing set (for online evaluation)"),
]

VIEWS = [
    ("front", "Front"),
    ("left_45", "Left 45°"),
    ("left_90", "Left 90°"),
    ("right_45", "Right 45°"),
    ("right_90", "Right 90°"),
    ("left_stereo", "Front (Stereo)"),
    ("center", "Center (for LiDAR)"),
]

DATA_GROUPS = [
    ("img", "zip", "RGB Image"),
    ("det_2d", "json", "2D Detection and Tracking"),
    ("det_3d", "json", "3D Detection and Tracking"),
    ("semseg", "zip", "Semantic Segmentation"),
    ("det_insseg_2d", "json", "Instance Segmentation"),
    ("flow", "zip", "Optical Flow"),
    ("depth", "zip", "Depth Maps (24-bit)"),
    ("depth_8bit", "zip", "Depth Maps (8-bit)"),
    ("seq", "csv", "Sequence Info"),
    ("lidar", "zip", "LiDAR Point Cloud"),
]

# Thread-safe progress tracking
progress_lock = Lock()
completed_downloads = 0


class ProgressBar(tqdm.tqdm):
    def update_to(self, batch=1, batch_size=1, total=None):
        if total is not None:
            self.total = total
        self.update(batch * batch_size - self.n)


def setup_logger():
    log_formatter = logging.Formatter(
        "[%(asctime)s] SHIFT Downloader - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)
    return logger


def get_url_discrete(rate, split, view, group, ext):
    url = BASE_URL + "discrete/{rate}/{split}/{view}/{group}.{ext}".format(
        rate=rate, split=split, view=view, group=group, ext=ext
    )
    return url


def get_url_continuous(rate, shift_length, split, view, group, ext):
    url = BASE_URL + "continuous/{rate}/{shift_length}/{split}/{view}/{group}.{ext}".format(
        rate=rate, shift_length=shift_length, split=split, view=view, group=group, ext=ext
    )
    return url


def string_to_list(option_str):
    option_str = option_str.replace(" ", "").lstrip("[").rstrip("]")
    return option_str.split(",")


def parse_options(option_str, bounds, name):
    if option_str == "all":
        return bounds
    candidates = {}
    for item in bounds:
        candidates[item[0]] = item
    used = []
    try:
        option_list = string_to_list(option_str)
    except Exception as e:
        logger.error("Error in parsing options." + str(e))
    for option in option_list:
        if option not in candidates:
            logger.info(
                "Invalid option '{option}' for '{name}'. ".format(option=option, name=name)
                + "Please check the download document (https://www.vis.xyz/shift/download/)."
            )
        else:
            used.append(candidates[option])
    if len(used) == 0:
        logger.error(
            "No '{name}' is specified to download. ".format(name=name)
            + "If you want to download all {name}s, please use '--{name} all'.".format(name=name)
        )
        sys.exit(1)
    return used


def download_file(url, out_file, download_info=None):
    """Download a single file with progress tracking."""
    global completed_downloads
    
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.isfile(out_file):
        try:
            logger.info(f"Starting download: {url}")
            if download_info:
                logger.info(
                    "Downloading - Shift: {shift}, Framerate: {rate}, Split: {split}, View: {view}, Data group: {group}.".format(
                        **download_info
                    )
                )
            
            fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
            f = os.fdopen(fh, "w")
            f.close()
            filename = url.split("/")[-1]
            
            with ProgressBar(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
                urllib.urlretrieve(url, out_file_tmp, reporthook=t.update_to)
            
            os.rename(out_file_tmp, out_file)
            
            with progress_lock:
                completed_downloads += 1
                logger.info(f"Successfully downloaded ({completed_downloads} completed): {filename}")
                
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            # Clean up temporary file if it exists
            if 'out_file_tmp' in locals() and os.path.exists(out_file_tmp):
                try:
                    os.remove(out_file_tmp)
                except:
                    pass
            raise e
    else:
        with progress_lock:
            completed_downloads += 1
            logger.warning(f"Skipping download of existing file ({completed_downloads} completed): {out_file}")


def create_download_task(rate, rate_name, split, split_name, view, view_name, group, ext, group_name, args):
    """Create a download task dictionary."""
    if rate == "videos" and group in ["img"]:
        ext = "tar"
    
    if args.shift == "discrete":
        url = get_url_discrete(rate, split, view, group, ext)
        out_file = os.path.join(args.out_dir, "discrete", rate, split, view, group + "." + ext)
    else:
        shift_length = args.shift.split("/")[-1]
        url = get_url_continuous(rate, shift_length, split, view, group, ext)
        out_file = os.path.join(
            args.out_dir, "continuous", rate, shift_length, split, view, group + "." + ext
        )
    
    download_info = {
        'shift': args.shift,
        'rate': rate_name,
        'split': split_name,
        'view': view_name,
        'group': group_name,
    }
    
    return {
        'url': url,
        'out_file': out_file,
        'download_info': download_info
    }


def main():
    global completed_downloads
    
    parser = argparse.ArgumentParser(description="Downloads SHIFT Dataset public release (Multi-threaded).")
    parser.add_argument("out_dir", help="output directory in which to store the data.")
    parser.add_argument("--split", type=str, default="", help="specific splits to download.")
    parser.add_argument("--view", type=str, default="", help="specific views to download.")
    parser.add_argument("--group", type=str, default="", help="specific data groups to download.")
    parser.add_argument("--framerate", type=str, default="", help="specific frame rate to download.")
    parser.add_argument(
        "--shift",
        type=str,
        default="discrete",
        choices=["discrete", "continuous/1x", "continuous/10x", "continuous/100x"],
        help="specific shift type to download.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="number of concurrent download threads (default: 8)."
    )
    args = parser.parse_args()

    print(
        "Welcome to use SHIFT Dataset download script (Multi-threaded)! \n"
        "By continuing you confirm that you have agreed to the SHIFT's user license.\n"
    )

    frame_rates = parse_options(args.framerate, FRAME_RATES, "frame rate")
    splits = parse_options(args.split, SPLITS, "split")
    views = parse_options(args.view, VIEWS, "view")
    data_groups = parse_options(args.group, DATA_GROUPS, "data group")
    total_files = len(frame_rates) * len(splits) * len(views) * len(data_groups)
    
    logger.info(f"Number of files to download: {total_files}")
    logger.info(f"Using {args.threads} concurrent threads")

    # Check if LiDAR is requested and handle it appropriately
    has_lidar = any(group[0] == "lidar" for group in data_groups)
    if has_lidar:
        logger.warning("LiDAR data is only available for 'center' view. LiDAR downloads will be limited to center view only.")

    # Create all download tasks
    download_tasks = []
    for rate, rate_name in frame_rates:
        for split, split_name in splits:
            for view, view_name in views:
                for group, ext, group_name in data_groups:
                    # Skip LiDAR data for non-center views
                    if group == "lidar" and view != "center":
                        logger.debug(f"Skipping LiDAR data for {view_name} view (only available for center view)")
                        continue
                    
                    task = create_download_task(rate, rate_name, split, split_name, view, view_name, 
                                              group, ext, group_name, args)
                    download_tasks.append(task)

    # Execute downloads with ThreadPoolExecutor
    completed_downloads = 0
    failed_downloads = []
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all download tasks
        future_to_task = {
            executor.submit(download_file, task['url'], task['out_file'], task['download_info']): task 
            for task in download_tasks
        }
        
        # Process completed downloads
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                future.result()  # This will raise any exception that occurred
            except Exception as e:
                failed_downloads.append({
                    'task': task,
                    'error': str(e)
                })
                logger.error(f"Failed to download {task['url']}: {str(e)}")

    # Report results
    successful_downloads = completed_downloads - len(failed_downloads)
    logger.info(f"Download completed! Successfully downloaded: {successful_downloads}/{total_files}")
    
    if failed_downloads:
        logger.error(f"Failed downloads: {len(failed_downloads)}")
        for failed in failed_downloads:
            logger.error(f"  - {failed['task']['url']}: {failed['error']}")
    else:
        logger.info("All downloads completed successfully!")


if __name__ == "__main__":
    logger = setup_logger()
    main()
