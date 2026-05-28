from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from tqdm import tqdm


FILELIST_URL = "misc/filelist.json"
DEEPFAKES_DETECTION_URL = "misc/deepfake_detection_filenames.json"

SERVERS = {
    "EU": "http://canis.vc.in.tum.de:8100/",
    "EU2": "http://kaldir.vc.in.tum.de/faceforensics/",
    "CA": "http://falas.cmpt.sfu.ca:8100/",
}

PROJECT_DATASETS = {
    "original": "original_sequences/youtube",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}

OPTIONAL_DATASETS = {
    "DeepFakeDetection_original": "original_sequences/actors",
    "DeepFakeDetection": "manipulated_sequences/DeepFakeDetection",
    "original_youtube_videos_info": "misc/downloaded_youtube_videos_info.zip",
    "original_youtube_videos": "misc/downloaded_youtube_videos.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the FaceForensics++ subsets that are useful for this project. "
            "Default behavior downloads c23 videos for original + four FF++ manipulation methods."
        )
    )
    parser.add_argument("output_path", type=Path, help="Destination directory.")
    parser.add_argument(
        "--compression",
        type=str,
        default="c23",
        choices=["raw", "c23", "c40"],
        help="Compression level. c23 is recommended for this project.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
        choices=list(PROJECT_DATASETS) + list(OPTIONAL_DATASETS),
        help=(
            "Datasets to download. By default, only the minimum subset needed for image-level "
            "deepfake training is downloaded."
        ),
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=None,
        help="Limit the number of sequence videos per dataset for smoke tests.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="EU",
        choices=sorted(SERVERS),
        help="Download server.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive Terms-of-Use confirmation prompt.",
    )
    return parser.parse_args()


def build_base_urls(server_name: str) -> tuple[str, str]:
    server_url = SERVERS[server_name]
    return server_url + "webpage/FaceForensics_TOS.pdf", server_url + "v3/"


def reporthook(count: int, block_size: int, total_size: int) -> None:
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = max(time.time() - start_time, 1e-6)
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / max(total_size, 1))
    sys.stdout.write(
        "\rProgress: %d%%, %d MB, %d KB/s, %d seconds passed"
        % (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


def download_file(url: str, out_file: Path, report_progress: bool = False) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        tqdm.write(f"WARNING: skipping existing file {out_file}")
        return
    fh, tmp_path = tempfile.mkstemp(dir=str(out_file.parent))
    os.close(fh)
    if report_progress:
        urllib.request.urlretrieve(url, tmp_path, reporthook=reporthook)
    else:
        urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, out_file)


def download_files(filenames: list[str], base_url: str, output_path: Path) -> None:
    for filename in tqdm(filenames):
        download_file(base_url + filename, output_path / filename)


def load_ffpp_sequence_list(base_url: str) -> list[str]:
    file_pairs = json.loads(urllib.request.urlopen(base_url + "/" + FILELIST_URL).read().decode("utf-8"))
    original_names: list[str] = []
    for pair in file_pairs:
        original_names.extend(pair)
    return sorted(set(original_names))


def load_manipulated_sequence_list(base_url: str) -> list[str]:
    file_pairs = json.loads(urllib.request.urlopen(base_url + "/" + FILELIST_URL).read().decode("utf-8"))
    names: list[str] = []
    for pair in file_pairs:
        names.append("_".join(pair))
        names.append("_".join(pair[::-1]))
    return names


def load_dfd_sequence_list(base_url: str, actors: bool) -> list[str]:
    filepaths = json.loads(urllib.request.urlopen(base_url + "/" + DEEPFAKES_DETECTION_URL).read().decode("utf-8"))
    key = "actors" if actors else "DeepFakesDetection"
    return list(filepaths[key])


def maybe_trim(filelist: list[str], num_videos: int | None) -> list[str]:
    if num_videos is None or num_videos <= 0:
        return filelist
    print(f"Downloading only the first {num_videos} files for this dataset.")
    return filelist[:num_videos]


def confirm_tos(tos_url: str, assume_yes: bool) -> None:
    print("This script is for academic use only.")
    print("Please make sure you have been approved by FaceForensics++ and agreed to the Terms of Use:")
    print(tos_url)
    if assume_yes:
        return
    print("***")
    print("Press Enter to continue, or CTRL-C to abort.")
    input("")


def download_special_archive(dataset: str, archive_relpath: str, base_url: str, output_root: Path) -> None:
    archive_name = "downloaded_videos.zip" if dataset == "original_youtube_videos" else "downloaded_videos_info.zip"
    print(f"Downloading archive for {dataset} -> {archive_name}")
    download_file(base_url + "/" + archive_relpath, output_root / archive_name, report_progress=True)


def main() -> None:
    args = parse_args()
    tos_url, base_url = build_base_urls(args.server)
    confirm_tos(tos_url, args.yes)
    args.output_path.mkdir(parents=True, exist_ok=True)

    for dataset in args.dataset:
        if dataset in {"original_youtube_videos", "original_youtube_videos_info"}:
            download_special_archive(  # reproduction-oriented source material, not needed for basic training
                dataset,
                OPTIONAL_DATASETS[dataset],
                base_url,
                args.output_path,
            )
            continue

        if dataset == "original":
            relpath = PROJECT_DATASETS[dataset]
            filelist = [name + ".mp4" for name in maybe_trim(load_ffpp_sequence_list(base_url), args.num_videos)]
            out_dir = args.output_path / relpath / args.compression / "videos"
            download_url = f"{base_url}{relpath}/{args.compression}/videos/"
            print(f"Downloading original FF++ videos ({args.compression}) to {out_dir}")
            download_files(filelist, download_url, out_dir)
            continue

        if dataset in {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}:
            relpath = PROJECT_DATASETS[dataset]
            filelist = [name + ".mp4" for name in maybe_trim(load_manipulated_sequence_list(base_url), args.num_videos)]
            out_dir = args.output_path / relpath / args.compression / "videos"
            download_url = f"{base_url}{relpath}/{args.compression}/videos/"
            print(f"Downloading manipulated FF++ videos for {dataset} ({args.compression}) to {out_dir}")
            download_files(filelist, download_url, out_dir)
            continue

        if dataset == "DeepFakeDetection_original":
            relpath = OPTIONAL_DATASETS[dataset]
            filelist = [name + ".mp4" for name in maybe_trim(load_dfd_sequence_list(base_url, actors=True), args.num_videos)]
            out_dir = args.output_path / relpath / args.compression / "videos"
            download_url = f"{base_url}{relpath}/{args.compression}/videos/"
            print(f"Downloading optional DeepFakeDetection originals ({args.compression}) to {out_dir}")
            download_files(filelist, download_url, out_dir)
            continue

        if dataset == "DeepFakeDetection":
            relpath = OPTIONAL_DATASETS[dataset]
            filelist = [name + ".mp4" for name in maybe_trim(load_dfd_sequence_list(base_url, actors=False), args.num_videos)]
            out_dir = args.output_path / relpath / args.compression / "videos"
            download_url = f"{base_url}{relpath}/{args.compression}/videos/"
            print(f"Downloading optional DeepFakeDetection manipulated videos ({args.compression}) to {out_dir}")
            download_files(filelist, download_url, out_dir)
            continue

        raise ValueError(f"Unsupported dataset: {dataset}")


if __name__ == "__main__":
    main()
