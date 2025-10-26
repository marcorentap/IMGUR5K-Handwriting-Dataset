"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
IMGUR5K is shared as a set of image urls with annotations. This code downloads
th images and verifies the hash to the image to avoid data contamination.

Usage:
      python downloaad_imgur5k.py --dataset_info_dir <dir_with_annotaion_and_hashes> --output_dir <path_to_store_images>

Output:
     Images dowloaded to output_dir
     data_annotations.json : json file with image annotation mappings -> dowloaded to dataset_info_dir
"""

import concurrent.futures
import argparse
import hashlib
import json
import numpy as np
import os
import requests

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Processing imgur5K dataset download..."
    )
    parser.add_argument(
        "--dataset_info_dir",
        type=str,
        default="dataset_info",
        required=False,
        help="Directory with dataset information",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images",
        required=False,
        help="Directory path to download the image",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=64,
        required=False,
        help="Maximum number of worker threads for downloading images",
    )
    args = parser.parse_args()
    return args


# Image hash computed for image using md5..
def compute_image_hash(img_path):
    return hashlib.md5(open(img_path, "rb").read()).hexdigest()


# Create a sub json based on split idx
def _create_split_json(anno_json, _split_idx):
    split_json = {}

    split_json["index_id"] = {}
    split_json["index_to_ann_map"] = {}
    split_json["ann_id"] = {}

    for _idx in _split_idx:
        # Check if the idx is not bad
        if _idx not in anno_json["index_id"]:
            continue

        split_json["index_id"][_idx] = anno_json["index_id"][_idx]
        split_json["index_to_ann_map"][_idx] = anno_json["index_to_ann_map"][_idx]

        for ann_id in split_json["index_to_ann_map"][_idx]:
            split_json["ann_id"][ann_id] = anno_json["ann_id"][ann_id]

    return split_json


def download_and_verify_image(
    index, output_dir, hash_dict, headers, invalid_urls, counters
):
    image_path = f"{output_dir}/{index}.jpg"
    image_url = f"https://i.imgur.com/{index}.jpg"

    if os.path.exists(image_path):
        print(f"File already exists for {image_path}, skipping download.")
        return

    print(f"Downloading {image_url}")
    img_data = requests.get(image_url, headers=headers).content

    if len(img_data) < 100:
        print(f"URL retrieval for {index} failed!!\n")
        invalid_urls.append(image_url)
        return

    with open(image_path, "wb") as handler:
        handler.write(img_data)

    counters["tot_evals"] += 1
    current_hash = compute_image_hash(image_path)
    if hash_dict[index] != current_hash:
        print(
            f"For IMG: {index}, ref hash: {hash_dict[index]} != cur hash: {current_hash}"
        )
        os.remove(image_path)
        invalid_urls.append(image_url)
        return
    else:
        counters["num_match"] += 1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(
        f"{args.dataset_info_dir}/imgur5k_hashes.lst", "r", encoding="utf-8"
    ) as _H:
        hashes = _H.readlines()
        hash_dict = {hash.split()[0]: hash.split()[1] for hash in hashes}

    invalid_urls = []
    counters = {"tot_evals": 0, "num_match": 0}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.3"
    }

    max_workers = args.max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                download_and_verify_image,
                index,
                args.output_dir,
                hash_dict,
                headers,
                invalid_urls,
                counters,
            )
            for index in hash_dict.keys()
        ]

        # Optionally wait for all tasks to complete:
        for future in concurrent.futures.as_completed(futures):
            pass  # Could handle exceptions here if desired


if __name__ == "__main__":
    main()
