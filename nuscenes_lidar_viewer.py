#!/usr/bin/env python

from nuscenes.nuscenes import NuScenes
import pptk

import argparse

import numpy as np
import tqdm

parser = argparse.ArgumentParser(description='Nuscenes lidar publisher')
parser.add_argument('nuscenes_dir', metavar='DIR', help='path to dataset')
parser.add_argument('--version', metavar='DIR', default="v1.0-trainval",
                    help='nuscenes dataset version')
parser.add_argument('--scene', metavar='STR', default="",
                    help='scene to publish')
args = parser.parse_args()



def load_from_file(file_name):
    """Load and pad, so that point cloud is 'rectangular' """
    assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

    scan = np.fromfile(file_name, dtype=np.float32)
    # [x, y, z, I, ring_index]
    scan = scan.reshape((-1, 5))
    # sort by ring_index
    ring_index = scan[:, 4].round().astype(np.int32)
    if (ring_index < 0).any() or (ring_index > 31).any():
        raise ValueError(f"Invalid scan {file_name}.")

    sort_indices = np.argsort(ring_index)
    scan = scan[sort_indices, :4]

    row, row_lengths = np.unique(ring_index, return_counts=True)
    if len(row) != 32:
        raise ValueError("Not all rings available.")

    h, w = 32, row_lengths.max()
    # row=0 is at the bottom
    lidar_image = np.empty(shape=(h, w, 4), dtype=np.float32)
    row_begins = np.concatenate(([0], np.cumsum(row_lengths[:-1])))
    for r, l, start in zip(row, row_lengths, row_begins):
        lidar_image[r, :l, :] = scan[start:start+l, :]
        lidar_image[r, l:, :] = float("nan")

    return lidar_image


if __name__ == '__main__':

    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_dir, verbose=True)
    #

    scene = None
    try:
        if args.scene:
            print(f"Found {len(nusc.scene)} scenes. Searching for {args.scene}.")
            index = next(i for i, x in enumerate(nusc.scene) if x["name"] == args.scene)
        else:
            print(f"Found {len(nusc.scene)} scenes. Using first scene.")
            index = 0
        scene = nusc.scene[index]
        print(f"Found scene {args.scene}:")
        print(nusc.scene[index])
    except StopIteration:
        print(f"Scene {args.scene} not found.")
        quit(1)

    nbr_samples = scene["nbr_samples"]
    print(f"The scene has {nbr_samples} samples.")

    # list((timestamp [us], point cloud))
    scans = []

    sample = nusc.get("sample", scene["first_sample_token"])
    next_sample_token: str = sample["data"]["LIDAR_TOP"]

    while next_sample_token:

        lidar_sample_data = nusc.get("sample_data", next_sample_token)
        # data_path, _, _ = nusc.get_sample_data(next_sample_token)
        data_path = nusc.get_sample_data_path(next_sample_token)

        # POINT CLOUD [(x y z I) x P]. Intensity from 0.0 to 255.0
        point_cloud = load_from_file(data_path)
        if point_cloud.dtype != np.float32:
            raise RuntimeError("Point cloud has wrong data type.")
        # [P x (x y z I)]
        scans.append((point_cloud, lidar_sample_data["timestamp"]))

        next_sample_token = lidar_sample_data["next"]

    for scan in tqdm.tqdm(scans):
        pc = scan[0].reshape(-1, 4)
        v = pptk.viewer(pc[:, :3], pc[:, 3])
        v.set(point_size=0.05)
        v.wait()
        v.close()

