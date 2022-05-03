#!/usr/bin/env python

import rospy
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import geometry_msgs.msg

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

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


def makePointCloud2Msg(points, frame_time, parent_frame, pcd_format):
    ros_dtype = sensor_msgs.PointField.FLOAT32

    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
        name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate(pcd_format)]

    # header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())
    secs = frame_time // 1000000
    nsecs = (frame_time - (secs * 1000000)) * 1000
    header = std_msgs.Header(frame_id=parent_frame,
                             stamp=rospy.Time(secs=frame_time // 1000000, nsecs=nsecs))

    num_field = len(pcd_format)
    return sensor_msgs.PointCloud2(
        header=header,
        height=points.shape[0],
        width=points.shape[1],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * num_field),
        row_step=(itemsize * num_field * points.shape[1]),
        data=data
    )


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

    # use stable sort (!) so that original order of points is not changed
    # and rows move from "early" to "late"
    sort_indices = np.argsort(ring_index, kind="stable")
    scan = scan[sort_indices, :4]

    # replace all close points (<1.5m) with nan.
    # (does nuscenes map the invalid points to ~(0, 0, 0)? )
    # mask = np.linalg.norm(scan[:, :3], axis=-1) < 1.5
    mask = np.linalg.norm(scan[:, :3], axis=-1) < 1.8
    scan[mask, :] = float("nan")

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

    rospy.init_node('NuscenesLidarPublisher')
    # r = rospy.Rate(20)
    r = rospy.Rate(10)

    scan_publisher = rospy.Publisher('velodyne_points', sensor_msgs.PointCloud2,
                                     queue_size=100)

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
        scan_publisher.publish(makePointCloud2Msg(scan[0], scan[1], "KITTI",
                                                  ['x', 'y', 'z', 'intensity']))
        r.sleep()
