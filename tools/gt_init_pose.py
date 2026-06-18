#!/usr/bin/env python3
"""Compute the relocalization INITIAL POSE from post-processed ground truth (IE/SPAN .txt)
for a given bag, expressed in the map's ENU frame. Prints plain pose + ready-to-paste
roslaunch overrides / rosparam commands.

Method (see README.md):
  position = GT lla at the bag's first lidar stamp -> ENU in the map frame (gtlib.LLA0),
             z forced to the map ground datum (0) because GT ellipsoidal height is on a
             different datum (~8 m off);
  yaw      = course-over-ground atan2(VNorth,VEast) from GT velocity (falls back to the
             first moving sample / GT heading when stationary) + a BODY-FRAME offset:
             honda velodyne rig is Y-FORWARD -> --yaw-offset -90; livox mid360 X-FORWARD -> 0.

Usage:
  source /opt/ros/noetic/setup.bash
  python3 gt_init_pose.py --car honda --bag /path/xxx.bag                 # honda default -90
  python3 gt_init_pose.py --car kia   --bag /path/yyy.bag --frame shatin
  python3 gt_init_pose.py --car honda --time 1780633583.9 --yaw-offset -90
"""
import argparse, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtlib import load_car_gt, init_pose_at, LLA0

LIDAR = {'honda': '/velodyne_points', 'kia': '/livox/lidar'}
DEFAULT_YAW_OFF = {'honda': -90.0, 'kia': 0.0}   # honda velodyne body is Y-forward


def first_lidar_stamp(bag, topic):
    import rosbag
    with rosbag.Bag(bag, 'r') as b:
        for _, msg, _ in b.read_messages(topics=[topic]):
            return msg.header.stamp.to_sec()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--car', required=True, choices=['honda', 'kia'])
    ap.add_argument('--bag', default=None, help='bag: init time = its first lidar stamp')
    ap.add_argument('--time', type=float, default=None, help='explicit unix time instead of --bag')
    ap.add_argument('--frame', default='shatin', choices=sorted(LLA0),
                    help='map ENU frame (default: shatin = unified map frame)')
    ap.add_argument('--yaw-offset', type=float, default=None,
                    help=f'deg added to GT course (body convention); default {DEFAULT_YAW_OFF}')
    ap.add_argument('--gt-root', default='/media/quintinumi/dataset/bag/2026-6-5/ground_truth')
    ap.add_argument('--init-z', type=float, default=0.0)
    args = ap.parse_args()

    if args.time is not None:
        t = args.time
    elif args.bag:
        t = first_lidar_stamp(args.bag, LIDAR[args.car])
        if t is None:
            sys.exit(f"no {LIDAR[args.car]} in {args.bag}")
    else:
        sys.exit("need --bag or --time")

    gt = load_car_gt(args.car, args.gt_root)
    ip = init_pose_at(gt, t, frame=args.frame)
    if ip is None:
        sys.exit(f"no GT within 0.5 s of t={t:.2f} (GT covers [{gt[0,0]:.1f},{gt[-1,0]:.1f}])")
    pos, yaw = ip
    off = args.yaw_offset if args.yaw_offset is not None else DEFAULT_YAW_OFF[args.car]
    yaw += math.radians(off)
    qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)

    print(f"# car={args.car} frame={args.frame} t={t:.3f} yaw_offset={off:+.0f}deg "
          f"yaw_enu={math.degrees(yaw):.1f}deg")
    print(f"pose: {pos[0]:.3f} {pos[1]:.3f} {args.init_z:.3f}   q: 0 0 {qz:.5f} {qw:.5f}")
    print(f"\n# roslaunch override (reloc_honda_rviz.launch / reloc_kia_rviz.launch):")
    print(f"  px:={pos[0]:.3f} py:={pos[1]:.3f} pz:={args.init_z:.3f} qz:={qz:.5f} qw:={qw:.5f}")
    print(f"\n# rosparam (then rosrun fastlio_localization publish_init_pose.py):")
    print(f"  rosparam set init_pose/position '{pos[0]:.3f} {pos[1]:.3f} {args.init_z:.3f}'")
    print(f"  rosparam set init_pose/orientation '0 0 {qz:.5f} {qw:.5f}'")


if __name__ == '__main__':
    main()
