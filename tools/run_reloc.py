#!/usr/bin/env python3
"""Run fastlio_localization relocalization for ONE bag/segment against the unified map,
output a TUM trajectory in the unified (shatin ENU) frame.

Orchestration: roscore -> launch node -> wait 'Global map loaded' -> publish /initpose
(computed from GT) -> record /Odom_lio -> rosbag play [t0,t1] filtered -> convert to TUM.
"""
import argparse, os, signal, subprocess, sys, time, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtlib import load_car_gt, init_pose_at, LLA0  # noqa
from geo import geodetic2enu  # noqa

ROS = ("source /opt/ros/noetic/setup.bash && "
       "source /home/quintinumi/packages/ws_fastlio_localization/devel/setup.bash && ")
LIDAR = {'honda': '/velodyne_points', 'kia': '/livox/lidar'}
IMU = {'honda': '/imu/data', 'kia': '/livox/imu'}
LAUNCH = {'honda': 'reloc_unified_velodyne.launch', 'kia': 'reloc_unified_mid360.launch'}
SINGLE_LAUNCH = {'honda': 'reloc_single_velodyne.launch', 'kia': 'reloc_single_mid360.launch'}


def sh(cmd, **kw):
    return subprocess.Popen(['bash', '-lc', ROS + cmd], **kw)


def shout(cmd):
    return subprocess.run(['bash', '-lc', ROS + cmd], capture_output=True, text=True)


def bag_lidar_start(bag, topic):
    """Return (bag_record_start, first_lidar_record_t, first_lidar_header_stamp)."""
    import rosbag
    b = rosbag.Bag(bag, 'r')
    t0 = b.get_start_time()
    rt, st = None, None
    for tp, msg, t in b.read_messages(topics=[topic]):
        rt = t.to_sec()
        st = msg.header.stamp.to_sec()
        break
    b.close()
    return t0, rt, st


def quat_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def onboard_init(bag, t0_rec, frame, win=25.0):
    """Init from the bag's own u-blox /ublox_driver/receiver_lla (NavSatFix).
    Returns (pos_enu, yaw) or None. yaw from onboard motion course."""
    import rosbag, rospy
    pts = []
    with rosbag.Bag(bag, 'r') as b:
        if '/ublox_driver/receiver_lla' not in b.get_type_and_topic_info().topics:
            return None
        for tp, m, t in b.read_messages(topics=['/ublox_driver/receiver_lla'],
                                        start_time=rospy.Time(t0_rec - 2),
                                        end_time=rospy.Time(t0_rec + win)):
            if m.status.status < 0 or m.latitude == 0:
                continue
            pts.append((m.latitude, m.longitude, m.altitude))
    if not pts:
        return None
    enu = np.array([geodetic2enu(p[0], p[1], p[2], *LLA0[frame]) for p in pts])
    pos = enu[0]
    yaw = 0.0
    for i in range(1, len(enu)):
        d = enu[i] - enu[0]
        if np.hypot(d[0], d[1]) > 2.0:
            yaw = math.atan2(d[1], d[0]); break
    return pos, yaw


def wait_log(path, needle, timeout, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            with open(path, 'r', errors='ignore') as f:
                if needle in f.read():
                    return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def convert_tum(bag, out_tum, topic='/Odom_lio', stamp_off=0.0, keep_z=True):
    import rosbag
    n = 0
    with rosbag.Bag(bag, 'r') as b, open(out_tum, 'w') as f:
        for tp, msg, t in b.read_messages(topics=[topic]):
            if msg._type != 'nav_msgs/Odometry':
                continue
            ts = msg.header.stamp.to_sec()
            ts = (ts + stamp_off) if ts > 0 else t.to_sec()
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            z = p.z if keep_z else 0.0
            f.write(f"{ts:.9f} {p.x:.6f} {p.y:.6f} {z:.6f} {q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--car', required=True, choices=['honda', 'kia'])
    ap.add_argument('--bag', required=True, nargs='+',
                    help='one or more bags; multiple are played back-to-back as one session')
    ap.add_argument('--out', required=True, help='output basename (dir/name, no ext)')
    ap.add_argument('--t0', type=float, default=None, help='abs record start (default bag start)')
    ap.add_argument('--t1', type=float, default=None, help='abs record end (default bag end)')
    ap.add_argument('--yaw-offset', type=float, default=0.0, help='deg added to GT-course init yaw')
    ap.add_argument('--rate', type=float, default=1.0)
    ap.add_argument('--stamp-off', type=float, default=0.0)
    ap.add_argument('--pos', type=str, default=None, help='override init pos "x y z"')
    ap.add_argument('--yaw', type=float, default=None, help='override init yaw (rad, enu)')
    ap.add_argument('--init-z', type=float, default=0.0,
                    help='force init z to map datum (GT ellipsoidal height is ~8m off); default 0')
    ap.add_argument('--onboard', action='store_true', help='init from bag u-blox instead of GT')
    ap.add_argument('--map', type=str, default=None,
                    help='single-map experiment: path to ONE map (own ENU frame); uses --frame')
    ap.add_argument('--frame', type=str, default='shatin',
                    help='ENU frame (LLA0 key) for init pose / output: shatin|hkstp|honda260605')
    ap.add_argument('--map-ready-timeout', type=float, default=180.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    node_log = args.out + '.node.log'
    play_log = args.out + '.play.log'
    rec_bag = args.out + '.odom.bag'
    tum = args.out + '.tum'
    meta = args.out + '.meta.txt'

    lidar, imu = LIDAR[args.car], IMU[args.car]
    bag0 = args.bag[0]
    multi = len(args.bag) > 1
    bstart, lrt, lstamp = bag_lidar_start(bag0, lidar)
    if lstamp is None:
        print(f"[ERR] no {lidar} in {bag0}"); sys.exit(2)
    t0 = args.t0 if (args.t0 is not None and not multi) else lrt
    # GT lookup uses sensor header stamp at the play-start frame (approx t0 - lrt + lstamp)
    gt_t = lstamp + (t0 - lrt)

    # init pose
    if args.pos is not None and args.yaw is not None:
        pos = np.array([float(x) for x in args.pos.split()]); yaw = args.yaw; src = 'override'
    elif args.onboard:
        ob = onboard_init(bag0, t0, args.frame)
        if ob is None:
            print(f"[ERR] no onboard GNSS for {args.bag}"); sys.exit(3)
        pos, yaw = ob; src = 'onboard'
    else:
        gt = load_car_gt(args.car)
        ip = init_pose_at(gt, gt_t, frame=args.frame)
        if ip is None:
            print(f"[ERR] no GT near t={gt_t:.2f} for {args.car}; trying onboard")
            ob = onboard_init(bag0, t0, args.frame)
            if ob is None:
                sys.exit(3)
            pos, yaw = ob; src = 'onboard-fallback'
        else:
            pos, yaw = ip; src = 'gt'
    if args.init_z is not None:
        pos = pos.copy(); pos[2] = args.init_z  # use map z datum, not GT ellipsoidal height
    yaw += math.radians(args.yaw_offset)
    qx, qy, qz, qw = quat_yaw(yaw)
    play_off = max(0.0, t0 - bstart)
    dur = (args.t1 - t0) if args.t1 else None

    with open(meta, 'w') as f:
        f.write(f"car={args.car} bag={args.bag}\n")
        f.write(f"bag_start={bstart:.3f} first_lidar_rt={lrt:.3f} first_lidar_stamp={lstamp:.3f}\n")
        f.write(f"t0={t0:.3f} t1={args.t1} gt_t={gt_t:.3f} src={src}\n")
        f.write(f"init_pos={pos.tolist()} init_yaw_deg={math.degrees(yaw):.2f} yaw_offset={args.yaw_offset}\n")
        f.write(f"init_quat=({qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f})\n")
        f.write(f"play_off={play_off:.3f} dur={dur} rate={args.rate}\n")
    print(open(meta).read())

    # 1) roscore
    rc = shout("pgrep -x rosmaster >/dev/null && echo UP || echo DOWN")
    if 'DOWN' in rc.stdout:
        sh("roscore", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)

    # 2) launch node
    launch = SINGLE_LAUNCH[args.car] if args.map else LAUNCH[args.car]
    maparg = f"map:={args.map}" if args.map else ""
    with open(node_log, 'w') as nl:
        node = sh(f"stdbuf -oL -eL roslaunch fastlio_localization {launch} {maparg}",
                  stdout=nl, stderr=subprocess.STDOUT)
    print("[*] waiting for global map load...")
    if not wait_log(node_log, "Global map loaded", args.map_ready_timeout, node):
        print("[ERR] map load timeout/failed; see", node_log)
        shout("rosnode kill /laserMapping"); node.terminate(); sys.exit(4)
    time.sleep(2.0)

    # 3) publish init pose (latched, background)
    shout(f"rosparam set init_pose/position '{pos[0]} {pos[1]} {pos[2]}'")
    shout(f"rosparam set init_pose/orientation '{qx} {qy} {qz} {qw}'")
    ip_pub = sh("rosrun fastlio_localization publish_init_pose.py",
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)

    # 4) record
    rec = sh(f"rosbag record -O {rec_bag} /Odom_lio __name:=relocrec",
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)

    # 5) play filtered window (single bag windowed; multi-bag session plays full back-to-back)
    bags_str = ' '.join(args.bag)
    if multi:
        play_cmd = (f"rosbag play --quiet -r {args.rate} "
                    f"--topics {lidar} {imu} -- {bags_str}")
    else:
        dur_arg = f"-u {dur}" if dur else ""
        play_cmd = (f"rosbag play --quiet -r {args.rate} -s {play_off} {dur_arg} "
                    f"--topics {lidar} {imu} -- {bags_str}")
    print("[*] playing:", play_cmd)
    with open(play_log, 'w') as pl:
        play = sh(play_cmd, stdout=pl, stderr=subprocess.STDOUT)
    play.wait()
    print("[*] play done, draining...")
    time.sleep(4.0)

    # 6) teardown
    shout("rosnode kill /relocrec"); time.sleep(2.0)
    ip_pub.terminate()
    shout("rosnode kill /laserMapping"); time.sleep(1.0)
    node.terminate()
    try: rec.wait(timeout=10)
    except Exception: rec.terminate()
    time.sleep(1.0)

    # 7) convert
    n = convert_tum(rec_bag, tum, stamp_off=args.stamp_off)
    print(f"[OK] {n} odom poses -> {tum}")


if __name__ == '__main__':
    main()
