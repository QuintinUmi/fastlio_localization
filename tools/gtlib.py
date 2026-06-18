#!/usr/bin/env python3
"""Parse Inertial-Explorer GT .txt (honda/kia) and provide init-pose lookup in a map ENU frame."""
import numpy as np
import glob, os
from geo import geodetic2enu, geodetic2ecef, ecef2enu_matrix

# IE .txt columns (after 2-line header):
# UTCTime Week GPSTime Lat Lon H-Ell X Y Z VelBdyX VelBdyY VelBdyZ AccX AccY AccZ Roll Pitch Heading Q VEast VUp VNorth VXe VYe VZe
COL = dict(t=0, lat=3, lon=4, h=5, roll=15, pitch=16, heading=17, q=18, veast=19, vup=20, vnorth=21)

# Map ENU reference LLAs recovered from GLIO_batch_enu.csv (see ttc/geo.py output)
LLA0 = {
    'shatin': (22.424855511, 114.213887863, 5.675),
    'hkstp':  (22.424882804, 114.213935288, 4.984),
    'honda260605': (22.390396562, 114.207876720, 7.052),  # far-south map from honda 13-10-42
}
UNIFIED = 'shatin'  # unified output frame


def load_gt_txt(path):
    rows = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            p = line.split()
            if len(p) < 22:
                continue
            try:
                t = float(p[0])
            except ValueError:
                continue
            if t < 1e9:  # skip header/garbage
                continue
            rows.append([float(p[i]) for i in (0, 3, 4, 5, 15, 16, 17, 19, 20, 21)])
    a = np.array(rows)  # cols: t,lat,lon,h,roll,pitch,heading,veast,vup,vnorth
    return a


def load_car_gt(car, gt_root='/media/quintinumi/dataset/bag/2026-6-5/ground_truth'):
    files = sorted(glob.glob(os.path.join(gt_root, car, '*.txt')))
    segs = [load_gt_txt(f) for f in files]
    segs = [s for s in segs if len(s)]
    allgt = np.vstack(segs)
    allgt = allgt[np.argsort(allgt[:, 0])]
    return allgt  # sorted by time


def lookup(gt, t):
    """Nearest GT row to time t (returns None if >0.5s away)."""
    i = np.searchsorted(gt[:, 0], t)
    cands = [j for j in (i - 1, i) if 0 <= j < len(gt)]
    if not cands:
        return None
    j = min(cands, key=lambda k: abs(gt[k, 0] - t))
    if abs(gt[j, 0] - t) > 0.5:
        return None
    return gt[j]


def gt_to_enu(gt, frame=UNIFIED):
    lla0 = LLA0[frame]
    out = np.zeros((len(gt), 3))
    for i in range(len(gt)):
        out[i] = geodetic2enu(gt[i, 1], gt[i, 2], gt[i, 3], *lla0)
    return out


def init_pose_at(gt, t, frame=UNIFIED, vmin=1.0):
    """Return (pos_enu[3], yaw_enu) for time t in `frame`. yaw from course-over-ground
    (atan2(VN,VE)); falls back to GT Heading if slow."""
    row = lookup(gt, t)
    if row is None:
        return None
    pos = geodetic2enu(row[1], row[2], row[3], *LLA0[frame])
    ve, vn = row[7], row[9]
    speed = np.hypot(ve, vn)
    if speed >= vmin:
        yaw = np.arctan2(vn, ve)
    else:
        # course from a short window of motion around t
        i = np.searchsorted(gt[:, 0], t)
        yaw = None
        for k in range(i, min(i + 300, len(gt))):
            ve2, vn2 = gt[k, 7], gt[k, 9]
            if np.hypot(ve2, vn2) >= vmin:
                yaw = np.arctan2(vn2, ve2); break
        if yaw is None:  # use heading (azimuth from N, CW) -> ENU yaw
            yaw = np.radians(90.0 - row[6])
    return pos, yaw


if __name__ == '__main__':
    for car in ('honda', 'kia'):
        gt = load_car_gt(car)
        enu = gt_to_enu(gt)
        print(f"=== {car}: {len(gt)} GT rows, t[{gt[0,0]:.1f},{gt[-1,0]:.1f}] ===")
        print(f"   unified-ENU extent E[{enu[:,0].min():.0f},{enu[:,0].max():.0f}] "
              f"N[{enu[:,1].min():.0f},{enu[:,1].max():.0f}]")
