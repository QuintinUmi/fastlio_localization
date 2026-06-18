#!/usr/bin/env python3
"""WGS84 geodetic <-> ECEF <-> ENU helpers (no external geo deps) + map-origin recovery."""
import numpy as np

A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2 - F)


def geodetic2ecef(lat_deg, lon_deg, h):
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    N = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - E2) + h) * np.sin(lat)
    return np.array([x, y, z])


def ecef2enu_matrix(lat0_deg, lon0_deg):
    lat0 = np.radians(lat0_deg); lon0 = np.radians(lon0_deg)
    sl, cl = np.sin(lat0), np.cos(lat0)
    so, co = np.sin(lon0), np.cos(lon0)
    return np.array([
        [-so, co, 0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def geodetic2enu(lat, lon, h, lat0, lon0, h0):
    ecef = geodetic2ecef(lat, lon, h)
    ecef0 = geodetic2ecef(lat0, lon0, h0)
    R = ecef2enu_matrix(lat0, lon0)
    return R @ (ecef - ecef0)


def ecef2geodetic(x, y, z):
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(8):
        N = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - E2 * N / (N + h)))
    N = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h


def enu2geodetic(e, n, u, lat0, lon0, h0):
    R = ecef2enu_matrix(lat0, lon0)
    ecef0 = geodetic2ecef(lat0, lon0, h0)
    ecef = ecef0 + R.T @ np.array([e, n, u])
    return ecef2geodetic(*ecef)


def recover_lla0(csv_path, n=2000):
    """Recover the ENU reference LLA used by GLIO from a *GLIO_batch_enu.csv*.
    cols: time,week,sec, lat,lon,alt, yaw,pitch,roll, e,n,u, vx,vy,vz"""
    d = np.loadtxt(csv_path, delimiter=',', max_rows=n)
    lat, lon, alt = d[:, 3], d[:, 4], d[:, 5]
    e, n_, u = d[:, 9], d[:, 10], d[:, 11]
    # invert each row: lla0 ~ enu2geodetic(-enu, ref=row_lla); average
    lats, lons, hs = [], [], []
    for i in range(len(d)):
        la, lo, hh = enu2geodetic(-e[i], -n_[i], -u[i], lat[i], lon[i], alt[i])
        lats.append(la); lons.append(lo); hs.append(hh)
    lla0 = (np.mean(lats), np.mean(lons), np.mean(hs))
    # verify: re-project all rows and report residual
    res = []
    for i in range(len(d)):
        enu = geodetic2enu(lat[i], lon[i], alt[i], *lla0)
        res.append(enu - np.array([e[i], n_[i], u[i]]))
    res = np.array(res)
    return lla0, np.std(lats) * 111320, np.std(lons) * 111320, np.abs(res).max(0)


if __name__ == '__main__':
    import sys
    maps = {
        'shatin': '/home/quintinumi/git_clone/ws_glio_astri/src/GLIO_ASTRI/result_shatin_260520_final/0GLIO_batch_enu.csv',
        'hkstp': '/home/quintinumi/git_clone/ws_glio_astri/src/GLIO_ASTRI/result_hkstp1218/0GLIO_batch_enu.csv',
    }
    lla0s = {}
    for name, p in maps.items():
        lla0, slat, slon, resmax = recover_lla0(p)
        lla0s[name] = lla0
        d = np.loadtxt(p, delimiter=',')
        e, n_, u = d[:, 9], d[:, 10], d[:, 11]
        print(f"=== {name} ===")
        print(f"  lla0 = lat {lla0[0]:.9f}, lon {lla0[1]:.9f}, h {lla0[2]:.3f}")
        print(f"  recovery scatter: lat {slat*1000:.2f} mm, lon {slon*1000:.2f} mm, reproj max |e,n,u| {resmax} m")
        print(f"  ENU extent: E [{e.min():.1f},{e.max():.1f}] ({e.max()-e.min():.0f}m), "
              f"N [{n_.min():.1f},{n_.max():.1f}] ({n_.max()-n_.min():.0f}m), "
              f"U [{u.min():.1f},{u.max():.1f}]")
        print(f"  n keyframes: {len(d)}")
    # transform between the two ENU frames: where is hkstp origin in shatin frame?
    if 'shatin' in lla0s and 'hkstp' in lla0s:
        sh = lla0s['shatin']; hk = lla0s['hkstp']
        hk_in_sh = geodetic2enu(hk[0], hk[1], hk[2], *sh)
        print(f"\n=== frame relation ===")
        print(f"  hkstp origin in shatin ENU: {hk_in_sh} (dist {np.linalg.norm(hk_in_sh[:2]):.1f} m horiz)")
