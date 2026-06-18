# FAST-LIO Localization — prior-map relocalization

A LiDAR-inertial **map-based relocalization** package built on
[FAST-LIO2](https://github.com/hku-mars/FAST_LIO) (HKU-MARS). The original FAST-LIO builds a
map online; this fork adds a **localization mode** that loads a *prior* point-cloud map and
keeps the platform globally registered to it in real time. A tooling layer (`tools/`) can
also compute the initial pose automatically from post-processed navigation data.

Validated on a multi-vehicle 2026 dataset (Velodyne-32 and Livox MID360) relocalizing
against a stitched city-scale map: scan-to-map plane residual ≈ 0.03–0.07 m.

> Research fork. The LiDAR-inertial odometry core (iEKF, ikd-Tree, IMU preintegration) is
> FAST-LIO2; please cite the upstream papers (see *Acknowledgements*). The original upstream
> README is kept as `README_fastlio_upstream.md`.

---

## 1. How it works

The core idea is simple: **the prior PCD map is loaded into a dedicated ikd-Tree at startup,
and from then on every scan is registered against it with the same point-to-plane residual
and on-manifold iterated-EKF optimization that FAST-LIO already uses** — only the tree holds
the *prior* map instead of an online-accumulated one. No separate registration pipeline is
introduced; localization is just FAST-LIO's optimization running against the loaded map.

```
 prior map PCD ─► ikd-Tree (prior map)
                        ▲
 LiDAR ─► preprocess ─► point-to-plane residual ─►┐
 IMU   ─► preintegration ─────────────────────────┤ iterated EKF ─► /Odom_lio (pose in MAP frame)
 /initpose (initial pose, once) ──────────────────┘
```

* **Initialization** — the first scan needs a global pose: publish it once on `/initpose`
  (`nav_msgs/Odometry`, latched), or let NDT coarse-align the first scans to the map.
* **Tracking** — the iEKF fuses IMU preintegration with the scan↔prior-map plane residual
  every frame, so the estimate stays locked to the map datum.
* **Drift backstop (optional)** — an asynchronous backend periodically re-aligns the current
  scan to the map and corrects slow drift on long / geometry-poor stretches.

`/Odom_lio` is the IMU-body pose **in the prior-map ENU frame**; save it to TUM for
evaluation.

---

## 2. Build

ROS1 (Noetic). Depends on PCL, Eigen, `livox_ros_driver(2)` (for Livox), and the bundled
`IKFoM` / `ikd-Tree` headers.

```bash
# from your catkin workspace root
catkin_make            # or: catkin build
source devel/setup.bash
```

Executable: `fastlio_mapping` (node name `laserMapping` in the launch files).

---

## 3. Workflow

### 3.1 Build a prior map (mapping mode)
Run FAST-LIO mapping (or pure odometry) over a mapping log and save the cloud:
```bash
roslaunch fastlio_localization mapping_velodyne.launch      # or mapping_* / pure_odom_*
```
Any point-cloud map in an ENU frame works (it may also come from an external mapper). Note
the map's **ENU origin** `(lat0, lon0, h0)` — it defines the localization frame.

### 3.2 Relocalize on the prior map (localization mode)
Point the config at your map and play the log:
```bash
# terminal 1: node (loads map, waits for first scan / initpose)
roslaunch fastlio_localization reloc_unified_velodyne.launch       # velodyne
roslaunch fastlio_localization reloc_unified_mid360.launch         # livox mid360
#   or reloc_single_{velodyne,mid360}.launch map:=/path/to/map.pcd

# terminal 2: provide an initial pose, then play
rosparam set /init_pose/position "X Y Z"
rosparam set /init_pose/orientation "qx qy qz qw"
rosrun fastlio_localization publish_init_pose.py
rosbag play --topics <LIDAR> <IMU> -- your.bag
```
Wait for `Global map loaded` before playing. `/Odom_lio` is the localized trajectory.

### 3.3 Automated run with computed initial pose
`tools/` computes the initial pose from post-processed navigation data and orchestrates the
whole run (start node → wait for map → publish `/initpose` → play bag(s) → record
`/Odom_lio` → TUM):
```bash
cd tools
python3 gt_init_pose.py --car kia  --bag your.bag                 # print init pose only
python3 run_reloc.py    --car honda --bag your.bag --out out_dir  # full pipeline
```
See `tools/README.md` for the init-pose math and pipeline details.

---

## 4. Configuration

Two YAMLs are loaded per run: a **sensor** config and a **localization** config.

**Sensor** (`config/velodyne*.yaml`, `config/mid360.yaml`, `config/ouster64.yaml`, …):
`lid_topic`, `imu_topic`, `lidar_type`, `scan_line`, extrinsic LiDAR→IMU, and
`preprocess/timestamp_unit`.

**Localization** (`config/localization*.yaml`):
* `map/map_file_path`, `map/voxel_leaf` (0 = keep input resolution; avoids int32 VoxelGrid
  overflow on city-scale maps);
* `ndt/*` coarse-align resolutions/iterations;
* `global_point_residual/*` (scan↔prior-map factor: `max_points`, `voxel_leaf`,
  `max_point2plane_dist`, `robust_kernel`, fitness weighting);
* `local_point_residual/*`, `submap/*`, `global_align/*`, `threading/*`, `debug/*`.

`config/localization_unified*.yaml` target the stitched city map; `_honda` is a more robust
variant (larger `max_point2plane_dist`, Huber) for harder logs.

---

## 5. Launch files

| Launch | Purpose |
|---|---|
| `mapping_*.launch` | online mapping (avia/horizon/ouster64/velodyne/marsim) |
| `pure_odom_{velodyne,mid360}.launch` | map-less odometry (front-end / map building source) |
| `reloc_unified_{velodyne,mid360}.launch` | relocalize on the unified map, **no rviz** (batch) |
| `reloc_single_{velodyne,mid360}.launch` | relocalize on a single map via `map:=` |
| `reloc_{honda,kia}_rviz*.launch` | relocalize **with rviz**, init pose via `px/py/pz/qx..qw` args |
| `localization_*_*.launch` | scene-specific localization presets |

---

## 6. Conventions & pitfalls (validated the hard way)

- **Map frame is ENU at the map origin.** `/Odom_lio` and `/initpose` are in this frame.
- **Init z = map datum (0), not ellipsoidal height** — they can differ by several metres;
  using ellipsoidal height destabilizes vertical convergence.
- **Body mounting yaw offset matters.** A Y-forward LiDAR/IMU body (e.g. a Velodyne mounted
  Y-forward) needs init `yaw = course − 90°`; an X-forward body needs `0°`. Wrong offset →
  the estimate drifts sideways while yaw *looks* correct.
- **Tilted sensors need a full 3-axis init orientation**, not yaw-only, or IMU/gravity
  preintegration diverges on the first frames.
- **Deskew time unit**: set `preprocess/timestamp_unit` to match the per-point `time` field
  (Velodyne `/velodyne_points` in **seconds** → unit 0; microseconds → 2). A wrong unit
  causes divergence only once the platform moves.
- **Evaluate in 2D horizontal** when the GT vertical datum differs from the map; account for
  the per-log clock offset before computing APE.

---

## 7. Outputs

- `/Odom_lio` — localized IMU-body pose in the map ENU frame (save to TUM for evo).
- `/cloud_registered`, `/cloud_registered_body` — deskewed scan (map / body frame).
- `/global_map`, `/submap`, `/path` — visualization (rviz config in `rviz_cfg/`).

---

## 8. Repository layout

```
config/    sensor + localization YAMLs
launch/    mapping / pure-odom / relocalization launch files
scripts/   publish_init_pose.py (/initpose helper)
src/       laserMapping.cpp (node), preprocess, IMU_Processing
tools/     init-pose pipeline (geo.py, gtlib.py, gt_init_pose.py, run_reloc.py)
msg/       Pose6D.msg
```

---

## 9. Acknowledgements

Built on **FAST-LIO2** (W. Xu, Y. Cai, D. He, J. Lin, F. Zhang, HKU-MARS), with **IKFoM**
and **ikd-Tree**. Relocalization design draws on
[FAST_LIO_LOCALIZATION](https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION). Please cite the
FAST-LIO papers when using this work. See `LICENSE`.
