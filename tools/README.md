# 真值初始化重定位工具（GT → /initpose → FAST-LIO 地图重定位）

通过后处理真值（Inertial Explorer / SPAN 的 .txt）给 fastlio_localization 计算初始位姿，
并可一键编排"起节点→等地图→发初值→播包→录 /Odom_lio→转 TUM"的整条重定位流水线。
2026-06 多车数据集（honda velodyne / kia mid360）对 GLIO 统一大图重定位即用此流程，
重定位对地图面匹配残差 0.03–0.07 m。

## 方法原理

**1. 为什么需要外部初值**
地图坐标系的原点是建图起始位置，重定位包的起点不在原点附近时，必须给 EKF 一个
全局初值。本包通过 `/initpose`（nav_msgs/Odometry，latched）一次性写入 EKF 初始状态
（laserMapping.cpp 首帧应用一次）。

**2. 初值怎么从真值算**
- **时间**：取包内第一帧点云的 header.stamp（传感器戳；注意 rosbag 录制时间比传感器
  戳晚 ~0.1 s，勿混用）。IE 真值第 1 列 UTCTime 与 bag 同为 unix 纪元，可直接查表。
- **位置**：该时刻 GT 的 (lat,lon,H-Ell) → 目标地图 ENU 帧（帧原点 lla0 见 `gtlib.LLA0`，
  由 GLIO 结果反解，工具见 GLIO_ASTRI/tools）。**z 强制 = 0（地图地面基准）**：GT 椭球高
  与地图 z 基准差 ~8 m，直接用会让收敛不稳。
- **航向**：yaw = 对地航向 atan2(VNorth, VEast)（IE 列 VEast/VNorth）；静止时向后找首个
  运动样本，再退化用 GT Heading。然后加**车体安装偏置**：
  - honda（velodyne+外置IMU）车体是 **Y-FORWARD/X-RIGHT/Z-UP** → `yaw_offset = -90°`
    （忘加这 90° 的现象：估计 yaw 看似跟 GT 航向一致，但位置往侧向积分发散）；
  - kia（livox mid360）X-FORWARD → `yaw_offset = 0`。
  roll/pitch 给 0 即可（IMU 重力对齐会自行收敛）。

**3. 无真值的包**
用包内自带 u-blox `/ublox_driver/receiver_lla` 的首个定位 + 短窗运动方向做初值
（`run_reloc.py --onboard`），精度米级，足够全局配准收敛。

**4. 评估约定**
输出 TUM 在地图 ENU 帧。和 GT 比要用 **2D 水平 APE**（z 基准不同），且每车录制机
与 GPS 有 0~1 s 时钟差，需扫描 t_offset 取最小（或把 offset 烧进时间戳后再 evo）。
注意：重定位轨迹 vs 建图轨迹仅差 ~0.07 m；vs IE 真值的 ~0.5 m 是 GLIO 图本身相对
IE 的空间变化偏差（图的 datum），不是重定位误差。

## 工具

| 文件 | 作用 |
|---|---|
| `geo.py` | WGS84 ECEF/ENU 数学（无第三方地理库依赖） |
| `gtlib.py` | IE .txt 真值解析、地图帧 lla0 表（LLA0）、`init_pose_at()` 初值计算 |
| `gt_init_pose.py` | 给定 bag（或时刻）→ 打印初值 + roslaunch/rosparam 粘贴命令 |
| `run_reloc.py` | 整条流水线编排：起节点→等"Global map loaded"→发 /initpose→播包(支持**多包连播**)→录 /Odom_lio→转 TUM |

相关 launch（../launch/）：`reloc_unified_velodyne.launch`、`reloc_unified_mid360.launch`
（无 rviz 批跑）；`reloc_honda_rviz.launch`、`reloc_kia_rviz.launch`（带 rviz 观看，
接收 px/py/qz/qw 覆盖）；`reloc_single_*.launch`（`map:=` 指定单图）。

## 使用

```bash
source /opt/ros/noetic/setup.bash
source ~/packages/ws_fastlio_localization/devel/setup.bash
cd ~/packages/ws_fastlio_localization/src/fastlio_localization/tools

# 1) 只算初值（拿去 rviz launch 覆盖参数）
python3 gt_init_pose.py --car kia --bag /media/.../kia/2026-06-05-12-26-39.bag

# 2) 一键跑一个段（含 QA 用 TUM 输出；honda 默认 yaw-offset 用 -90）
python3 run_reloc.py --car honda \
  --bag /media/.../honda/2026-06-05-12-26-23_0.bag \
  --out /tmp/honda_1226 --yaw-offset -90 --rate 1.0

# 3) 相邻连续包合并为一个会话连播（一个初值跑到底）
python3 run_reloc.py --car kia \
  --bag /media/.../kia/2026-06-05-12-42-08.bag /media/.../kia/2026-06-05-12-45-49.bag \
  --out /tmp/kia_merge

# 4) 无真值包：车载 GNSS 初值
python3 run_reloc.py --car kia --bag /media/.../kia/2026-06-05-12-16-56.bag \
  --out /tmp/kia_obd --onboard

# 5) 换图/换帧（单图实验）：--map 指定 pcd，--frame 指定该图的 ENU 帧名(gtlib.LLA0 键)
python3 run_reloc.py --car honda --bag ... --out ... \
  --map /path/honda260605_05.pcd --frame honda260605 --yaw-offset -90
```

## 注意事项（实测坑）

- **velodyne 去畸变时间单位**：honda `/velodyne_points` 的 per-point `time` 是**秒**
  （0~0.1 跨度），config 必须 `timestamp_unit: 0`（见 `config/velodyne_honda.yaml`）；
  错配成微秒(2) 的现象 = 静止正常、一动就漂。
- **honda 车体 Y-forward**：初值 yaw 必须 course−90°（见上）。
- **大图要降速播包**：0.5 m 图(7.7M 点)可 -r 1.0；0.15 m 图(40M 点) ikd-tree 查询慢
  ~6×，必须 -r 0.5，否则异步全局配准跟不上 → 漂移/发散。特征稀疏区（远南环）任何
  图都建议 -r 0.5。
- **地图覆盖**：包轨迹超出地图覆盖会直接发散；跨多张图的段要用拼接大图
  （见 GLIO_ASTRI/tools 的拼图工具）。
- 新地图帧：先用 GLIO_ASTRI/tools/recover_map_origin.py 反解 lla0，加进 `gtlib.LLA0`。
