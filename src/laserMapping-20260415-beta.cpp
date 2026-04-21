// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/SVD>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/pcl_config.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/filter.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
static bool g_local_tree_empty = false;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, lid_topic2, imu_topic;

// --- localization params (subset for step 1) ---
struct LocalizationParams {
    std::string map_file_path = "";
    double map_voxel_leaf = 0.4;
    bool map_remove_outliers = false;
    int map_outlier_nb_neighbors = 20;
    double map_outlier_stddev_mul = 2.0;

    // global point residual config (later steps will use)
    bool gpr_enable = true;
    int gpr_max_points = 8000;
    double gpr_voxel_leaf = 0.5;
    int gpr_plane_min_neighbors = 10;
    double gpr_max_point2plane_dist = 1.5;
    double gpr_sigma_m = 0.06;
    std::string gpr_robust_kernel = "huber";
    double gpr_huber_delta = 0.5;
    double gpr_cauchy_c = 0.5;

    // ROI (for later use when querying global map)
    double roi_radius = 60.0;
};
LocalizationParams g_loc;

// ===== 后端定位线程相关 =====
struct BackendConfig {
    bool enable = false;                    // 是否启用后端
    int min_local_points = 50000;           // 触发条件：local tree 最小点数
    float min_local_radius = 30.0f;         // 触发条件：local map 最小半径
    int min_frames = 100;                   // 触发条件：最小运行帧数
    
    float submap_radius = 25.0f;            // submap 提取半径
    float submap_voxel_size = 0.4f;         // submap 降采样
    
    int update_interval = 1;                // 每 N 帧更新一次
    float max_point2plane_dist = 2.5f;      // 点到面最大距离
    float global_residual_weight = 6.0f;    // 全局残差权重
    int max_global_points = 4000;           // 最大全局点数
};
BackendConfig g_backend_cfg;

struct BackendAlignConfig {
    bool enable = true;                     // continuous NDT/GICP alignment
    int stride = 1;                         // run every N backend updates
    double roi_radius = 60.0;               // initial ROI radius
    double fallback_expand = 1.5;           // ROI expansion factor
    double roi_max = 120.0;                 // max ROI radius
    double fitness_thresh = 2.0;            // accept threshold (smaller is better)

    std::vector<double> ndt_resolutions;    // e.g. [3.0,2.0,1.0]
    std::vector<int> ndt_max_iterations;    // e.g. [40,50,60]
    double ndt_step_size = 0.7;
    double ndt_trans_eps = 1e-3;
    double ndt_score_clip = 5.0;

    bool gicp_enable = true;
    int gicp_max_iterations = 60;
    double gicp_max_corr_dist = 2.0;
    double gicp_trans_eps = 1e-4;
    double gicp_fit_eps = 1e-5;
    int gicp_corr_randomness = 30;
    bool gicp_use_recip = true;
};
BackendAlignConfig g_backend_align_cfg;

struct BackendState {
    bool is_active = false;                 // 是否已激活
    bool is_running = false;                // 线程是否在运行
    int frame_count = 0;                    // 总帧数
    int update_count = 0;                   // 更新计数
    
    // 后端优化结果
    state_ikfom global_state;               // 全局地图中的位姿
    bool has_global_state = false;          // 是否有有效全局状态
    
    // 统计
    int total_updates = 0;
    int successful_updates = 0;
} g_backend_state;

// 后端线程通信
std::mutex mtx_backend;
std::condition_variable cv_backend;
std::thread backend_thread;
bool backend_should_stop = false;

// 后端数据队列
struct BackendData {
    state_ikfom frontend_state;             // 前端状态
    PointCloudXYZI::Ptr submap;             // 提取的 submap
    PointCloudXYZI::Ptr current_scan_local; // 当前帧点云（local 坐标）
    double timestamp;                       // 时间戳
};
std::deque<BackendData> backend_queue;
const int MAX_BACKEND_QUEUE_SIZE = 5;       // 队列最大长度

// 后端 EKF 实例
esekfom::esekf<state_ikfom, 12, input_ikfom> kf_backend;

pcl::PointCloud<PointType>::Ptr g_global_map_raw(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr g_global_map_ds(new pcl::PointCloud<PointType>());


// ===== 后端可视化全局变量 =====
PointCloudXYZI::Ptr g_current_submap(new PointCloudXYZI());
PointCloudXYZI::Ptr g_current_submap_global(new PointCloudXYZI());  // 全局坐标系下的 submap
PointCloudXYZI::Ptr g_current_scan_global(new PointCloudXYZI());    // 全局坐标系下的当前帧
PointCloudXYZI::Ptr g_current_scan_local(new PointCloudXYZI());      // local 坐标系下的当前帧（后端匹配）
state_ikfom g_submap_anchor_frontend_state; // 提取当前 submap 时对应的前端 local 位姿锚点
Eigen::Isometry3d g_T_map_local = Eigen::Isometry3d::Identity(); // local -> map
std::atomic_bool  g_has_T_map_local{false};
std::mutex        g_mtx_TmapLocal; // 若你不想用 atomic，也可以只靠这个互斥
std::mutex mtx_backend_vis;  // 保护可视化数据

// 标记是否加载成功
bool g_global_map_ready = false;
bool g_first_global_align_done = false;
bool g_global_ikdtree_ready = false;
double g_backend_residual_mean = 0.0;
int g_backend_effective_points = 0;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_lidar2 = 0, last_timestamp_imu = -1.0;
double lidar_mean_scantime = 0.0, lidar_mean_scantime2 = 0.0;
int scan_num = 0, scan_num2 = 0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, lidar2_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
int lidar_type, lidar_type2;

bool multi_lidar = true;
// Extrinsic: L2->L1（用于点云坐标统一，Matrix4f 供 PCL 使用）
Eigen::Matrix4f T_L1_L2_ = Eigen::Matrix4f::Identity();
bool extrinsic_imu_to_lidars = false; // true 表示用 IMU->L1 和 IMU->L2 推导；false 直接使用 L2->L1
// publish tf 可视化（L1 wrt drone）
Eigen::Matrix4d LiDAR1_wrt_drone = Eigen::Matrix4d::Identity();
bool publish_tf_results = false;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<double>                     time_buffer2;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer2;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;
KD_TREE<PointType> ikdtree_global; // 只读全局树

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<Preprocess> p_pre2(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}


Eigen::Quaterniond ndt_init_q = Eigen::Quaterniond::Identity();
Eigen::Vector3d ndt_init_t = Eigen::Vector3d::Zero();
bool has_ndt_init_pose = false;

void ndtInitPoseCallback(const nav_msgs::OdometryConstPtr& msg){
    ndt_init_q.x() = msg->pose.pose.orientation.x;
    ndt_init_q.y() = msg->pose.pose.orientation.y;
    ndt_init_q.z() = msg->pose.pose.orientation.z;
    ndt_init_q.w() = msg->pose.pose.orientation.w;

    ndt_init_t.x() = msg->pose.pose.position.x;
    ndt_init_t.y() = msg->pose.pose.position.y;
    ndt_init_t.z() = msg->pose.pose.position.z;

    has_ndt_init_pose = true;
    ROS_WARN("[NDT Init] ===== RECEIVED INIT POSE =====");
    ROS_INFO("[NDT Init] Position: (%.3f, %.3f, %.3f)", 
        ndt_init_t.x(), ndt_init_t.y(), ndt_init_t.z());
    ROS_INFO("[NDT Init] Quaternion: (%.3f, %.3f, %.3f, %.3f)",
        ndt_init_q.x(), ndt_init_q.y(), ndt_init_q.z(), ndt_init_q.w());
}


BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    std::unique_lock<std::mutex> lk(mtx_buffer);
    scan_count++;
    double preprocess_start_time = omp_get_wtime();

    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (scan_count < MAXN) s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    lk.unlock();
    sig_buffer.notify_all();
}

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    std::unique_lock<std::mutex> lk(mtx_buffer);
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    if (!time_sync_en && fabs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && fabs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    if (scan_count < MAXN) s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    lk.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (fabs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    std::unique_lock<std::mutex> lk(mtx_buffer);

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;
    imu_buffer.push_back(msg);
    lk.unlock();
    sig_buffer.notify_all();
}

void standard_pcl_cbk2(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    std::unique_lock<std::mutex> lk(mtx_buffer);
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();

    if (msg->header.stamp.toSec() < last_timestamp_lidar2)
    {
        ROS_ERROR("lidar2 loop back, clear buffer");
        lidar_buffer2.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre2->process(msg, ptr);  // 使用第二个实例
    lidar_buffer2.push_back(ptr);
    time_buffer2.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar2 = msg->header.stamp.toSec();

    if (scan_count < MAXN) s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    lk.unlock();
    sig_buffer.notify_all();
}

void livox_pcl_cbk2(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    std::unique_lock<std::mutex> lk(mtx_buffer);
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();

    if (msg->header.stamp.toSec() < last_timestamp_lidar2)
    {
        ROS_ERROR("lidar2 loop back, clear buffer");
        lidar_buffer2.clear();
    }
    last_timestamp_lidar2 = msg->header.stamp.toSec();

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre2->process(msg, ptr);  // 使用第二个实例
    lidar_buffer2.push_back(ptr);
    time_buffer2.push_back(last_timestamp_lidar2);

    if (scan_count < MAXN) s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    lk.unlock();
    sig_buffer.notify_all();
}

// 需要：外部已构造好的 T_L1_L2_ (L2->L1 的 4x4 变换，float 或 double 都可，调用时会 cast<float>())
double l2l_max_gap = 0.05;
bool sync_packages(MeasureGroup &meas)
{
    // 单雷达
    if (!multi_lidar)
    {
        if (lidar_buffer.empty() || imu_buffer.empty()) return false;

        if (!lidar_pushed)
        {
            meas.lidar = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();

            // 统计 L1 扫描时间
            if (!meas.lidar || meas.lidar->points.size() <= 1)
            {
                if (lidar_mean_scantime <= 1e-6) lidar_mean_scantime = 0.1; // 合理的默认初值，避免 0
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
                ROS_WARN("Too few input point cloud (L1)!");
            }
            else
            {
                double scantime = meas.lidar->points.back().curvature / 1000.0; // ms->s
                if (lidar_mean_scantime <= 1e-6) lidar_mean_scantime = scantime; // 首次初始化
                if (scantime < 0.5 * lidar_mean_scantime) {
                    lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
                } else {
                    scan_num = std::max(1, scan_num + 1);
                    lidar_end_time = meas.lidar_beg_time + scantime;
                    lidar_mean_scantime += (scantime - lidar_mean_scantime) / double(scan_num);
                }
            }
            if (lidar_type == MARSIM) lidar_end_time = meas.lidar_beg_time;

            meas.lidar_end_time = lidar_end_time;
            meas.has_lidar2 = false;
            lidar_pushed = true;
        }

        if (last_timestamp_imu < lidar_end_time) return false;

        // IMU 打包
        meas.imu.clear();
        while (!imu_buffer.empty())
        {
            double imu_time = imu_buffer.front()->header.stamp.toSec();
            if (imu_time > lidar_end_time) break;
            meas.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }

        // 消费 L1
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        lidar_pushed = false;
        return true;
    }

    // 双雷达
    if (lidar_buffer.empty() || lidar_buffer2.empty() || imu_buffer.empty()) return false;

    if (!lidar_pushed)
    {
        // --- L1 ---
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();

        if (!meas.lidar || meas.lidar->points.size() <= 1)
        {
            if (lidar_mean_scantime <= 1e-6) lidar_mean_scantime = 0.1;
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud (L1)!");
        }
        else
        {
            double scantime = meas.lidar->points.back().curvature / 1000.0;
            if (lidar_mean_scantime <= 1e-6) lidar_mean_scantime = scantime;
            if (scantime < 0.5 * lidar_mean_scantime) {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            } else {
                scan_num = std::max(1, scan_num + 1);
                lidar_end_time = meas.lidar_beg_time + scantime;
                lidar_mean_scantime += (scantime - lidar_mean_scantime) / double(scan_num);
            }
        }
        if (lidar_type == MARSIM) lidar_end_time = meas.lidar_beg_time;
        meas.lidar_end_time = lidar_end_time;

        // --- L2 ---
        meas.lidar2 = lidar_buffer2.front();
        meas.lidar2_beg_time = time_buffer2.front();

        if (!meas.lidar2 || meas.lidar2->points.size() <= 1)
        {
            if (!meas.lidar2 || meas.lidar2->points.empty())
            {
                meas.has_lidar2 = false;
                lidar2_end_time = meas.lidar2_beg_time;
            }
            else
            {
                if (lidar_mean_scantime2 <= 1e-6) lidar_mean_scantime2 = 0.1;
                lidar2_end_time = meas.lidar2_beg_time + lidar_mean_scantime2;
                meas.has_lidar2 = true;
                ROS_WARN("Too few input point cloud (L2)!");
            }
        }
        else
        {
            double scantime2 = meas.lidar2->points.back().curvature / 1000.0;
            if (lidar_mean_scantime2 <= 1e-6) lidar_mean_scantime2 = scantime2;
            if (scantime2 < 0.5 * lidar_mean_scantime2) {
                lidar2_end_time = meas.lidar2_beg_time + lidar_mean_scantime2;
            } else {
                scan_num2 = std::max(1, scan_num2 + 1);
                lidar2_end_time = meas.lidar2_beg_time + scantime2;
                lidar_mean_scantime2 += (scantime2 - lidar_mean_scantime2) / double(scan_num2);
            }
            meas.has_lidar2 = true;
        }
        if (lidar_type == MARSIM) lidar2_end_time = meas.lidar2_beg_time;
        meas.lidar2_end_time = lidar2_end_time;

        // 可选：两路帧结束时间对齐阈值检查，避免错配
        if (meas.has_lidar2 && l2l_max_gap > 0.0)
        {
            if (std::fabs(lidar_end_time - lidar2_end_time) > l2l_max_gap)
            {
                ROS_WARN_STREAM("L1/L2 end time gap too large: "
                                << std::fixed << std::setprecision(6)
                                << std::fabs(lidar_end_time - lidar2_end_time)
                                << " > " << l2l_max_gap << ", skip this pair.");
                // 丢弃更旧的一路，或者简单清空并等待下一对
                // 这里选择清空两路，等待下一次对齐（策略可按需修改）
                lidar_buffer.pop_front();
                time_buffer.pop_front();
                lidar_buffer2.pop_front();
                time_buffer2.pop_front();
                lidar_pushed = false;
                return false;
            }
        }

        // 将 L2 点云变换到 L1 坐标（使用 L2->L1）
        if (meas.has_lidar2)
        {
            Eigen::Matrix4f T = T_L1_L2_.cast<float>();
            pcl::transformPointCloud(*meas.lidar2, *meas.lidar2, T);
        }

        lidar_pushed = true;
    }

    const double end_time_max = meas.has_lidar2 ? std::max(lidar_end_time, lidar2_end_time) : lidar_end_time;
    if (last_timestamp_imu < end_time_max) return false;

    // IMU 打包
    meas.imu.clear();
    while (!imu_buffer.empty())
    {
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > end_time_max) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    // 消费两路
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_buffer2.pop_front();
    time_buffer2.pop_front();

    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

bool LoadGlobalMapAndBuildIndex()
{
    if (g_loc.map_file_path.empty()) {
        ROS_WARN("[localization] map_file_path is empty, skip loading global map.");
        return false;
    }
    
    ROS_INFO("[localization] Loading global map: %s", g_loc.map_file_path.c_str());
    
    // 加载 PCD
    if (pcl::io::loadPCDFile<PointType>(g_loc.map_file_path, *g_global_map_raw) != 0) {
        ROS_ERROR("[localization] Failed to load PCD: %s", g_loc.map_file_path.c_str());
        return false;
    }
    
    ROS_INFO("[localization] Loaded %zu points from PCD", g_global_map_raw->size());
    
    // 体素降采样
    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(g_loc.map_voxel_leaf, g_loc.map_voxel_leaf, g_loc.map_voxel_leaf);
    vg.setInputCloud(g_global_map_raw);
    vg.filter(*g_global_map_ds);
    
    ROS_INFO("[localization] Downsampled to %zu points (voxel=%.2f)", 
             g_global_map_ds->size(), g_loc.map_voxel_leaf);

    // 可选统计滤波
    if (g_loc.map_remove_outliers) {
        pcl::StatisticalOutlierRemoval<PointType> sor;
        sor.setInputCloud(g_global_map_ds);
        sor.setMeanK(g_loc.map_outlier_nb_neighbors);
        sor.setStddevMulThresh(g_loc.map_outlier_stddev_mul);
        pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
        sor.filter(*filtered);
        
        size_t before = g_global_map_ds->size();
        *g_global_map_ds = *filtered;
        ROS_INFO("[localization] Statistical outlier removal: %zu -> %zu points", 
                 before, g_global_map_ds->size());
    }

    // 检查点云
    if (g_global_map_ds->empty()) {
        ROS_ERROR("[localization] Global map is empty after preprocessing.");
        return false;
    }

    // 构建 ikd-Tree（注意：ikd-Tree 使用 Build 方法，不是 setInputCloud）
    try {
        ikdtree_global.Build(g_global_map_ds->points);  // ✅ 使用 Build 方法
        g_global_ikdtree_ready = true;
        ROS_INFO("[localization] ikdtree_global built successfully. Size: %d", 
                 ikdtree_global.size());
    } catch (const std::exception& e) {
        ROS_ERROR("[localization] Failed to build ikdtree: %s", e.what());
        return false;
    }

    ROS_INFO("[localization] Global map loaded successfully:");
    ROS_INFO("  - Raw points: %zu", g_global_map_raw->size());
    ROS_INFO("  - Downsampled: %zu (leaf=%.2fm)", g_global_map_ds->size(), g_loc.map_voxel_leaf);
    ROS_INFO("  - ikdtree size: %d", ikdtree_global.size());
    
    return true;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(!scan_pub_en) return;

    PointCloudXYZI::Ptr src(dense_pub_en ? feats_undistort : feats_down_body);
    int size = src->points.size();

    // 步骤1：LiDAR/body -> local world
    PointCloudXYZI::Ptr cloud_local(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++) {
        RGBpointBodyToWorld(&src->points[i], &cloud_local->points[i]);
    }

    // 步骤2：local -> map（如有）
    PointCloudXYZI::Ptr cloud_map = cloud_local;
    bool has_map = g_has_T_map_local.load(std::memory_order_acquire);
    if (has_map) {
        std::lock_guard<std::mutex> lk(g_mtx_TmapLocal);
        cloud_map.reset(new PointCloudXYZI());
        pcl::transformPointCloud(*cloud_local, *cloud_map, g_T_map_local.matrix().cast<float>());
    }

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud_map, msg);
    msg.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg.header.frame_id = "map";
    pubLaserCloudFull.publish(msg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "map";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "map";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    // 1) 组装 local 下的 IMU位姿
    Eigen::Isometry3d T_local_I = Eigen::Isometry3d::Identity();
    T_local_I.linear()      = state_point.rot.toRotationMatrix();
    T_local_I.translation() = state_point.pos;

    // 2) 如有 T_map_local，则左乘到 map
    Eigen::Isometry3d T_map_I = T_local_I;
    bool has_map = g_has_T_map_local.load(std::memory_order_acquire);
    if (has_map) {
        std::lock_guard<std::mutex> lk(g_mtx_TmapLocal);
        T_map_I = g_T_map_local * T_local_I;
    }

    // 3) 发布 odom（map 框架）
    odomAftMapped.header.frame_id = "map";
    odomAftMapped.child_frame_id  = "body";
    odomAftMapped.header.stamp    = ros::Time().fromSec(lidar_end_time);

    Eigen::Quaterniond q(T_map_I.rotation());
    Eigen::Vector3d    t = T_map_I.translation();

    odomAftMapped.pose.pose.position.x = t.x();
    odomAftMapped.pose.pose.position.y = t.y();
    odomAftMapped.pose.pose.position.z = t.z();
    odomAftMapped.pose.pose.orientation.x = q.x();
    odomAftMapped.pose.pose.orientation.y = q.y();
    odomAftMapped.pose.pose.orientation.z = q.z();
    odomAftMapped.pose.pose.orientation.w = q.w();

    pubOdomAftMapped.publish(odomAftMapped);

    // 协方差保持原逻辑
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    // 4) TF: map -> body
    static tf::TransformBroadcaster br;
    tf::Transform T_tf;
    tf::Quaternion q_tf(q.x(), q.y(), q.z(), q.w());
    T_tf.setOrigin(tf::Vector3(t.x(), t.y(), t.z()));
    T_tf.setRotation(q_tf);
    br.sendTransform(tf::StampedTransform(T_tf, odomAftMapped.header.stamp, "map", "body"));
}

void publish_path(const ros::Publisher& pubPath)
{
    // 1) 组装 local 下的 IMU位姿 T_local_I
    Eigen::Isometry3d T_local_I = Eigen::Isometry3d::Identity();
    T_local_I.linear()      = state_point.rot.toRotationMatrix();
    T_local_I.translation() = state_point.pos;

    // 2) 如有 T_map_local，则左乘得到 map 下位姿 T_map_I
    Eigen::Isometry3d T_map_I = T_local_I;
    const bool has_map = g_has_T_map_local.load(std::memory_order_acquire);
    if (has_map) {
        std::lock_guard<std::mutex> lk(g_mtx_TmapLocal);
        T_map_I = g_T_map_local * T_local_I; // local -> map
    }

    // 3) 写入 PoseStamped
    const Eigen::Vector3d t = T_map_I.translation();
    const Eigen::Quaterniond q(T_map_I.rotation());

    msg_body_pose.pose.position.x = t.x();
    msg_body_pose.pose.position.y = t.y();
    msg_body_pose.pose.position.z = t.z();
    msg_body_pose.pose.orientation.x = q.x();
    msg_body_pose.pose.orientation.y = q.y();
    msg_body_pose.pose.orientation.z = q.z();
    msg_body_pose.pose.orientation.w = q.w();

    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "map";

    // 4) 累计并发布 Path（避免太密）
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) {
        path.header.frame_id = "map";
        path.header.stamp = msg_body_pose.header.stamp;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void publish_global_map(const ros::Publisher& pub)
{
    if (!g_global_map_ready || g_global_map_ds->empty()) return;
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*g_global_map_ds, msg);
    msg.header.frame_id = "map";
    msg.header.stamp = ros::Time::now();
    pub.publish(msg);
}


// ===== 检查是否应该激活后端 =====
// ===== 检查是否应该激活后端 =====
bool shouldActivateBackend()
{
    if (g_backend_state.is_active) return true;
    if (!g_backend_cfg.enable) return false;
    
    // 帧数检查
    if (g_backend_state.frame_count < g_backend_cfg.min_frames) {
        return false;
    }
    
    // 点数检查
    int local_points = ikdtree.size();
    if (local_points < g_backend_cfg.min_local_points) {
        ROS_INFO_THROTTLE(5.0, "[Backend] Waiting for local map: %d / %d points", 
                         local_points, g_backend_cfg.min_local_points);
        return false;
    }
    
    // 全局地图检查
    if (!g_global_ikdtree_ready) {
        ROS_WARN_THROTTLE(5.0, "[Backend] Global map not ready!");
        return false;
    }
    
    // ===== 新增：initpose 检查 =====
    if (!has_ndt_init_pose) {
        ROS_WARN_THROTTLE(5.0, "[Backend] Waiting for /initpose to activate backend...");
        return false;
    }
    
    g_backend_state.is_active = true;
    ROS_WARN("[Backend] ==== BACKEND ACTIVATED ==== Local points: %d, Frames: %d",
             local_points, g_backend_state.frame_count);
    
    return true;
}

// ===== 从 local ikdtree 提取 submap =====
// ===== 从 local ikdtree 提取 submap =====
bool extractSubmapFromLocal(
    KD_TREE<PointType>& local_tree,
    const V3D& center_pos,
    float radius,
    PointCloudXYZI::Ptr& submap_out)
{
    if (local_tree.Root_Node == nullptr) {
        ROS_WARN_THROTTLE(1.0, "[Submap] Local tree is empty!");
        return false;
    }
    
    // 构造搜索框
    BoxPointType search_box;
    for (int i = 0; i < 3; i++) {
        search_box.vertex_min[i] = center_pos(i) - radius;
        search_box.vertex_max[i] = center_pos(i) + radius;
    }
    
    // 从 ikdtree 搜索
    PointVector points_in_box;
    local_tree.Box_Search(search_box, points_in_box);
    
    if (points_in_box.size() < 100) {
        ROS_WARN_THROTTLE(1.0, "[Submap] Too few points in box: %zu", points_in_box.size());
        return false;
    }
    
    // 转为点云
    submap_out->clear();
    submap_out->points = points_in_box;
    submap_out->width = points_in_box.size();
    submap_out->height = 1;
    submap_out->is_dense = true;
    
    // 降采样（可选，基于配置）
    if (g_backend_cfg.submap_voxel_size > 0.01f && submap_out->size() > 10000) {
        pcl::VoxelGrid<PointType> vg;
        vg.setLeafSize(g_backend_cfg.submap_voxel_size, 
                       g_backend_cfg.submap_voxel_size, 
                       g_backend_cfg.submap_voxel_size);
        vg.setInputCloud(submap_out);
        PointCloudXYZI::Ptr filtered(new PointCloudXYZI());
        vg.filter(*filtered);
        submap_out = filtered;
    }
    
    ROS_INFO_THROTTLE(2.0, "[Submap] Extracted %zu points (radius=%.1fm, center=[%.1f, %.1f, %.1f])", 
                     submap_out->size(), radius, center_pos.x(), center_pos.y(), center_pos.z());
    return true;
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

// ===== 后端观测模型：Submap 对 Global Map =====
void h_share_model_submap(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    if (!g_global_ikdtree_ready ||
        ((!g_current_submap || g_current_submap->empty()) &&
         (!g_current_scan_local || g_current_scan_local->empty()))) {
        ekfom_data.valid = false;
        ROS_WARN_THROTTLE(1.0, "[Backend] Global map or backend submap not ready!");
        return;
    }

    const double t_match_begin = omp_get_wtime();

    static PointCloudXYZI::Ptr submap_body(new PointCloudXYZI());
    static PointCloudXYZI::Ptr submap_world(new PointCloudXYZI());
    static PointCloudXYZI::Ptr scan_body(new PointCloudXYZI());
    static PointCloudXYZI::Ptr scan_world(new PointCloudXYZI());
    static PointCloudXYZI::Ptr local_norm_submap(new PointCloudXYZI());
    static PointCloudXYZI::Ptr local_norm_scan(new PointCloudXYZI());
    static PointCloudXYZI::Ptr global_norm_submap(new PointCloudXYZI());
    static PointCloudXYZI::Ptr global_norm_scan(new PointCloudXYZI());
    static PointCloudXYZI::Ptr laserCloudOri_local(new PointCloudXYZI(100000, 1));
    static PointCloudXYZI::Ptr corr_normvect_local(new PointCloudXYZI(100000, 1));
    static PointCloudXYZI::Ptr laserCloudOri_global(new PointCloudXYZI(100000, 1));
    static PointCloudXYZI::Ptr corr_normvect_global(new PointCloudXYZI(100000, 1));
    static KD_TREE<PointType> submap_tree;

    static std::vector<bool> selected_local_submap;
    static std::vector<bool> selected_global_submap;
    static std::vector<bool> selected_local_scan;
    static std::vector<bool> selected_global_scan;
    static std::vector<float> res_local_submap;
    static std::vector<float> res_global_submap;
    static std::vector<float> res_local_scan;
    static std::vector<float> res_global_scan;

    submap_body->clear();
    submap_world->clear();
    scan_body->clear();
    scan_world->clear();
    local_norm_submap->clear();
    local_norm_scan->clear();
    global_norm_submap->clear();
    global_norm_scan->clear();
    laserCloudOri_local->clear();
    corr_normvect_local->clear();
    laserCloudOri_global->clear();
    corr_normvect_global->clear();

    const int submap_size = g_current_submap ? (int)g_current_submap->size() : 0;
    const int scan_size = (g_current_scan_local && !g_current_scan_local->empty())
        ? (int)g_current_scan_local->size() : 0;

    if (submap_size == 0 && scan_size == 0) {
        ekfom_data.valid = false;
        ROS_WARN_THROTTLE(1.0, "[Backend] No submap or scan for backend match!");
        return;
    }

    if (submap_size > 0) {
        submap_body->resize(submap_size);
        submap_world->resize(submap_size);
        local_norm_submap->resize(submap_size);
        global_norm_submap->resize(submap_size);
        if ((int)selected_local_submap.size() < submap_size) selected_local_submap.resize(submap_size, false);
        if ((int)selected_global_submap.size() < submap_size) selected_global_submap.resize(submap_size, false);
        if ((int)res_local_submap.size() < submap_size) res_local_submap.resize(submap_size, 0.0f);
        if ((int)res_global_submap.size() < submap_size) res_global_submap.resize(submap_size, 0.0f);
        std::fill(selected_local_submap.begin(), selected_local_submap.begin() + submap_size, false);
        std::fill(selected_global_submap.begin(), selected_global_submap.begin() + submap_size, false);
    }

    if (scan_size > 0) {
        scan_body->resize(scan_size);
        scan_world->resize(scan_size);
        local_norm_scan->resize(scan_size);
        global_norm_scan->resize(scan_size);
        if ((int)selected_local_scan.size() < scan_size) selected_local_scan.resize(scan_size, false);
        if ((int)selected_global_scan.size() < scan_size) selected_global_scan.resize(scan_size, false);
        if ((int)res_local_scan.size() < scan_size) res_local_scan.resize(scan_size, 0.0f);
        if ((int)res_global_scan.size() < scan_size) res_global_scan.resize(scan_size, 0.0f);
        std::fill(selected_local_scan.begin(), selected_local_scan.begin() + scan_size, false);
        std::fill(selected_global_scan.begin(), selected_global_scan.begin() + scan_size, false);
    }

    const Eigen::Matrix3d R_local_imu_anchor = g_submap_anchor_frontend_state.rot.toRotationMatrix();
    const Eigen::Vector3d t_local_imu_anchor = g_submap_anchor_frontend_state.pos;

    const float dist_th_local = 5.0f;
    const float dist_th_global = 12.0f;
    const float plane_eps = 0.1f;
    const float s_gate = 0.7f;
    const float r_abs_max_global = (float)g_loc.gpr_max_point2plane_dist;
    const double alpha_global = 3.0;
    const double w_global_base = 2.0;
    const double huber_delta = g_loc.gpr_huber_delta;
    const bool use_global_residual = g_loc.gpr_enable && g_global_ikdtree_ready;
    const int global_max_keep = (g_loc.gpr_max_points > 0) ? g_loc.gpr_max_points : 4000;
    const float tau_global_better = 0.85f;

    static int s_boost_frames = 0;
    double boost_gain = (s_boost_frames > 0) ? 2.0 : 1.0;

    static bool keep_clearing_local_tree = false;
    const double good_thresh = std::max(0.15, 0.35 * (double)g_loc.gpr_max_point2plane_dist);
    const double bad_thresh  = std::max(0.2 , 0.60 * (double)g_loc.gpr_max_point2plane_dist);

    auto huber_weight = [&](double r_abs, double delta)->double {
        if (delta <= 0) return 1.0;
        return (r_abs <= delta) ? 1.0 : (delta / r_abs);
    };

    if (submap_size > 0) {
        for (int i = 0; i < submap_size; i++) {
            const PointType &pt_local = g_current_submap->points[i];
            V3D p_local(pt_local.x, pt_local.y, pt_local.z);
            V3D p_imu_anchor = R_local_imu_anchor.transpose() * (p_local - t_local_imu_anchor);

            PointType pt_body;
            pt_body.x = p_imu_anchor(0);
            pt_body.y = p_imu_anchor(1);
            pt_body.z = p_imu_anchor(2);
            pt_body.intensity = pt_local.intensity;
            submap_body->points[i] = pt_body;

            V3D p_global_backend = s.rot * p_imu_anchor + s.pos;
            PointType pt_world_backend;
            pt_world_backend.x = p_global_backend(0);
            pt_world_backend.y = p_global_backend(1);
            pt_world_backend.z = p_global_backend(2);
            pt_world_backend.intensity = pt_local.intensity;
            submap_world->points[i] = pt_world_backend;
        }
        submap_tree.Build(submap_world->points);
    }

    double sum_rg_kept = 0.0;
    int rg_kept_cnt = 0;

    if (submap_size > 0) {
        #ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
            #pragma omp parallel for
        #endif
        for (int i = 0; i < submap_size; i++)
        {
            const PointType &point_body = submap_body->points[i];
            const PointType &point_world = submap_world->points[i];
            const V3D p_body(point_body.x, point_body.y, point_body.z);

            if (submap_tree.Root_Node != nullptr) {
                std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
                KD_TREE<PointType>::PointVector points_near_local;
                submap_tree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near_local, pointSearchSqDis);

                const bool ok_local =
                    (points_near_local.size() >= NUM_MATCH_POINTS) &&
                    (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= dist_th_local);

                if (ok_local)
                {
                    VF(4) pabcd;
                    if (esti_plane(pabcd, points_near_local, plane_eps))
                    {
                        const float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                        const float body_norm = std::max<float>(1e-3f, (float)p_body.norm());
                        const float s_val = 1.0f - 0.9f * std::fabs(pd2) / std::sqrt(body_norm);

                        if (s_val > s_gate)
                        {
                            selected_local_submap[i] = true;
                            local_norm_submap->points[i].x = pabcd(0);
                            local_norm_submap->points[i].y = pabcd(1);
                            local_norm_submap->points[i].z = pabcd(2);
                            local_norm_submap->points[i].intensity = pd2;
                            res_local_submap[i] = std::fabs(pd2);
                        }
                    }
                }
            }

            if (use_global_residual)
            {
                KD_TREE<PointType>::PointVector points_near_global;
                points_near_global.reserve(NUM_MATCH_POINTS);
                std::vector<float> pointSearchSqDisG(NUM_MATCH_POINTS);

                ikdtree_global.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near_global, pointSearchSqDisG);
                const bool ok_global =
                    (points_near_global.size() >= NUM_MATCH_POINTS) &&
                    (pointSearchSqDisG[NUM_MATCH_POINTS - 1] <= dist_th_global);

                if (ok_global)
                {
                    VF(4) pabcd_g;
                    if (esti_plane(pabcd_g, points_near_global, plane_eps))
                    {
                        const float pd2g = pabcd_g(0) * point_world.x + pabcd_g(1) * point_world.y + pabcd_g(2) * point_world.z + pabcd_g(3);
                        const float body_norm = std::max<float>(1e-3f, (float)p_body.norm());
                        const float s_val_g = 1.0f - 0.9f * std::fabs(pd2g) / std::sqrt(body_norm);

                        if (s_val_g > s_gate && std::fabs(pd2g) <= r_abs_max_global)
                        {
                            selected_global_submap[i] = true;
                            global_norm_submap->points[i].x = pabcd_g(0);
                            global_norm_submap->points[i].y = pabcd_g(1);
                            global_norm_submap->points[i].z = pabcd_g(2);
                            global_norm_submap->points[i].intensity = pd2g;
                            res_global_submap[i] = std::fabs(pd2g);
                        }
                    }
                }
            }
        }
    }

    if (scan_size > 0) {
        #ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
            #pragma omp parallel for
        #endif
        for (int i = 0; i < scan_size; i++)
        {
            const PointType &pt_local = g_current_scan_local->points[i];
            V3D p_local(pt_local.x, pt_local.y, pt_local.z);
            V3D p_imu_anchor = R_local_imu_anchor.transpose() * (p_local - t_local_imu_anchor);

            PointType pt_body;
            pt_body.x = p_imu_anchor(0);
            pt_body.y = p_imu_anchor(1);
            pt_body.z = p_imu_anchor(2);
            pt_body.intensity = pt_local.intensity;
            scan_body->points[i] = pt_body;

            V3D p_global_backend = s.rot * p_imu_anchor + s.pos;
            PointType pt_world_backend;
            pt_world_backend.x = p_global_backend(0);
            pt_world_backend.y = p_global_backend(1);
            pt_world_backend.z = p_global_backend(2);
            pt_world_backend.intensity = pt_local.intensity;
            scan_world->points[i] = pt_world_backend;

            if (submap_tree.Root_Node != nullptr) {
                std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
                KD_TREE<PointType>::PointVector points_near_local;
                submap_tree.Nearest_Search(pt_world_backend, NUM_MATCH_POINTS, points_near_local, pointSearchSqDis);

                const bool ok_local =
                    (points_near_local.size() >= NUM_MATCH_POINTS) &&
                    (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= dist_th_local);

                if (ok_local)
                {
                    VF(4) pabcd;
                    if (esti_plane(pabcd, points_near_local, plane_eps))
                    {
                        const float pd2 = pabcd(0) * pt_world_backend.x + pabcd(1) * pt_world_backend.y + pabcd(2) * pt_world_backend.z + pabcd(3);
                        const float body_norm = std::max<float>(1e-3f, (float)p_imu_anchor.norm());
                        const float s_val = 1.0f - 0.9f * std::fabs(pd2) / std::sqrt(body_norm);

                        if (s_val > s_gate)
                        {
                            selected_local_scan[i] = true;
                            local_norm_scan->points[i].x = pabcd(0);
                            local_norm_scan->points[i].y = pabcd(1);
                            local_norm_scan->points[i].z = pabcd(2);
                            local_norm_scan->points[i].intensity = pd2;
                            res_local_scan[i] = std::fabs(pd2);
                        }
                    }
                }
            }

            if (use_global_residual)
            {
                KD_TREE<PointType>::PointVector points_near_global;
                points_near_global.reserve(NUM_MATCH_POINTS);
                std::vector<float> pointSearchSqDisG(NUM_MATCH_POINTS);

                ikdtree_global.Nearest_Search(pt_world_backend, NUM_MATCH_POINTS, points_near_global, pointSearchSqDisG);
                const bool ok_global =
                    (points_near_global.size() >= NUM_MATCH_POINTS) &&
                    (pointSearchSqDisG[NUM_MATCH_POINTS - 1] <= dist_th_global);

                if (ok_global)
                {
                    VF(4) pabcd_g;
                    if (esti_plane(pabcd_g, points_near_global, plane_eps))
                    {
                        const float pd2g = pabcd_g(0) * pt_world_backend.x + pabcd_g(1) * pt_world_backend.y + pabcd_g(2) * pt_world_backend.z + pabcd_g(3);
                        const float body_norm = std::max<float>(1e-3f, (float)p_imu_anchor.norm());
                        const float s_val_g = 1.0f - 0.9f * std::fabs(pd2g) / std::sqrt(body_norm);

                        if (s_val_g > s_gate && std::fabs(pd2g) <= r_abs_max_global)
                        {
                            selected_global_scan[i] = true;
                            global_norm_scan->points[i].x = pabcd_g(0);
                            global_norm_scan->points[i].y = pabcd_g(1);
                            global_norm_scan->points[i].z = pabcd_g(2);
                            global_norm_scan->points[i].intensity = pd2g;
                            res_global_scan[i] = std::fabs(pd2g);
                        }
                    }
                }
            }
        }
    }

    if (use_global_residual)
    {
        int replaced_cnt = 0;
        double sum_rg = 0.0, sum_rl = 0.0;

        for (int i = 0; i < submap_size; i++)
        {
            if (!selected_global_submap[i]) continue;
            const float r_g = res_global_submap[i];
            if (selected_local_submap[i])
            {
                const float r_l = res_local_submap[i];
                if (r_g <= tau_global_better * r_l)
                {
                    selected_local_submap[i] = false;
                    replaced_cnt++;
                    sum_rg += r_g;
                    sum_rl += r_l;
                }
            }
        }

        for (int i = 0; i < scan_size; i++)
        {
            if (!selected_global_scan[i]) continue;
            const float r_g = res_global_scan[i];
            if (selected_local_scan[i])
            {
                const float r_l = res_local_scan[i];
                if (r_g <= tau_global_better * r_l)
                {
                    selected_local_scan[i] = false;
                    replaced_cnt++;
                    sum_rg += r_g;
                    sum_rl += r_l;
                }
            }
        }

        if (replaced_cnt > 0)
        {
            ROS_INFO_THROTTLE(1.0, "[G>L] replaced %d pts, mean r_g=%.3f, r_l=%.3f",
                              replaced_cnt,
                              (float)(sum_rg / std::max(1, replaced_cnt)),
                              (float)(sum_rl / std::max(1, replaced_cnt)));
        }
    }

    int effct_feat_num_local = 0;
    int effct_feat_num_global = 0;
    int global_kept = 0;
    total_residual = 0.0;

    for (int i = 0; i < submap_size; i++)
    {
        if (selected_local_submap[i])
        {
            laserCloudOri_local->points[effct_feat_num_local] = submap_body->points[i];
            corr_normvect_local->points[effct_feat_num_local] = local_norm_submap->points[i];
            total_residual += res_local_submap[i];
            effct_feat_num_local++;
        }
    }

    for (int i = 0; i < scan_size; i++)
    {
        if (selected_local_scan[i])
        {
            laserCloudOri_local->points[effct_feat_num_local] = scan_body->points[i];
            corr_normvect_local->points[effct_feat_num_local] = local_norm_scan->points[i];
            total_residual += res_local_scan[i];
            effct_feat_num_local++;
        }
    }

    if (use_global_residual)
    {
        for (int i = 0; i < submap_size; i++)
        {
            if (!selected_global_submap[i]) continue;
            if (global_kept >= global_max_keep) break;

            laserCloudOri_global->points[effct_feat_num_global] = submap_body->points[i];
            corr_normvect_global->points[effct_feat_num_global] = global_norm_submap->points[i];
            total_residual += res_global_submap[i];
            sum_rg_kept += res_global_submap[i];
            rg_kept_cnt++;
            effct_feat_num_global++;
            global_kept++;
        }

        for (int i = 0; i < scan_size; i++)
        {
            if (!selected_global_scan[i]) continue;
            if (global_kept >= global_max_keep) break;

            laserCloudOri_global->points[effct_feat_num_global] = scan_body->points[i];
            corr_normvect_global->points[effct_feat_num_global] = global_norm_scan->points[i];
            total_residual += res_global_scan[i];
            sum_rg_kept += res_global_scan[i];
            rg_kept_cnt++;
            effct_feat_num_global++;
            global_kept++;
        }
    }

    const int total_eff = effct_feat_num_local + effct_feat_num_global;
    g_backend_effective_points = total_eff;

    if (total_eff < 1)
    {
        ekfom_data.valid = false;
        g_backend_residual_mean = 0.0;
        ROS_WARN_THROTTLE(0.2, "No Effective Points! (local=%d, global=%d)", effct_feat_num_local, effct_feat_num_global);
        return;
    }

    if (use_global_residual && rg_kept_cnt > 0)
    {
        const double mean_rg_kept = sum_rg_kept / (double)rg_kept_cnt;

        if (mean_rg_kept <= good_thresh) {
            if (keep_clearing_local_tree) {
                ROS_INFO_THROTTLE(0.5, "[LOCAL TREE] global residual good (%.3f <= %.3f), stop clearing local tree.",
                                  (float)mean_rg_kept, (float)good_thresh);
            }
            keep_clearing_local_tree = false;
        } else if (mean_rg_kept > bad_thresh) {
            if (!keep_clearing_local_tree) {
                ROS_WARN_THROTTLE(0.5, "[LOCAL TREE] global residual bad (%.3f > %.3f), start clearing local tree.",
                                  (float)mean_rg_kept, (float)bad_thresh);
            }
            keep_clearing_local_tree = true;
        }
    }

    const int total_eff_safe = std::max(1, total_eff);
    g_backend_residual_mean = total_residual / total_eff_safe;

    const double t_solve_begin = omp_get_wtime();
    ekfom_data.h_x = MatrixXd::Zero(total_eff, 12);
    ekfom_data.h.resize(total_eff);

    for (int r = 0; r < effct_feat_num_local; r++)
    {
        const PointType &laser_p  = laserCloudOri_local->points[r];
        const PointType &norm_p   = corr_normvect_local->points[r];

        const V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat; point_be_crossmat << SKEW_SYM_MATRX(point_this_be);

        const V3D point_this = point_this_be;
        M3D point_crossmat; point_crossmat << SKEW_SYM_MATRX(point_this);

        V3D n_world(norm_p.x, norm_p.y, norm_p.z);
        const double pd2 = norm_p.intensity;
        const double w_robust = huber_weight(std::fabs(pd2), huber_delta);

        n_world *= w_robust;

        const V3D C(s.rot.conjugate() * n_world);
        const V3D A(point_crossmat * C);

        ekfom_data.h_x.block<1, 12>(r, 0) << n_world(0), n_world(1), n_world(2), VEC_FROM_ARRAY(A),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        ekfom_data.h(r) = -pd2 * w_robust;
    }

    const double w_global_full = std::max(1.0, alpha_global * w_global_base * boost_gain);
    for (int j = 0; j < effct_feat_num_global; j++)
    {
        const int r = effct_feat_num_local + j;

        const PointType &laser_p  = laserCloudOri_global->points[j];
        const PointType &norm_p_g = corr_normvect_global->points[j];

        const V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat; point_be_crossmat << SKEW_SYM_MATRX(point_this_be);

        const V3D point_this = point_this_be;
        M3D point_crossmat; point_crossmat << SKEW_SYM_MATRX(point_this);

        V3D n_world_g(norm_p_g.x, norm_p_g.y, norm_p_g.z);
        const double pd2_g_raw = norm_p_g.intensity;
        const double w_robust_g = huber_weight(std::fabs(pd2_g_raw), huber_delta);

        const double w_all = w_global_full * w_robust_g;
        n_world_g *= w_all;
        const double pd2_g = pd2_g_raw * w_all;

        const V3D C(s.rot.conjugate() * n_world_g);
        const V3D A(point_crossmat * C);

        ekfom_data.h_x.block<1, 12>(r, 0) << n_world_g(0), n_world_g(1), n_world_g(2), VEC_FROM_ARRAY(A),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        ekfom_data.h(r) = -pd2_g;
    }

    solve_time += omp_get_wtime() - t_solve_begin;

    if (keep_clearing_local_tree)
    {
        using TreeT = KD_TREE<PointType>;
        typename TreeT::PointVector new_pts;
        new_pts.clear();

        if (scan_size > 0) {
            for (int i = 0; i < scan_size; ++i)
            {
                new_pts.push_back(scan_world->points[i]);
            }
        } else {
            for (int i = 0; i < submap_size; ++i)
            {
                new_pts.push_back(submap_world->points[i]);
            }
        }

        submap_tree.Build(new_pts);
        s_boost_frames = std::max(s_boost_frames, 8);

        ROS_WARN_THROTTLE(0.5, "[LOCAL TREE] cleared & rebuilt with current frame (%ld pts).", new_pts.size());
    }

    if (s_boost_frames > 0) s_boost_frames--;

    match_time += omp_get_wtime() - t_match_begin;
}

// ===== 将点云从 local 坐标系变换到 global 坐标系 =====
void transformPointCloudToGlobal(
    const PointCloudXYZI::Ptr& cloud_local,
    PointCloudXYZI::Ptr& cloud_global,
    const Eigen::Matrix4d& T_map_local)
{
    cloud_global->clear();
    cloud_global->reserve(cloud_local->size());
    
    Eigen::Matrix3d R = T_map_local.topLeftCorner<3,3>();
    Eigen::Vector3d t = T_map_local.block<3,1>(0,3);
    
    for (const auto& pt : cloud_local->points) {
        Eigen::Vector3d p_local(pt.x, pt.y, pt.z);
        Eigen::Vector3d p_global = R * p_local + t;
        
        PointType pt_global;
        pt_global.x = p_global.x();
        pt_global.y = p_global.y();
        pt_global.z = p_global.z();
        pt_global.intensity = pt.intensity;
        
        cloud_global->push_back(pt_global);
    }
}

static Eigen::Matrix4d ComputeMapImuFromMapLocal(
    const Eigen::Matrix4d& T_map_local,
    const state_ikfom& local_state)
{
    Eigen::Matrix4d T_local_imu = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q_local = local_state.rot;
    q_local.normalize();
    T_local_imu.topLeftCorner<3,3>() = q_local.toRotationMatrix();
    T_local_imu.block<3,1>(0,3) = local_state.pos;
    return T_map_local * T_local_imu;
}

// ===== 根据后端状态计算 T_map_local =====
static Eigen::Matrix3d OrthonormalizeRotation(const Eigen::Matrix3d &R)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R_ortho = U * V.transpose();
    if (R_ortho.determinant() < 0.0) {
        U.col(2) *= -1.0;
        R_ortho = U * V.transpose();
    }
    return R_ortho;
}

Eigen::Matrix4d computeTransformMapLocal(
    const state_ikfom& global_state,
    const state_ikfom& local_state)
{
    // global_state: 后端优化的全局位姿 (IMU in map frame)
    // local_state: 前端位姿 (IMU in local frame)
    
    // T_map_imu_global = [R_map_imu, t_map_imu]
    Eigen::Quaterniond q_map = global_state.rot;
    q_map.normalize();
    Eigen::Matrix3d R_map_imu = q_map.toRotationMatrix();
    Eigen::Vector3d t_map_imu = global_state.pos;
    
    // T_local_imu_local = [R_local_imu, t_local_imu]
    Eigen::Quaterniond q_local = local_state.rot;
    q_local.normalize();
    Eigen::Matrix3d R_local_imu = q_local.toRotationMatrix();
    Eigen::Vector3d t_local_imu = local_state.pos;
    
    // T_map_local = T_map_imu_global * T_imu_local^{-1}
    //             = T_map_imu_global * (T_local_imu)^{-1}
    Eigen::Matrix4d T_local_imu = Eigen::Matrix4d::Identity();
    T_local_imu.topLeftCorner<3,3>() = R_local_imu;
    T_local_imu.block<3,1>(0,3) = t_local_imu;
    
    Eigen::Matrix4d T_map_imu = Eigen::Matrix4d::Identity();
    T_map_imu.topLeftCorner<3,3>() = R_map_imu;
    T_map_imu.block<3,1>(0,3) = t_map_imu;
    
    Eigen::Matrix4d T_map_local = T_map_imu * T_local_imu.inverse();
    
    return T_map_local;
}

struct AlignResult {
    bool ok = false;
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    double score = 1e9;
    std::string method = "none";
};

static pcl::PointCloud<PointType>::Ptr CropROI(
    const pcl::PointCloud<PointType>::Ptr& cloud,
    const Eigen::Vector3f& center,
    float radius)
{
    pcl::PointCloud<PointType>::Ptr out(new pcl::PointCloud<PointType>());
    if (!cloud) return out;
    out->reserve(10000);
    float r2 = radius * radius;
    for (const auto& p : cloud->points) {
        float dx = p.x - center.x();
        float dy = p.y - center.y();
        float dz = p.z - center.z();
        if (dx*dx + dy*dy + dz*dz <= r2) {
            out->push_back(p);
        }
    }
    out->width = out->points.size();
    out->height = 1;
    out->is_dense = true;
    return out;
}

static pcl::PointCloud<PointType>::Ptr VoxelDown(
    const pcl::PointCloud<PointType>::Ptr& cloud,
    float leaf)
{
    pcl::PointCloud<PointType>::Ptr ds(new pcl::PointCloud<PointType>());
    if (!cloud || cloud->empty()) return ds;
    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(leaf, leaf, leaf);
    vg.setInputCloud(cloud);
    vg.filter(*ds);
    return ds;
}

static AlignResult RunNDTGICPOnce(
    const pcl::PointCloud<PointType>::Ptr& source_local,
    const Eigen::Matrix4f& T_guess_map_local,
    const Eigen::Vector3f& guess_center_map)
{
    AlignResult res;
    if (!g_global_map_ready || !g_global_map_ds || g_global_map_ds->empty()) {
        ROS_WARN_THROTTLE(1.0, "[Backend Align] Global map empty.");
        return res;
    }
    if (!source_local || source_local->empty()) {
        ROS_WARN_THROTTLE(1.0, "[Backend Align] Source cloud empty.");
        return res;
    }

    float rr = static_cast<float>(g_backend_align_cfg.roi_radius);
    pcl::PointCloud<PointType>::Ptr target_roi = CropROI(g_global_map_ds, guess_center_map, rr);
    if (target_roi->size() < 500 && g_backend_align_cfg.fallback_expand > 1.0) {
        rr = std::min<float>(rr * g_backend_align_cfg.fallback_expand,
                             static_cast<float>(g_backend_align_cfg.roi_max));
        target_roi = CropROI(g_global_map_ds, guess_center_map, rr);
    }
    if (target_roi->size() < 500) {
        ROS_WARN_THROTTLE(1.0, "[Backend Align] ROI too small (%zu)", target_roi->size());
        return res;
    }

    const float leaf = std::max(0.4f, g_backend_cfg.submap_voxel_size);
    auto src_ds = VoxelDown(source_local, leaf);
    auto tgt_ds = VoxelDown(target_roi, leaf);
    if (src_ds->empty() || tgt_ds->empty()) {
        ROS_WARN_THROTTLE(1.0, "[Backend Align] Downsampled cloud empty.");
        return res;
    }

    Eigen::Matrix4f T = T_guess_map_local;
    if (!T.allFinite()) {
        T = Eigen::Matrix4f::Identity();
    }

    bool ndt_ok = false;
    double best_score = std::numeric_limits<double>::infinity();
    std::vector<double> ndt_res = g_backend_align_cfg.ndt_resolutions;
    std::vector<int> ndt_its = g_backend_align_cfg.ndt_max_iterations;
    if (ndt_res.empty() || ndt_its.empty() || ndt_res.size() != ndt_its.size()) {
        ndt_res = {2.0};
        ndt_its = {30};
    }
    for (size_t i = 0; i < ndt_res.size(); ++i) {
            const double reso = ndt_res[i];
            const int max_it = ndt_its[i];
            pcl::NormalDistributionsTransform<PointType, PointType> ndt;
            ndt.setResolution(reso);
            ndt.setStepSize(g_backend_align_cfg.ndt_step_size);
            ndt.setTransformationEpsilon(g_backend_align_cfg.ndt_trans_eps);
            ndt.setMaximumIterations(max_it);
            ndt.setInputTarget(tgt_ds);
            ndt.setInputSource(src_ds);

            pcl::PointCloud<PointType> aligned;
            try {
                ndt.align(aligned, T);
            } catch (const std::exception& e) {
                ROS_WARN_THROTTLE(1.0, "[Backend Align][NDT] exception: %s", e.what());
                break;
            }
            if (!ndt.hasConverged()) {
                continue;
            }
            Eigen::Matrix4f T_new = ndt.getFinalTransformation();
            if (!T_new.allFinite()) {
                break;
            }
            T = T_new;
            best_score = ndt.getFitnessScore();
            ndt_ok = true;
    }

    if (ndt_ok && (best_score <= g_backend_align_cfg.ndt_score_clip)) {
        res.ok = true;
        res.T = T;
        res.score = best_score;
        res.method = "ndt";
        return res;
    }

    if (g_backend_align_cfg.gicp_enable) {
        pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
        gicp.setMaximumIterations(g_backend_align_cfg.gicp_max_iterations);
        gicp.setTransformationEpsilon(g_backend_align_cfg.gicp_trans_eps);
        gicp.setEuclideanFitnessEpsilon(g_backend_align_cfg.gicp_fit_eps);
        gicp.setMaxCorrespondenceDistance(g_backend_align_cfg.gicp_max_corr_dist);
        gicp.setCorrespondenceRandomness(g_backend_align_cfg.gicp_corr_randomness);
#ifdef PCL_VERSION_COMPARE
#if PCL_VERSION_COMPARE(>=, 1, 10, 0)
    gicp.setUseReciprocalCorrespondences(g_backend_align_cfg.gicp_use_recip);
#endif
#endif
        gicp.setInputTarget(tgt_ds);
        gicp.setInputSource(src_ds);
        pcl::PointCloud<PointType> aligned;
        gicp.align(aligned, T);
        if (gicp.hasConverged()) {
            res.ok = true;
            res.T = gicp.getFinalTransformation();
            res.score = gicp.getFitnessScore();
            res.method = "gicp";
        }
    }

    return res;
}

// ===== 后端线程主函数 =====
// ===== 后端线程主函数 =====
// ===== 后端线程主函数 =====
// ===== 后端线程主函数 =====
void backendThreadFunc()
{
    ROS_INFO("[Backend] Backend thread started.");
    g_backend_state.is_running = true;
    
    // 初始化后端 EKF
    double epsi[23];
    std::fill(epsi, epsi + 23, 0.001);
    kf_backend.init_dyn_share(get_f, df_dx, df_dw, h_share_model_submap, NUM_MAX_ITERATIONS, epsi);
    
    bool backend_initialized = false;
    bool has_prev_frontend = false;
    state_ikfom prev_frontend_state;
    
    while (!backend_should_stop)
    {
        BackendData data;
        
        // 等待数据
        {
            std::unique_lock<std::mutex> lock(mtx_backend);
            cv_backend.wait(lock, [&]{ 
                return !backend_queue.empty() || backend_should_stop; 
            });
            
            if (backend_should_stop) break;
            if (backend_queue.empty()) continue;
            
            data = backend_queue.front();
            backend_queue.pop_front();
        }
        
        // ===== 首次初始化：NDT 配准 =====
        if (!backend_initialized)
        {
            ROS_INFO("[Backend] First submap received. Performing NDT initialization...");
            
            // 检查全局地图
            if (!g_global_map_ready || !g_global_map_ds || g_global_map_ds->empty())
            {
                ROS_ERROR("[Backend] Global map not ready for NDT!");
                continue;
            }
            
            // 检查 submap
            if (!data.submap || data.submap->size() < 100)
            {
                ROS_WARN("[Backend] Submap too small (%zu points), waiting...", 
                         data.submap ? data.submap->size() : 0);
                continue;
            }
            
            // 检查初始位姿
            if (!has_ndt_init_pose)
            {
                ROS_WARN_THROTTLE(2.0, "[Backend] Waiting for /initpose...");
                // 将数据放回队列
                {
                    std::lock_guard<std::mutex> lock(mtx_backend);
                    backend_queue.push_front(data);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            
            // === NDT 配准 ===
            try {
                // 准备点云
                pcl::PointCloud<PointType>::Ptr src(new pcl::PointCloud<PointType>(*data.submap));
                pcl::PointCloud<PointType>::Ptr tgt(new pcl::PointCloud<PointType>(*g_global_map_ds));
                
                // 去除 NaN
                std::vector<int> idx;
                pcl::removeNaNFromPointCloud(*src, *src, idx);
                pcl::removeNaNFromPointCloud(*tgt, *tgt, idx);
                src->is_dense = true;
                tgt->is_dense = true;
                
                // 点数检查
                if (src->size() < 100 || tgt->size() < 100)
                {
                    ROS_ERROR("[Backend NDT] Insufficient points: src=%zu, tgt=%zu", 
                              src->size(), tgt->size());
                    continue;
                }
                
                // 降采样 src（NDT 输入不要太密集）
                if (src->size() > 20000) {
                    pcl::VoxelGrid<PointType> vg;
                    vg.setLeafSize(0.5f, 0.5f, 0.5f);
                    vg.setInputCloud(src);
                    pcl::PointCloud<PointType>::Ptr src_ds(new pcl::PointCloud<PointType>());
                    vg.filter(*src_ds);
                    src = src_ds;
                    ROS_INFO("[Backend NDT] Source downsampled: %zu -> %zu", 
                             data.submap->size(), src->size());
                }
                
                ROS_INFO("[Backend NDT] Source: %zu points, Target: %zu points", 
                         src->size(), tgt->size());
                
                // NDT 配置
                pcl::NormalDistributionsTransform<PointType, PointType> ndt;
                ndt.setResolution(2.0);           // NDT 体素大小
                ndt.setStepSize(0.2);             // 优化步长
                ndt.setTransformationEpsilon(1e-3);
                ndt.setMaximumIterations(50);
                ndt.setInputTarget(tgt);
                ndt.setInputSource(src);
                
                // 初始位姿（来自 /initpose）
                Eigen::Matrix4f T_init = Eigen::Matrix4f::Identity();
                T_init.topLeftCorner<3,3>() = ndt_init_q.toRotationMatrix().cast<float>();
                T_init.block<3,1>(0,3) = ndt_init_t.cast<float>();
                
                ROS_INFO("[Backend NDT] Initial pose: t=[%.2f, %.2f, %.2f]",
                         ndt_init_t.x(), ndt_init_t.y(), ndt_init_t.z());
                
                // 执行配准
                pcl::PointCloud<PointType> aligned;
                double ndt_start = omp_get_wtime();
                ndt.align(aligned, T_init);
                double ndt_time = omp_get_wtime() - ndt_start;
                
                // 检查收敛
                if (ndt.hasConverged()) {
                    Eigen::Matrix4f T_map_local_f = ndt.getFinalTransformation();
                    if (T_map_local_f.allFinite()) {
                        Eigen::Isometry3d T_map_local_iso = Eigen::Isometry3d::Identity();
                        T_map_local_iso.linear()  = T_map_local_f.topLeftCorner<3,3>().cast<double>();
                        T_map_local_iso.translation() = T_map_local_f.block<3,1>(0,3).cast<double>();

                        // 1) 保存/发布 local->map 变换（唯一真相）
                        {
                            std::lock_guard<std::mutex> lk(g_mtx_TmapLocal);
                            g_T_map_local = T_map_local_iso;
                            g_has_T_map_local.store(true, std::memory_order_release);
                        }

                        // 2) 初始化后端状态到 map（IMU in map）
                        //    T_map_imu = T_map_local * T_local_imu
                        Eigen::Matrix3d R_local_imu = data.frontend_state.rot.toRotationMatrix();
                        Eigen::Vector3d t_local_imu = data.frontend_state.pos;

                        Eigen::Matrix3d R_map_imu = g_T_map_local.rotation() * R_local_imu;
                        Eigen::Vector3d t_map_imu = g_T_map_local.rotation() * t_local_imu + g_T_map_local.translation();

                        state_ikfom init_state = data.frontend_state;
                        init_state.rot = Eigen::Quaterniond(R_map_imu);
                        init_state.pos = t_map_imu;
                        kf_backend.change_x(init_state);

                        g_backend_state.global_state = init_state;
                        g_backend_state.has_global_state = true;
                        prev_frontend_state = data.frontend_state;
                        has_prev_frontend = true;

                        // 3) 可视化：把 submap 变到 map
                        {
                            std::lock_guard<std::mutex> lk(mtx_backend_vis);
                            g_current_submap = data.submap;
                            transformPointCloudToGlobal(g_current_submap, g_current_submap_global, g_T_map_local.matrix());
                        }

                        ROS_INFO("[Backend NDT] init done. fitness=%.6f", ndt.getFitnessScore());
                        backend_initialized = true;
                        continue;
                    }
                }
                
                // 获取结果
                Eigen::Matrix4f T_map_submap_f = ndt.getFinalTransformation();
                Eigen::Matrix4d T_map_submap = T_map_submap_f.cast<double>();
                
                // 检查有效性
                if (!T_map_submap.allFinite())
                {
                    ROS_ERROR("[Backend NDT] Result contains NaN/Inf!");
                    continue;
                }
                
                double fitness = ndt.getFitnessScore();
                if (fitness > 1.0) {  // 阈值可调
                    ROS_WARN("[Backend NDT] Fitness score high (%.3f), result may be unreliable!", fitness);
                    // 可以选择拒绝或继续
                }
                
                // === 计算全局初始状态 ===
                // T_map_submap 是：submap(local) -> map(global)
                // submap 中心在 local 原点附近
                // 所以 T_map_local ≈ T_map_submap
                
                Eigen::Matrix3d R_map_local = T_map_submap.topLeftCorner<3,3>();
                Eigen::Vector3d t_map_local = T_map_submap.block<3,1>(0,3);
                
                // 前端状态：IMU in local frame
                Eigen::Matrix3d R_local_imu = data.frontend_state.rot.toRotationMatrix();
                Eigen::Vector3d t_local_imu = data.frontend_state.pos;
                
                // 全局状态：IMU in map frame
                // T_map_imu = T_map_local * T_local_imu
                Eigen::Matrix3d R_map_imu = R_map_local * R_local_imu;
                Eigen::Vector3d t_map_imu = R_map_local * t_local_imu + t_map_local;
                
                // 初始化后端 EKF 状态
                state_ikfom init_state = data.frontend_state;
                init_state.rot = Eigen::Quaterniond(R_map_imu);
                init_state.pos = t_map_imu;
                
                kf_backend.change_x(init_state);
                g_backend_state.global_state = init_state;
                g_backend_state.has_global_state = true;
                backend_initialized = true;
                prev_frontend_state = data.frontend_state;
                has_prev_frontend = true;
                
                // 保存变换矩阵（local -> map）
                {
                    std::lock_guard<std::mutex> lock(g_mtx_TmapLocal);
                    g_T_map_local.linear() = R_map_local;
                    g_T_map_local.translation() = t_map_local;
                    g_has_T_map_local.store(true, std::memory_order_release);
                }
                
                // 变换 submap 到全局坐标系
                {
                    std::lock_guard<std::mutex> lock(mtx_backend_vis);
                    transformPointCloudToGlobal(data.submap, g_current_submap_global, T_map_submap);
                    g_current_submap = data.submap;  // 保存 local 版本
                }
                
                ROS_WARN("[Backend NDT] ==== INITIALIZATION SUCCESS ====");
                ROS_INFO("[Backend NDT] Time: %.3f ms, Fitness: %.6f, Iters: %d",
                         ndt_time * 1000, fitness, ndt.getFinalNumIteration());
                
                Eigen::Vector3d rpy = R_map_imu.eulerAngles(2,1,0) * 180.0 / M_PI;  // yaw, pitch, roll
                ROS_INFO("[Backend NDT] Global pose: t=[%.2f, %.2f, %.2f], RPY=[%.1f, %.1f, %.1f] deg",
                         t_map_imu.x(), t_map_imu.y(), t_map_imu.z(), rpy.x(), rpy.y(), rpy.z());
                ROS_INFO("[Backend NDT] Submap: %zu pts (local) -> %zu pts (global)",
                         data.submap->size(), g_current_submap_global->size());
                
                continue;
                
            } catch (const std::exception& e) {
                ROS_ERROR("[Backend NDT] Exception: %s", e.what());
                continue;
            }
        }
        
        // ===== 后续优化（保持原样，使用 h_share_model_submap） =====
        if (!backend_initialized)
        {
            ROS_WARN("[Backend] Not initialized, skipping.");
            continue;
        }
        
        double t_start = omp_get_wtime();
        
        // 预测项：仅使用“后端上一次全局状态 + 前端相对运动增量”。
        // 不再每次强行绑定前端绝对位姿，保证后端可被 global map 持续拉回与重定位。
        state_ikfom pred_state = g_backend_state.global_state;
        if (has_prev_frontend) {
            Eigen::Matrix4d T_local_prev = Eigen::Matrix4d::Identity();
            T_local_prev.topLeftCorner<3,3>() = prev_frontend_state.rot.toRotationMatrix();
            T_local_prev.block<3,1>(0,3) = prev_frontend_state.pos;

            Eigen::Matrix4d T_local_cur = Eigen::Matrix4d::Identity();
            T_local_cur.topLeftCorner<3,3>() = data.frontend_state.rot.toRotationMatrix();
            T_local_cur.block<3,1>(0,3) = data.frontend_state.pos;

            // 前端相对运动：prev IMU -> cur IMU（在 prev IMU 坐标中表达）
            const Eigen::Matrix4d T_delta = T_local_prev.inverse() * T_local_cur;

            Eigen::Matrix4d T_map_prev = Eigen::Matrix4d::Identity();
            T_map_prev.topLeftCorner<3,3>() = g_backend_state.global_state.rot.toRotationMatrix();
            T_map_prev.block<3,1>(0,3) = g_backend_state.global_state.pos;

            const Eigen::Matrix4d T_map_pred = T_map_prev * T_delta;
            pred_state.pos = T_map_pred.block<3,1>(0,3);
            pred_state.rot = Eigen::Quaterniond(T_map_pred.topLeftCorner<3,3>());
        }

        // Continuous NDT/GICP alignment (optional)
        if (g_backend_align_cfg.enable && g_backend_align_cfg.stride > 0 &&
            (g_backend_state.total_updates % g_backend_align_cfg.stride == 0)) {
            PointCloudXYZI::Ptr align_src = (data.submap && !data.submap->empty())
                ? data.submap : data.current_scan_local;
            if (align_src && !align_src->empty()) {
                Eigen::Matrix4f T_guess = Eigen::Matrix4f::Identity();
                if (g_has_T_map_local.load(std::memory_order_acquire)) {
                    std::lock_guard<std::mutex> lk(g_mtx_TmapLocal);
                    T_guess = g_T_map_local.matrix().cast<float>();
                } else {
                    T_guess = computeTransformMapLocal(pred_state, data.frontend_state).cast<float>();
                }
                Eigen::Vector3f center_map =
                    T_guess.topLeftCorner<3,3>() * data.frontend_state.pos.cast<float>() +
                    T_guess.block<3,1>(0,3);

                AlignResult align_res = RunNDTGICPOnce(align_src, T_guess, center_map);
                if (align_res.ok &&
                    (align_res.score <= g_backend_align_cfg.fitness_thresh ||
                     g_backend_align_cfg.fitness_thresh <= 0.0)) {
                    Eigen::Matrix4d T_map_local_align = align_res.T.cast<double>();
                    Eigen::Matrix3d R_align = T_map_local_align.topLeftCorner<3,3>();
                    R_align = OrthonormalizeRotation(R_align);
                    T_map_local_align.topLeftCorner<3,3>() = R_align;

                    Eigen::Matrix4d T_map_imu = ComputeMapImuFromMapLocal(
                        T_map_local_align, data.frontend_state);
                    pred_state.pos = T_map_imu.block<3,1>(0,3);
                    pred_state.rot = Eigen::Quaterniond(T_map_imu.topLeftCorner<3,3>());
                    pred_state.rot.normalize();

                    {
                        std::lock_guard<std::mutex> lk(g_mtx_TmapLocal);
                        g_T_map_local.linear() = T_map_local_align.topLeftCorner<3,3>();
                        g_T_map_local.translation() = T_map_local_align.block<3,1>(0,3);
                        g_has_T_map_local.store(true, std::memory_order_release);
                    }

                    ROS_INFO_THROTTLE(1.0, "[Backend Align] %s ok, score=%.4f", 
                        align_res.method.c_str(), align_res.score);
                }
            }
        }

        // 设置当前 submap（观测模型会用）
        kf_backend.change_x(pred_state);
        g_current_submap = data.submap;
        g_current_scan_local = data.current_scan_local;
        g_submap_anchor_frontend_state = data.frontend_state;

        g_backend_state.total_updates++;
        
        double solve_H_time = 0;
        try {
            kf_backend.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            
            g_backend_state.global_state = kf_backend.get_x();
            g_backend_state.global_state.rot.normalize();
            g_backend_state.successful_updates++;

            // 诊断：本次后端 IEKF 对全局状态的修正幅度（若长期接近 0，则约束几乎未生效）
            double corr_trans = (g_backend_state.global_state.pos - pred_state.pos).norm();
            Eigen::Quaterniond dq = pred_state.rot.conjugate() * g_backend_state.global_state.rot;
            dq.normalize();
            double corr_rot_deg = Eigen::AngleAxisd(dq).angle() * 180.0 / M_PI;
            
            // 计算并写回 T_map_local
            const int min_eff_points = 500;
            bool update_tmaplocal = g_backend_effective_points >= min_eff_points;
            Eigen::Matrix4d T_map_local_new = Eigen::Matrix4d::Identity();
            if (update_tmaplocal) {
                T_map_local_new = computeTransformMapLocal(
                    g_backend_state.global_state,  // IMU in map
                    data.frontend_state            // IMU in local
                );
                Eigen::Matrix3d R_map_local = T_map_local_new.topLeftCorner<3,3>();
                R_map_local = OrthonormalizeRotation(R_map_local);
                T_map_local_new.topLeftCorner<3,3>() = R_map_local;
            } else {
                ROS_WARN_THROTTLE(1.0, "[Backend] Skip T_map_local update, too few effective points: %d", g_backend_effective_points);
            }
            static Eigen::Matrix4d T_map_local_prev = Eigen::Matrix4d::Identity();
            if (update_tmaplocal) {
                Eigen::Matrix4d T_map_local_delta = T_map_local_prev.inverse() * T_map_local_new;
                Eigen::Matrix3d R_delta = T_map_local_delta.topLeftCorner<3,3>();
                Eigen::Vector3d t_delta = T_map_local_delta.block<3,1>(0,3);
                double delta_trans = t_delta.norm();
                double delta_rot_deg = Eigen::AngleAxisd(R_delta).angle() * 180.0 / M_PI;
                T_map_local_prev = T_map_local_new;
                {
                    std::lock_guard<std::mutex> lk(g_mtx_TmapLocal);
                    g_T_map_local.linear()      = T_map_local_new.topLeftCorner<3,3>();
                    g_T_map_local.translation() = T_map_local_new.block<3,1>(0,3);
                    g_has_T_map_local.store(true, std::memory_order_release);
                }
                ROS_INFO_THROTTLE(1.0,
                    "[Backend] T_map_local delta: %.3f m, %.3f deg",
                    delta_trans,
                    delta_rot_deg);
            }

            // 可视化 submap->map
            {
                std::lock_guard<std::mutex> lk(mtx_backend_vis);
                g_current_submap = data.submap;
                transformPointCloudToGlobal(g_current_submap, g_current_submap_global, g_T_map_local.matrix());
            }
            
            double t_cost = omp_get_wtime() - t_start;
            ROS_INFO_THROTTLE(1.0, 
                "[Backend] Update OK! Cost: %.3f ms, Success: %d/%d (%.1f%%), Submap: %zu pts, Eff=%d, ResMean=%.4f m, Corr=[%.3f m, %.3f deg]",
                t_cost * 1000,
                g_backend_state.successful_updates,
                g_backend_state.total_updates,
                100.0 * g_backend_state.successful_updates / std::max(1, g_backend_state.total_updates),
                g_current_submap_global->size(),
                g_backend_effective_points,
                g_backend_residual_mean,
                corr_trans,
                corr_rot_deg);

            prev_frontend_state = data.frontend_state;
            has_prev_frontend = true;
            
        } catch (const std::exception& e) {
            ROS_ERROR("[Backend] Optimization failed: %s", e.what());
        }
    }
    
    g_backend_state.is_running = false;
    ROS_INFO("[Backend] Backend thread stopped.");
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh; 

    // --- publish & core params ---
    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("localization/map/map_file_path", map_file_path, "");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, false);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    cout << "map_file_path" << map_file_path << endl;

    // --- topics & multi-lidar ---
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/multi_lidar", multi_lidar, false);
    nh.param<string>("common/lid_topic2", lid_topic2, "/lidar2");
    nh.param<bool>("common/publish_tf_results", publish_tf_results, false);

    // --- preprocess params for L1 ---
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("preprocess/feature_extract_enable", p_pre->feature_enabled, false);

    // --- preprocess params for L2（默认复制 L1，再读覆盖） ---
    double blind2 = p_pre->blind;
    int N_SCANS2 = p_pre->N_SCANS;
    int time_unit2 = p_pre->time_unit;
    int SCAN_RATE2 = p_pre->SCAN_RATE;
    int point_filter_num2 = p_pre->point_filter_num;
    bool feature_enabled2 = p_pre->feature_enabled;
    nh.param<int>("preprocess2/lidar_type", lidar_type2, lidar_type);
    nh.param<double>("preprocess2/blind", blind2, blind2);
    nh.param<int>("preprocess2/scan_line", N_SCANS2, N_SCANS2);
    nh.param<int>("preprocess2/timestamp_unit", time_unit2, time_unit2);
    nh.param<int>("preprocess2/scan_rate", SCAN_RATE2, SCAN_RATE2);
    nh.param<int>("preprocess2/point_filter_num", point_filter_num2, point_filter_num2);
    nh.param<bool>("preprocess2/feature_extract_enable", feature_enabled2, feature_enabled2);

    // --- mapping & filters ---
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);

    // --- extrinsic (IMU->L1) ---
    std::vector<double> extrinT(3, 0.0), extrinR(9, 0.0);
    nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT, std::vector<double>());
    nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR, std::vector<double>());

    // --- extrinsic multi lidar mode ---
    nh.param<bool>("mapping/extrinsic_imu_to_lidars", extrinsic_imu_to_lidars, false);

    // --- L2 extrinsic options (mutual exclusive by flag) ---
    std::vector<double> extrinT2(3, 0.0), extrinR2(9, 0.0);                    // IMU->L2 (when extrinsic_imu_to_lidars=true)
    std::vector<double> extrinT_L2wrtL1(3, 0.0), extrinR_L2wrtL1(9, 0.0);      // L2->L1 (when extrinsic_imu_to_lidars=false)
    nh.param<std::vector<double>>("mapping/extrinsic_T2", extrinT2, extrinT2);
    nh.param<std::vector<double>>("mapping/extrinsic_R2", extrinR2, extrinR2);
    nh.param<std::vector<double>>("mapping/extrinsic_T_L2_wrt_L1", extrinT_L2wrtL1, extrinT_L2wrtL1);
    nh.param<std::vector<double>>("mapping/extrinsic_R_L2_wrt_L1", extrinR_L2wrtL1, extrinR_L2wrtL1);

    // --- visualization TF (L1 wrt drone) ---
    std::vector<double> extrinT_L1wrtDrone(3, 0.0), extrinR_L1wrtDrone(9, 0.0);
    nh.param<std::vector<double>>("mapping/extrinsic_T_L1_wrt_drone", extrinT_L1wrtDrone, extrinT_L1wrtDrone);
    nh.param<std::vector<double>>("mapping/extrinsic_R_L1_wrt_drone", extrinR_L1wrtDrone, extrinR_L1wrtDrone);

    // --- optional sync thresholds ---
    nh.param<double>("sync/l2l_max_gap", l2l_max_gap, 0.05);

    // --- localization params ---
    nh.param<std::string>("localization/map/map_file_path", g_loc.map_file_path, std::string(""));
    nh.param<double>("localization/map/voxel_leaf", g_loc.map_voxel_leaf, 0.4);
    nh.param<bool>("localization/map/remove_outliers", g_loc.map_remove_outliers, false);
    nh.param<int>("localization/map/outlier_nb_neighbors", g_loc.map_outlier_nb_neighbors, 20);
    nh.param<double>("localization/map/outlier_stddev_mul", g_loc.map_outlier_stddev_mul, 2.0);

    nh.param<bool>("localization/global_point_residual/enable", g_loc.gpr_enable, true);
    nh.param<int>("localization/global_point_residual/max_points", g_loc.gpr_max_points, 1200);
    nh.param<double>("localization/global_point_residual/voxel_leaf", g_loc.gpr_voxel_leaf, 0.5);
    nh.param<int>("localization/global_point_residual/plane_min_neighbors", g_loc.gpr_plane_min_neighbors, 10);
    nh.param<double>("localization/global_point_residual/max_point2plane_dist", g_loc.gpr_max_point2plane_dist, 0.6);
    nh.param<double>("localization/global_point_residual/sigma_m", g_loc.gpr_sigma_m, 0.06);
    nh.param<std::string>("localization/global_point_residual/robust_kernel", g_loc.gpr_robust_kernel, std::string("huber"));
    nh.param<double>("localization/global_point_residual/huber_delta", g_loc.gpr_huber_delta, 0.5);
    nh.param<double>("localization/global_point_residual/cauchy_c", g_loc.gpr_cauchy_c, 0.5);

    nh.param<double>("localization/global_align/roi_radius", g_loc.roi_radius, 60.0);

    // --- backend params ---
    nh.param<bool>("backend/enable", g_backend_cfg.enable, false);
    nh.param<int>("backend/min_local_points", g_backend_cfg.min_local_points, 50000);
    nh.param<float>("backend/min_local_radius", g_backend_cfg.min_local_radius, 30.0f);
    nh.param<int>("backend/min_frames", g_backend_cfg.min_frames, 100);
    nh.param<float>("backend/submap_radius", g_backend_cfg.submap_radius, 25.0f);
    nh.param<float>("backend/submap_voxel_size", g_backend_cfg.submap_voxel_size, 0.4f);
    nh.param<int>("backend/update_interval", g_backend_cfg.update_interval, 1);
    nh.param<float>("backend/max_point2plane_dist", g_backend_cfg.max_point2plane_dist, 2.5f);
    nh.param<float>("backend/global_residual_weight", g_backend_cfg.global_residual_weight, 6.0f);
    nh.param<int>("backend/max_global_points", g_backend_cfg.max_global_points, 4000);

    nh.param<bool>("backend/continuous_align_enable", g_backend_align_cfg.enable, true);
    nh.param<int>("backend/continuous_align_stride", g_backend_align_cfg.stride, 1);

    nh.param<double>("localization/global_align/roi_radius", g_backend_align_cfg.roi_radius, 60.0);
    nh.param<double>("localization/global_align/fallback_expand_roi", g_backend_align_cfg.fallback_expand, 1.5);
    nh.param<double>("localization/global_align/max_roi_radius", g_backend_align_cfg.roi_max, 120.0);
    nh.param<double>("localization/global_align/fitness_thresh", g_backend_align_cfg.fitness_thresh, 2.0);

    nh.getParam("localization/ndt/resolutions", g_backend_align_cfg.ndt_resolutions);
    nh.getParam("localization/ndt/max_iterations", g_backend_align_cfg.ndt_max_iterations);
    nh.param<double>("localization/ndt/step_size", g_backend_align_cfg.ndt_step_size, 0.7);
    nh.param<double>("localization/ndt/trans_eps", g_backend_align_cfg.ndt_trans_eps, 1e-3);
    nh.param<double>("localization/ndt/score_clip", g_backend_align_cfg.ndt_score_clip, 5.0);

    nh.param<bool>("localization/gicp/enable", g_backend_align_cfg.gicp_enable, true);
    nh.param<int>("localization/gicp/max_iterations", g_backend_align_cfg.gicp_max_iterations, 60);
    nh.param<double>("localization/gicp/max_corr_dist", g_backend_align_cfg.gicp_max_corr_dist, 2.0);
    nh.param<double>("localization/gicp/trans_eps", g_backend_align_cfg.gicp_trans_eps, 1e-4);
    nh.param<double>("localization/gicp/euclidean_fitness_epsilon", g_backend_align_cfg.gicp_fit_eps, 1e-5);
    nh.param<int>("localization/gicp/correspondence_randomness", g_backend_align_cfg.gicp_corr_randomness, 30);
    nh.param<bool>("localization/gicp/use_reciprocal", g_backend_align_cfg.gicp_use_recip, true);
    
    ROS_INFO("[Backend] enable=%d, min_points=%d, submap_radius=%.1fm, update_interval=%d",
             g_backend_cfg.enable, g_backend_cfg.min_local_points, 
             g_backend_cfg.submap_radius, g_backend_cfg.update_interval);

    // --- display ---
    p_pre->lidar_type = lidar_type;
    std::cout << "p_pre->lidar_type " << p_pre->lidar_type << std::endl;

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "map";

    // --- variables definition ---
    int effect_feat_num = 0, frame_num = 0;
    double deltaT=0, deltaR=0, aver_time_consu=0, aver_time_icp=0, aver_time_match=0, aver_time_incre=0, aver_time_solve=0, aver_time_const_H_time=0;
    bool flg_EKF_converged=false, EKF_stop_flg = false;

    double FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    double HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    // --- IMU/Extrinsic/EKF init ---
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;

    double epsi[23]; std::fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    // --- 外参：构造 T_L1_L2_ (L2 -> L1) ---
    if (multi_lidar)
    {
        Eigen::Matrix4d T_I_L1 = Eigen::Matrix4d::Identity();
        T_I_L1.topLeftCorner<3,3>() = Lidar_R_wrt_IMU;
        T_I_L1.block<3,1>(0,3) = Lidar_T_wrt_IMU;

        Eigen::Matrix4d T_L1_L2_d = Eigen::Matrix4d::Identity();

        if (extrinsic_imu_to_lidars)
        {
            if (extrinR2.size() == 9 && extrinT2.size() == 3)
            {
                V3D t_I_L2; t_I_L2 << VEC_FROM_ARRAY(extrinT2);
                M3D R_I_L2; R_I_L2 << MAT_FROM_ARRAY(extrinR2);
                Eigen::Matrix4d T_I_L2 = Eigen::Matrix4d::Identity();
                T_I_L2.topLeftCorner<3,3>() = R_I_L2;
                T_I_L2.block<3,1>(0,3) = t_I_L2;

                // 注意方向：T_L1_L2 = (T_I_L1)^-1 * T_I_L2，作用于 L2 点 -> L1
                T_L1_L2_d = T_I_L1.inverse() * T_I_L2;
            }
            else
            {
                ROS_ERROR("extrinsic_imu_to_lidars=true but extrinsic_T2/R2 not provided correctly.");
            }
        }
        else
        {
            if (extrinR_L2wrtL1.size() == 9 && extrinT_L2wrtL1.size() == 3)
            {
                V3D t_L1_L2; t_L1_L2 << VEC_FROM_ARRAY(extrinT_L2wrtL1);
                M3D R_L1_L2; R_L1_L2 << MAT_FROM_ARRAY(extrinR_L2wrtL1);
                T_L1_L2_d.topLeftCorner<3,3>() = R_L1_L2;
                T_L1_L2_d.block<3,1>(0,3) = t_L1_L2;
            }
            else
            {
                ROS_ERROR("extrinsic_imu_to_lidars=false but extrinsic_T_L2_wrt_L1 / extrinsic_R_L2_wrt_L1 not provided correctly.");
            }
        }

        T_L1_L2_ = T_L1_L2_d.cast<float>();

        std::cout << "\033[32;1m[multi_lidar] ON\n"
                  << "  L1 type=" << lidar_type
                  << "  L2 type=" << lidar_type2 << "\n"
                  << "  T(L1 <- L2) = \n" << T_L1_L2_d << "\033[0m\n";
    }

    // --- 可视化 TF：L1 wrt drone ---
    if (publish_tf_results)
    {
        if (extrinR_L1wrtDrone.size() == 9 && extrinT_L1wrtDrone.size() == 3)
        {
            V3D t_L1_D; t_L1_D << VEC_FROM_ARRAY(extrinT_L1wrtDrone);
            M3D R_L1_D; R_L1_D << MAT_FROM_ARRAY(extrinR_L1wrtDrone);
            LiDAR1_wrt_drone.setIdentity();
            LiDAR1_wrt_drone.topLeftCorner<3,3>() = R_L1_D;
            LiDAR1_wrt_drone.block<3,1>(0,3) = t_L1_D;

            std::cout << "\033[32;1m[tf] LiDAR1 wrt Drone (T_Drone_L1):\n"
                      << LiDAR1_wrt_drone << "\033[0m\n";
        }
        else
        {
            ROS_WARN("publish_tf_results enabled but extrinsic_T/R_L1_wrt_drone not fully provided.");
        }
    }

    // --- debug record ---
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    // --- load and publish global map (optional) ---
    ros::Publisher pubGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("/global_map", 1, true);
    g_global_map_ready = LoadGlobalMapAndBuildIndex();
    if (!g_global_map_ready) {
        ROS_WARN("[localization] Global map not ready, will run without global residuals.");
    } else {
        publish_global_map(pubGlobalMap);
        ROS_INFO("[localization] Global map published to /global_map.");
    }
    if (g_global_map_ready) {
        if (!g_global_map_ds->empty()) {
            ikdtree_global.Build(g_global_map_ds->points);
            g_global_ikdtree_ready = true;
            ROS_INFO("[localization] ikdtree_global built. size=%zu", g_global_map_ds->points.size());
        } else {
            ROS_WARN("[localization] global map ds empty, skip ikdtree_global build.");
        }
    }

    // --- Preprocess init for L1 ---
    p_pre->set(p_pre->feature_enabled, p_pre->lidar_type, p_pre->blind, p_pre->point_filter_num);

    // --- Create and init Preprocess for L2 (if enabled) ---
    shared_ptr<Preprocess> p_pre2_local;
    ros::Subscriber sub_pcl2; // 只有在 multi_lidar 时使用
    if (multi_lidar) {
        p_pre2_local.reset(new Preprocess());
        p_pre2_local->lidar_type = lidar_type2;
        p_pre2_local->N_SCANS = N_SCANS2;
        p_pre2_local->time_unit = time_unit2;
        p_pre2_local->SCAN_RATE = SCAN_RATE2;
        p_pre2_local->point_filter_num = point_filter_num2;
        p_pre2_local->feature_enabled = feature_enabled2;
        p_pre2_local->blind = blind2;
        p_pre2_local->set(p_pre2_local->feature_enabled, p_pre2_local->lidar_type, p_pre2_local->blind, p_pre2_local->point_filter_num);

        // 将全局的 p_pre2 指向该实例（如果你在其他文件/回调用了全局 p_pre2）
        p_pre2 = p_pre2_local;
    }

    // --- ROS subscribe initialization ---
    ros::Subscriber sub_pcl = (p_pre->lidar_type == AVIA)
        ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk)
        : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);

    if (multi_lidar) {
        if (lidar_type2 == AVIA) {
            sub_pcl2 = nh.subscribe(lid_topic2, 200000, livox_pcl_cbk2);
        } else {
            sub_pcl2 = nh.subscribe(lid_topic2, 200000, standard_pcl_cbk2);
        }
        ROS_INFO_STREAM("[multi_lidar] enabled. L1=" << lid_topic << " (type=" << p_pre->lidar_type
                        << "), L2=" << lid_topic2 << " (type=" << lidar_type2 << ")");
    } else {
        ROS_INFO_STREAM("[multi_lidar] disabled. L1=" << lid_topic << " (type=" << p_pre->lidar_type << ")");
    }

    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/frontend/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

    // ===== 后端可视化 =====
    // ===== 后端可视化 publishers =====
    ros::Publisher pubBackendSubmap = nh.advertise<sensor_msgs::PointCloud2>("/backend/submap", 10);
    ros::Publisher pubBackendScanGlobal = nh.advertise<sensor_msgs::PointCloud2>("/backend/current_scan_global", 100);
    ros::Publisher pubBackendOdom = nh.advertise<nav_msgs::Odometry>("/backend/odometry", 100);
    ros::Publisher pubBackendPath = nh.advertise<nav_msgs::Path>("/backend/path", 100);
    
    nav_msgs::Path backend_path;
    backend_path.header.frame_id = "map";

    ros::Subscriber ndt_init_sub = nh.subscribe<nav_msgs::Odometry>(
        "/initpose",    // 你的输入话题名
        1, 
        ndtInitPoseCallback
    );

    // --- 启动后端线程 ---
    if (g_backend_cfg.enable && g_global_ikdtree_ready) {
        backend_thread = std::thread(backendThreadFunc);
        ROS_INFO("[Backend] Backend thread launched.");
    }

    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();

        // 用条件变量可以更省 CPU，这里保持你的节奏
        if (sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0, t1, t3, t5;
            double svd_time = 0;
            match_time = 0; kdtree_search_time = 0.0; solve_time = 0; solve_const_H_time = 0;
            t0 = omp_get_wtime();

            // 单雷达或双雷达：此处仍调用单雷达 Process。
            // 如需双雷达融合 deskew + 合并，请改为 p_imu->ProcessDual(Measures, kf, feats_undistort)（需你实现）
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (!feats_undistort || feats_undistort->empty())
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) >= INIT_TIME;

            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();

            // 首帧初始化 ikdtree/全局对齐逻辑（保持你的原样，略去中间未改动部分）
            static bool g_first_global_align_done_local = false;
            if (ikdtree.Root_Node == nullptr) {
                if (feats_down_size > 5) {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++) {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);

                    if (!g_first_global_align_done_local) {
                        g_first_global_align_done_local = true;
                        ROS_INFO("[Frontend] First local tree built. Backend will handle global initialization.");
                    }
                }
                continue;
            }

            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int rematch_num = 0;
            bool nearest_search_en = true;

            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];
            double t_update_end = omp_get_wtime();

            publish_odometry(pubOdomAftMapped);

            // ===== 后端定位处理 =====
            g_backend_state.frame_count++;
            
            if (shouldActivateBackend())
            {
                g_backend_state.update_count++;
                
                // 每隔 N 帧触发后端更新
                if (g_backend_state.update_count % g_backend_cfg.update_interval == 0)
                {
                    V3D current_pos = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                    
                    // 提取 submap
                    PointCloudXYZI::Ptr submap_extracted(new PointCloudXYZI());
                    bool extract_ok = extractSubmapFromLocal(
                        ikdtree,
                        current_pos,
                        g_backend_cfg.submap_radius,
                        submap_extracted);
                    
                    if (extract_ok && submap_extracted->size() > 100)
                    {
                        // 准备后端数据
                        BackendData backend_data;
                        backend_data.frontend_state = state_point;
                        backend_data.submap = submap_extracted;
                        backend_data.timestamp = Measures.lidar_beg_time;

                        PointCloudXYZI::Ptr current_scan_local(new PointCloudXYZI());
                        current_scan_local->resize(feats_down_body->size());
                        for (size_t i = 0; i < feats_down_body->size(); i++) {
                            pointBodyToWorld(&(feats_down_body->points[i]), &(current_scan_local->points[i]));
                        }
                        backend_data.current_scan_local = current_scan_local;

                        // 推送到队列
                        {
                            std::lock_guard<std::mutex> lock(mtx_backend);
                            
                            // 队列满则丢弃最旧的
                            if (backend_queue.size() >= MAX_BACKEND_QUEUE_SIZE) {
                                backend_queue.pop_front();
                                ROS_WARN_THROTTLE(1.0, "[Backend] Queue full, dropping oldest data.");
                            }
                            
                            backend_queue.push_back(backend_data);
                        }
                        cv_backend.notify_one();
                        
                        ROS_INFO_THROTTLE(2.0, "[Backend] Pushed data to queue. Queue size: %zu", 
                                         backend_queue.size());
                    }
                }
            }

            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();

            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            // ===== 后端可视化 =====
            // ===== 后端可视化 =====
            if (g_backend_state.is_active && g_backend_state.has_global_state)
            {
                // 1. 发布全局坐标系下的 submap
                PointCloudXYZI::Ptr submap_global_copy(new PointCloudXYZI());
                {
                    std::lock_guard<std::mutex> lock(mtx_backend_vis);
                    if (!g_current_submap_global->empty()) {
                        *submap_global_copy = *g_current_submap_global;
                    }
                }
                
                if (!submap_global_copy->empty()) {
                    sensor_msgs::PointCloud2 submap_msg;
                    pcl::toROSMsg(*submap_global_copy, submap_msg);
                    submap_msg.header.stamp = ros::Time().fromSec(lidar_end_time);
                    submap_msg.header.frame_id = "map";
                    pubBackendSubmap.publish(submap_msg);
                }
                
                // 2. 变换当前帧到全局坐标系
                if (feats_down_world && !feats_down_world->empty())
                {
                    PointCloudXYZI::Ptr current_scan_global(new PointCloudXYZI());
                    Eigen::Isometry3d T_map_local_copy = Eigen::Isometry3d::Identity();

                    {
                        std::lock_guard<std::mutex> lock(g_mtx_TmapLocal);
                        T_map_local_copy = g_T_map_local; // OK: Isometry3d 赋值
                    }

                    // 如果 transformPointCloudToGlobal 接受 Matrix4d，就传 matrix()
                    transformPointCloudToGlobal(feats_down_world, current_scan_global, T_map_local_copy.matrix());

                    if (!current_scan_global->empty()) {
                        sensor_msgs::PointCloud2 scan_msg;
                        pcl::toROSMsg(*current_scan_global, scan_msg);
                        scan_msg.header.stamp = ros::Time().fromSec(lidar_end_time);
                        scan_msg.header.frame_id = "map";
                        pubBackendScanGlobal.publish(scan_msg);
                    }
                }
                
                // 3. 发布后端 odometry
                nav_msgs::Odometry backend_odom;
                backend_odom.header.frame_id = "map";
                backend_odom.child_frame_id = "body_backend";
                backend_odom.header.stamp = ros::Time().fromSec(lidar_end_time);
                
                backend_odom.pose.pose.position.x = g_backend_state.global_state.pos(0);
                backend_odom.pose.pose.position.y = g_backend_state.global_state.pos(1);
                backend_odom.pose.pose.position.z = g_backend_state.global_state.pos(2);
                backend_odom.pose.pose.orientation.x = g_backend_state.global_state.rot.coeffs()[0];
                backend_odom.pose.pose.orientation.y = g_backend_state.global_state.rot.coeffs()[1];
                backend_odom.pose.pose.orientation.z = g_backend_state.global_state.rot.coeffs()[2];
                backend_odom.pose.pose.orientation.w = g_backend_state.global_state.rot.coeffs()[3];
                
                pubBackendOdom.publish(backend_odom);
                
                // 4. 发布后端 path
                geometry_msgs::PoseStamped backend_pose;
                backend_pose.header = backend_odom.header;
                backend_pose.pose = backend_odom.pose.pose;
                
                static int backend_path_counter = 0;
                backend_path_counter++;
                if (backend_path_counter % 5 == 0) {
                    backend_path.poses.push_back(backend_pose);
                    backend_path.header.stamp = ros::Time().fromSec(lidar_end_time);
                    pubBackendPath.publish(backend_path);
                }
                
                // 5. 打印调试信息
                static int print_counter = 0;
                print_counter++;
                if (print_counter % 50 == 0) {
                    V3D pos_frontend_local = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                    V3D pos_backend_map = g_backend_state.global_state.pos +
                            g_backend_state.global_state.rot * g_backend_state.global_state.offset_T_L_I;

                    Eigen::Isometry3d T_map_local_copy = Eigen::Isometry3d::Identity();
                    {
                    std::lock_guard<std::mutex> lock(g_mtx_TmapLocal);
                    T_map_local_copy = g_T_map_local;
                    }
                    V3D pos_frontend_map = T_map_local_copy.rotation() * pos_frontend_local +
                               T_map_local_copy.translation();

                    V3D diff = pos_backend_map - pos_frontend_map;
                    
                    ROS_INFO("[Backend Debug] Frontend pos (map): [%.2f, %.2f, %.2f]", 
                        pos_frontend_map.x(), pos_frontend_map.y(), pos_frontend_map.z());
                    ROS_INFO("[Backend Debug] Backend  pos (map): [%.2f, %.2f, %.2f]", 
                        pos_backend_map.x(), pos_backend_map.y(), pos_backend_map.z());
                    ROS_INFO("[Backend Debug] Difference (map):   [%.2f, %.2f, %.2f] (norm: %.2fm)", 
                        diff.x(), diff.y(), diff.z(), diff.norm());
                }
            }

            if (runtime_pos_log)
            {
                frame_num++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp   = aver_time_icp   * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;

                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                if (time_log_counter < MAXN) time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",
                       t1-t0, aver_time_match, aver_time_solve, t3-t1, t5-t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()
                         << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                         << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    // save map
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec5;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0; i < time_log_counter; i++){
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",
                    T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]),
                    int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    // 停止后端线程
    if (g_backend_cfg.enable && backend_thread.joinable()) {
        ROS_INFO("[Backend] Stopping backend thread...");
        {
            std::lock_guard<std::mutex> lock(mtx_backend);
            backend_should_stop = true;
        }
        cv_backend.notify_all();
        backend_thread.join();
        ROS_INFO("[Backend] Backend thread stopped.");
    }

    return 0;
}
