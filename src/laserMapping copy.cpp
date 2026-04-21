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

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>

#include <unordered_set>
#include <unordered_map>
#include <cmath>

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
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
int lidar_type;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
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
KD_TREE<PointType> ikdtree_global;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/* Localization */

struct LocalizationParams {
    std::string map_file_path = "";
    double map_voxel_leaf = 0.4;
    bool map_remove_outliers = false;
    int map_outlier_nb_neighbors = 20;
    double map_outlier_stddev_mul = 2.0;

    // global point residual config (later steps will use)
    bool gpr_enable = true;
    int gpr_max_points = 1200;
    double gpr_voxel_leaf = 0.5;
    int gpr_plane_min_neighbors = 10;
    double gpr_max_point2plane_dist = 0.6;
    double gpr_sigma_m = 0.06;
    std::string gpr_robust_kernel = "huber";
    double gpr_huber_delta = 0.5;
    double gpr_cauchy_c = 0.5;

    // ROI (for later use when querying global map)
    double roi_radius = 60.0;
};
LocalizationParams g_loc;

// 子图缓存（世界坐标点），用于从 ikdtree_global 回收
struct SubmapFrame {
    PointCloudXYZI::Ptr pts_world; // 使用 PCL 的 Ptr
    size_t num_points = 0;
};
std::deque<SubmapFrame> g_global_submap_frames; // 最近 K 帧加入到 ikdtree_global 的点
int g_global_submap_K = 3;                      // 叠加帧数（可用参数覆盖）
float g_global_submap_leaf = 0.5f;              // 子图体素叶大小
bool g_global_tree_ready = false;

pcl::PointCloud<PointType>::Ptr g_global_map_raw(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr g_global_map_ds(new pcl::PointCloud<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr g_kdtree_global(new pcl::KdTreeFLANN<PointType>());

// 标记是否加载成功
bool g_global_map_ready = false;

// 一次性全局初始对齐
bool g_global_align_done = false;
bool g_global_align_running = false;
ros::Publisher pubGlobalAligned; // 可视化对齐后的源云

namespace {
struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        return (static_cast<size_t>(k.x) * 73856093u) ^
               (static_cast<size_t>(k.y) * 19349663u) ^
               (static_cast<size_t>(k.z) * 83492791u);
    }
};

// 全局体素键构造函数
inline VoxelKey make_key(const Eigen::Vector3d& p, float voxel_leaf) {
    return VoxelKey{
        static_cast<int>(std::floor(p.x() / voxel_leaf)),
        static_cast<int>(std::floor(p.y() / voxel_leaf)),
        static_cast<int>(std::floor(p.z() / voxel_leaf))
    };
}
} // namespace

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

static inline void TransformBodyToWorld(const PointCloudXYZI::Ptr& in_body,
                                        PointCloudXYZI::Ptr& out_world,
                                        const state_ikfom& s)
{
    out_world.reset(new PointCloudXYZI());
    if (!in_body || in_body->empty()) {
        out_world->clear();
        out_world->width = 0; out_world->height = 1; out_world->is_dense = true;
        return;
    }

    out_world->resize(in_body->size());
    for (size_t i = 0; i < in_body->size(); ++i) {
        const auto& pi = in_body->points[i];
        PointType po;
        V3D p_body(pi.x, pi.y, pi.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        po.x = p_global(0);
        po.y = p_global(1);
        po.z = p_global(2);
        po.intensity = pi.intensity;
        out_world->points[i] = po;
    }
    out_world->width = out_world->points.size();
    out_world->height = 1;
    out_world->is_dense = true;
}

static inline PointCloudXYZI::Ptr VoxelDownXYZI(const PointCloudXYZI::Ptr& cloud, float leaf)
{
    PointCloudXYZI::Ptr ds(new PointCloudXYZI());
    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(leaf, leaf, leaf);
    vg.setInputCloud(cloud);
    vg.filter(*ds);
    return ds;
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

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();


        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        if(lidar_type == MARSIM)
            lidar_end_time = meas.lidar_beg_time;

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
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

// ========== Multi-frame GPR cache ==========
struct GPRFrame {
    PointCloudXYZI::Ptr pts; // 降采样后的雷达系点（与 feats_down_body 语义一致）
    double stamp = 0.0;
};

std::deque<GPRFrame> g_gpr_frames;          // 最近 K 帧缓存
int g_gpr_stack_k = 3;                       // 叠加帧数，默认 3，可参数覆盖
size_t g_gpr_max_cache_points = 40000;       // 缓存点数上限，避免内存膨胀

static inline Eigen::Matrix4f poseRPYToMat4f(float x,float y,float z,float roll,float pitch,float yaw){
    Eigen::Matrix3f R =
        Eigen::AngleAxisf(yaw,   Eigen::Vector3f::UnitZ()).toRotationMatrix() *
        Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()).toRotationMatrix() *
        Eigen::AngleAxisf(roll,  Eigen::Vector3f::UnitX()).toRotationMatrix();
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = Eigen::Vector3f(x,y,z);
    return T;
}

pcl::PointCloud<PointType>::Ptr CropROI(const pcl::PointCloud<PointType>::Ptr& cloud,
                                        const Eigen::Vector3f& center, float radius)
{
    pcl::PointCloud<PointType>::Ptr out(new pcl::PointCloud<PointType>());
    out->reserve(10000);
    float r2 = radius*radius;
    for (const auto& p : cloud->points) {
        float dx = p.x - center.x();
        float dy = p.y - center.y();
        float dz = p.z - center.z();
        if (dx*dx + dy*dy + dz*dz <= r2) out->push_back(p);
    }
    out->width = out->points.size();
    out->height = 1;
    out->is_dense = true;
    return out;
}

pcl::PointCloud<PointType>::Ptr VoxelDown(const pcl::PointCloud<PointType>::Ptr& cloud, float leaf)
{
    pcl::PointCloud<PointType>::Ptr ds(new pcl::PointCloud<PointType>());
    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(leaf, leaf, leaf);
    vg.setInputCloud(cloud);
    vg.filter(*ds);
    return ds;
}

struct AlignResult {
    bool ok = false;
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity(); // map <- lidar
    double score = 1e9;
    std::string method = "none";
};

AlignResult RunNDTGICPOnce(const pcl::PointCloud<PointType>::Ptr& source_lidar_frame,
                           const Eigen::Matrix4f& T_guess_map_lidar,
                           const Eigen::Vector3f& guess_center_map,
                           double roi_radius,
                           const std::vector<double>& ndt_resolutions,
                           const std::vector<int>& ndt_max_iterations,
                           double ndt_step_size, double ndt_trans_eps, double ndt_score_clip,
                           bool try_gicp, int gicp_max_iterations, double gicp_max_corr_dist,
                           double gicp_trans_eps, double gicp_fit_eps, int gicp_corr_randomness, bool gicp_use_recip, float roi_max)
{
    AlignResult res;
    
    if (!g_global_map_ready || !g_global_map_ds || g_global_map_ds->empty()) {
        ROS_WARN("[global_align] Global map empty.");
        return res;
    }
    if (!source_lidar_frame || source_lidar_frame->empty()) {
        ROS_WARN("[global_align] Source frame empty.");
        return res;
    }

    if (!g_global_map_ready || g_global_map_ds->empty() || !source_lidar_frame || source_lidar_frame->empty()) {
        ROS_WARN("[global_align] Invalid inputs: global map or source empty.");
        return res;
    }

    // 1) 目标 ROI
    float rr = static_cast<float>(roi_radius);
    pcl::PointCloud<PointType>::Ptr target_roi = CropROI(g_global_map_ds, guess_center_map, rr);
    if (target_roi->size() < 500) {
        ROS_WARN("[global_align] ROI too small (%zu), expanding...", target_roi->size());
        rr = std::min<float>(rr * 2.0f, roi_max);
        target_roi = CropROI(g_global_map_ds, guess_center_map, rr);
    }
    if (target_roi->size() < 500) {
        ROS_WARN("[global_align] ROI still too small, abort.");
        return res;
    }

    // 2) 下采样
    auto src_ds = VoxelDown(source_lidar_frame, 0.4f);
    auto tgt_ds = VoxelDown(target_roi,         0.4f);

    // 3) NDT 多分辨率递进
    Eigen::Matrix4f T = T_guess_map_lidar;
    double best_score = 1e9;
    bool ndt_ok = false;
    if (!ndt_resolutions.empty() && ndt_resolutions.size() == ndt_max_iterations.size()) {
        auto run_ndt_pyramid = [&](Eigen::Matrix4f& T,
                           const pcl::PointCloud<PointType>::Ptr& tgt_ds,
                           const pcl::PointCloud<PointType>::Ptr& src_ds,
                           bool& ndt_ok,
                           double& best_score) -> void
        {
            ndt_ok = false;
            best_score = std::numeric_limits<double>::infinity();

            // 防御性检查
            if (!tgt_ds) { ROS_ERROR("[global_align][NDT] tgt_ds is null."); return; }
            if (!src_ds) { ROS_ERROR("[global_align][NDT] src_ds is null."); return; }
            if (tgt_ds->empty()) { ROS_WARN("[global_align][NDT] target cloud empty."); return; }
            if (src_ds->empty()) { ROS_WARN("[global_align][NDT] source cloud empty."); return; }
            if (!T.allFinite()) {
                ROS_WARN("[global_align][NDT] initial T non-finite, reset Identity.");
                T = Eigen::Matrix4f::Identity();
            }

            // Preconditions already checked: tgt_ds/src_ds non-null and non-empty.
// Make deep copies to avoid concurrent modification and remove NaNs.
pcl::PointCloud<PointType>::Ptr tgt_local(new pcl::PointCloud<PointType>(*tgt_ds));
pcl::PointCloud<PointType>::Ptr src_local(new pcl::PointCloud<PointType>(*src_ds));
{
    std::vector<int> idx;
    pcl::removeNaNFromPointCloud(*tgt_local, *tgt_local, idx);
    pcl::removeNaNFromPointCloud(*src_local, *src_local, idx);
    tgt_local->is_dense = true;
    src_local->is_dense = true;
}

ndt_ok = false;
best_score = std::numeric_limits<double>::infinity();

for (size_t i = 0; i < ndt_resolutions.size(); ++i) {
    const double reso = ndt_resolutions[i];
    const int max_it = ndt_max_iterations[i];

    if (tgt_local->empty() || src_local->empty()) {
        ROS_WARN("[global_align][NDT] empty cloud at level %.2f, break.", reso);
        break;
    }
    if (!T.allFinite()) {
        ROS_WARN("[global_align][NDT] T non-finite before level %.2f, break.", reso);
        break;
    }

    pcl::NormalDistributionsTransform<PointType, PointType> ndt;
    ndt.setResolution(reso);
    ndt.setStepSize(ndt_step_size);
    ndt.setTransformationEpsilon(ndt_trans_eps);
    ndt.setMaximumIterations(max_it);
    ndt.setInputTarget(tgt_local);
    ndt.setInputSource(src_local);

    pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>());

    // Align guarded
    try {
        ndt.align(*aligned, T);
    } catch (const std::exception& e) {
        ROS_ERROR("[global_align][NDT] exception at res=%.2f: %s", reso, e.what());
        break;
    }

    if (!ndt.hasConverged()) {
        ROS_WARN("[global_align][NDT] level %.2f not converged.", reso);
        continue;
    }

    // Fetch transformation first; skip fitness score to avoid potential segfault
    Eigen::Matrix4f T_new = ndt.getFinalTransformation();
    if (!T_new.allFinite()) {
        ROS_ERROR("[global_align][NDT] T_new non-finite at res=%.2f", reso);
        break;
    }
    T = T_new;

    // Optional: basic sanity check on aligned cloud
    size_t aligned_sz = aligned ? aligned->size() : 0;
    if (aligned_sz < 3) {
        ROS_WARN("[global_align][NDT] aligned cloud too small (%zu) at res=%.2f", aligned_sz, reso);
    }

    // Temporarily skip getFitnessScore to avoid crash
    double s = 0.0; // placeholder score
    // If you want to try safer variant (often more stable): s = ndt.getFitnessScore(1.0);

    best_score = s;
    ndt_ok = true;
    ROS_INFO("[global_align][NDT] res=%.2f it=%d (score skipped)", reso, max_it);
}
ROS_DEBUG("[global_align][NDT] pyramid done. ok=%d best=%.3f", (int)ndt_ok, best_score);
        };

        // 使用
        bool ndt_ok = false;
        double best_score = std::numeric_limits<double>::infinity();
        run_ndt_pyramid(T, tgt_ds, src_ds, ndt_ok, best_score);
    }

    if (ndt_ok) {
        res.ok = true;
        res.T = T;
        res.score = best_score;
        res.method = "ndt";
        return res;
    }

    // 4) GICP 兜底
    if (try_gicp) {
        pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
        gicp.setMaximumIterations(gicp_max_iterations);
        gicp.setTransformationEpsilon(gicp_trans_eps);
        gicp.setEuclideanFitnessEpsilon(gicp_fit_eps);
        gicp.setMaxCorrespondenceDistance(gicp_max_corr_dist);
#ifdef PCL_VERSION_COMPARE
        // 版本差异处理略
#endif
        gicp.setInputTarget(tgt_ds);
        gicp.setInputSource(src_ds);
        pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>());
        gicp.align(*aligned, T);
        if (gicp.hasConverged()) {
            res.ok = true;
            res.T = gicp.getFinalTransformation();
            res.score = gicp.getFitnessScore();
            res.method = "gicp";
            ROS_INFO("[global_align][GICP] score=%.3f", res.score);
        } else {
            ROS_WARN("[global_align][GICP] failed.");
        }
    }

    return res;
}

bool LoadGlobalMapAndBuildIndex()
{
    if (g_loc.map_file_path.empty()) {
        ROS_WARN("[localization] map_file_path is empty, skip loading global map.");
        return false;
    }
    ROS_INFO("[localization] Loading global map: %s", g_loc.map_file_path.c_str());
    if (pcl::io::loadPCDFile<PointType>(g_loc.map_file_path, *g_global_map_raw) != 0) {
        ROS_ERROR("[localization] Failed to load PCD: %s", g_loc.map_file_path.c_str());
        return false;
    }
    // 体素降采样
    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(g_loc.map_voxel_leaf, g_loc.map_voxel_leaf, g_loc.map_voxel_leaf);
    vg.setInputCloud(g_global_map_raw);
    vg.filter(*g_global_map_ds);

    // 可选统计滤波
    if (g_loc.map_remove_outliers) {
        pcl::StatisticalOutlierRemoval<PointType> sor;
        sor.setInputCloud(g_global_map_ds);
        sor.setMeanK(g_loc.map_outlier_nb_neighbors);
        sor.setStddevMulThresh(g_loc.map_outlier_stddev_mul);
        pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
        sor.filter(*filtered);
        *g_global_map_ds = *filtered;
    }

    if (g_global_map_ds->empty()) {
        ROS_ERROR("[localization] Global map is empty after preprocessing.");
        return false;
    }

    // 构建 KD-Tree
    g_kdtree_global->setInputCloud(g_global_map_ds);

    ROS_INFO("[localization] Global map loaded: raw=%zu, ds=%zu, leaf=%.2f",
             g_global_map_raw->size(), g_global_map_ds->size(), g_loc.map_voxel_leaf);
    return true;
}

bool BuildGlobalIKDTreeFromMap()
{
    if (!g_global_map_ready || !g_global_map_ds || g_global_map_ds->empty()) {
        ROS_WARN("[global_tree] Global map not ready, skip build ikdtree_global.");
        return false;
    }
    // 强力降采样（可选，防止树过大；这里沿用 g_loc.map_voxel_leaf，必要可再大一点）
    pcl::PointCloud<PointType>::Ptr map_ds2(new pcl::PointCloud<PointType>());
    pcl::VoxelGrid<PointType> vg;
    float leaf = std::max<float>(0.4f, (float)g_loc.map_voxel_leaf);
    vg.setLeafSize(leaf, leaf, leaf);
    vg.setInputCloud(g_global_map_ds);
    vg.filter(*map_ds2);

    // 构建 ikd-tree（一次性）
    ikdtree_global.set_downsample_param(leaf);  // 下采参数只影响增量添加；Build 也会用体素中心策略
    ikdtree_global.Build(map_ds2->points);
    g_global_tree_ready = true;

    ROS_INFO("[global_tree] ikdtree_global built. global map pts=%zu, leaf=%.2f",
             map_ds2->size(), leaf);
    return true;
}

void GlobalTree_UpdateSubmap_WithCurrentFrames(const state_ikfom& s)
{
    if (!g_global_tree_ready) return;

    // 1) 准备“当前多帧子图（雷达系）”的合并源
    // 你已有 g_gpr_frames（最近 K 帧雷达系点），这里使用它；也可以只用本帧 feats_down_body
    // 将这些点分别变换到世界系并合并
    PointCloudXYZI::Ptr merged_world(new PointCloudXYZI());
    merged_world->reserve(30000);

    int used_frames = 0;
    for (int fi = (int)g_gpr_frames.size() - 1; fi >= 0 && used_frames < g_global_submap_K; --fi) {
        const auto& fr = g_gpr_frames[fi];
        if (!fr.pts || fr.pts->empty()) continue;

        PointCloudXYZI::Ptr world_pts;
        TransformBodyToWorld(fr.pts, world_pts, s);
        if (world_pts && !world_pts->empty()) {
            *merged_world += *world_pts;
            ++used_frames;
        }
    }
    if (merged_world->empty()) return;

    // 2) 体素降采样，控制子图规模
    auto submap_ds = VoxelDownXYZI(merged_world, g_global_submap_leaf);

    // 3) 删除旧的子图帧（从 ikdtree_global 中回收），保持队列不超过 K-1（因为本次会添加1帧合并子图）
    // 这里我们把“合并子图”当作一个帧入队，之前入队的全部作为“上一帧子图”，所以最多保留 K-1 个历史合并子图条目
    while ((int)g_global_submap_frames.size() >= (g_global_submap_K - 1) && !g_global_submap_frames.empty())
    {
        auto old = g_global_submap_frames.front();
        g_global_submap_frames.pop_front();
        if (old.pts_world && !old.pts_world->empty()) {
            // 删除旧子图点
            // 优先使用 Delete_Points（你的 ikd-tree 若不支持，请改为 Delete_Point_Boxes）
            ikdtree_global.Delete_Points(old.pts_world->points);
            // ROS_INFO("[global_tree] deleted old submap points: %d", deln);
        }
    }

    // 4) 将本次“合并子图”点加入全局树
    if (!submap_ds->empty()) {
        int addn = ikdtree_global.Add_Points(submap_ds->points, true);
        (void)addn;
        // ROS_INFO("[global_tree] add submap points: %d (ds=%zu)", addn, submap_ds->size());
        // 入队以便下一次回收
        SubmapFrame rec;
        rec.pts_world.reset(new PointCloudXYZI());
        *(rec.pts_world) = *submap_ds;
        rec.num_points = submap_ds->size();
        g_global_submap_frames.push_back(std::move(rec));
    }
}


PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
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
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
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
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
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

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void publish_global_map(const ros::Publisher& pub)
{
    if (!g_global_map_ready || g_global_map_ds->empty()) return;
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*g_global_map_ds, msg);
    msg.header.frame_id = "camera_init";
    msg.header.stamp = ros::Time::now();
    pub.publish(msg);
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    const float voxel_leaf = static_cast<float>(g_loc.gpr_voxel_leaf);
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

    // ========== GPR multi-frame cache update (once per frame) ==========
    {
        static double last_cached_lidar_time = -1.0;
        if (fabs(lidar_end_time - last_cached_lidar_time) > 1e-9 && feats_down_body && !feats_down_body->empty()) {
            PointCloudXYZI::Ptr copy(new PointCloudXYZI());
            *copy = *feats_down_body;

            g_gpr_frames.push_back(GPRFrame{copy, lidar_end_time});
            last_cached_lidar_time = lidar_end_time;

            while ((int)g_gpr_frames.size() > std::max(1, g_gpr_stack_k)) {
                g_gpr_frames.pop_front();
            }
            size_t total_pts = 0;
            for (auto &f : g_gpr_frames) total_pts += (f.pts ? f.pts->size() : 0);
            while (!g_gpr_frames.empty() && total_pts > g_gpr_max_cache_points) {
                total_pts -= (g_gpr_frames.front().pts ? g_gpr_frames.front().pts->size() : 0);
                g_gpr_frames.pop_front();
            }
        }
    }

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        // transform to world frame
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
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s_w = 1 - 0.9f * fabs(pd2) / sqrt(p_body.norm());

            if (s_w > 0.9f)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = fabs(pd2);
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

    /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
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
        point_crossmat << SKEW_SYM_MATRX(point_this);

        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        V3D C(s.rot.conjugate() * norm_vec);
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

        ekfom_data.h(i) = -norm_p.intensity;
    }

    // ========== BEGIN: Global Point Residual (GPR) append using ikdtree_global ==========
    do {
        if (!(g_loc.gpr_enable && g_global_tree_ready)) {
            break;
        }

        // 鲁棒核
        auto robust_weight = [&](double r)->double {
            if (g_loc.gpr_robust_kernel == "huber") {
                const double d = std::max(1e-12, g_loc.gpr_huber_delta);
                const double ar = std::abs(r);
                return (ar <= d) ? 1.0 : (d / ar);
            } else if (g_loc.gpr_robust_kernel == "cauchy") {
                const double c = std::max(1e-12, g_loc.gpr_cauchy_c);
                const double x = r / c;
                return 1.0 / (1.0 + x*x);
            } else {
                return 1.0;
            }
        };

        // 参数
        const int min_k = std::max(g_loc.gpr_plane_min_neighbors, 8);
        const double dist_thresh = std::max(1e-6, g_loc.gpr_max_point2plane_dist);
        const double plane_quality_max = 0.15; // 与本地相近
        const int max_points = std::max(1, g_loc.gpr_max_points);

        std::unordered_set<VoxelKey, VoxelKeyHash> used_voxels;
        used_voxels.reserve(size_t(max_points * 2));

        std::vector<Eigen::Matrix<double,1,12>> H_globals;
        std::vector<double> h_globals;
        H_globals.reserve(size_t(std::min<int>(max_points, 4096)));
        h_globals.reserve(size_t(std::min<int>(max_points, 4096)));

        PointVector neigh; neigh.reserve(64);
        std::vector<float> nn_sqdist; nn_sqdist.reserve(64);

        // 从新到旧遍历多帧缓存
        for (int fi = int(g_gpr_frames.size()) - 1; fi >= 0; --fi) {
            const auto& frame = g_gpr_frames[fi];
            if (!frame.pts || frame.pts->empty()) continue;

            const int Nf = frame.pts->size();
            for (int i = 0; i < Nf; ++i) {
                const auto& pb = frame.pts->points[i];

                // p_world = R * (R_LI * p_body + T_LI) + t
                Eigen::Vector3d p_body(pb.x, pb.y, pb.z);
                Eigen::Vector3d p_this = s.offset_R_L_I * p_body + s.offset_T_L_I;
                Eigen::Vector3d p_world = s.rot.toRotationMatrix() * p_this + s.pos;

                VoxelKey key = make_key(p_world, voxel_leaf);
                if (used_voxels.find(key) != used_voxels.end()) continue;

                // 用 ikdtree_global 查找邻域点
                neigh.clear();
                nn_sqdist.clear();
                PointType query; query.x = float(p_world.x()); query.y = float(p_world.y()); query.z = float(p_world.z());
                ikdtree_global.Nearest_Search(query, min_k, neigh, nn_sqdist);
                if ((int)neigh.size() < min_k) continue;

                // 用与本地一致的 esti_plane(PointVector) 拟合平面
                VF(4) pabcd;
                if (!esti_plane(pabcd, neigh, 0.1f)) continue;

                // 法向与点心
                Eigen::Vector3d n(pabcd(0), pabcd(1), pabcd(2));
                n.normalize();
                // 通过平面方程 ax+by+cz+d=0 的 d 来 recover 平面中心一点 c: c = -d * n (最近点到原点)
                Eigen::Vector3d c = -pabcd(3) * n;

                const double r = n.dot(p_world - c); // 点到平面距离（有符号）
                if (std::abs(r) > dist_thresh) continue;

                // 雅可比
                Eigen::Vector3d Cvec = s.rot.conjugate() * n; // R^T n
                Eigen::Matrix3d point_cross;
                point_cross <<     0, -p_this.z(),  p_this.y(),
                                p_this.z(),     0, -p_this.x(),
                               -p_this.y(),  p_this.x(),     0;
                Eigen::Vector3d Avec = point_cross * Cvec;

                Eigen::Matrix<double,1,12> Hrow;
                if (extrinsic_est_en) {
                    Eigen::Matrix3d point_be_cross;
                    point_be_cross <<    0, -p_body.z(),  p_body.y(),
                                       p_body.z(),     0, -p_body.x(),
                                      -p_body.y(),  p_body.x(),     0;
                    Eigen::Vector3d Bvec = point_be_cross * s.offset_R_L_I.conjugate() * Cvec;
                    Hrow << n.x(), n.y(), n.z(),
                            Avec.x(), Avec.y(), Avec.z(),
                            Bvec.x(), Bvec.y(), Bvec.z(),
                            Cvec.x(), Cvec.y(), Cvec.z();
                } else {
                    Hrow << n.x(), n.y(), n.z(),
                            Avec.x(), Avec.y(), Avec.z(),
                            0.0, 0.0, 0.0,
                            Cvec.x(), Cvec.y(), Cvec.z();
                }

                double w = robust_weight(r);
                w = std::max(1e-6, std::min(1.0, w));
                const double sqrt_w = std::sqrt(w);
                Hrow *= sqrt_w;
                const double h_val = -r * sqrt_w;

                H_globals.push_back(Hrow);
                h_globals.push_back(h_val);
                used_voxels.insert(key);

                if ((int)H_globals.size() >= max_points) break;
            }
            if ((int)H_globals.size() >= max_points) break;
        }

        // 追加到 ekfom_data（叠加在本地残差之后）
        if (!H_globals.empty()) {
            const int rows_local = ekfom_data.h_x.rows();
            const int rows_global = (int)H_globals.size();
            const int cols = ekfom_data.h_x.cols(); // 12

            Eigen::MatrixXd H_new(rows_local + rows_global, cols);
            if (rows_local > 0) H_new.topRows(rows_local) = ekfom_data.h_x;
            for (int i = 0; i < rows_global; ++i) {
                H_new.row(rows_local + i) = H_globals[i];
            }
            ekfom_data.h_x.swap(H_new);

            ekfom_data.h.resize(rows_local + rows_global);
            for (int i = 0; i < rows_global; ++i) {
                ekfom_data.h(rows_local + i) = h_globals[i];
            }

            ekfom_data.valid = true; // 即便本地为空，只要全局有残差，也可更新
            // ROS_INFO("[GPR] appended %d global residuals (local=%d).", rows_global, rows_local);
        }
    } while (false);
    // ========== END: Global Point Residual (GPR) append using ikdtree_global ==========

    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

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

    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    // --- global_align params ---
    std::vector<double> ndt_resolutions;  // e.g. [2.0,1.0,0.5]
    std::vector<int>    ndt_max_iterations; // e.g. [35,40,50]
    double ndt_step_size = 0.5, ndt_trans_eps = 1e-3, ndt_score_clip = 3.0;
    bool ndt_use_imu_rp_init = true;

    bool gicp_enable = true; 
    int gicp_max_iterations = 40, gicp_corr_randomness = 20;
    double gicp_max_corr_dist = 1.5, gicp_trans_eps = 1e-4, gicp_fit_eps = 1e-5;
    bool gicp_use_recip = true;

    double ga_roi_radius = g_loc.roi_radius; // 已读取
    double ga_fitness_thresh = 1.5;
    double ga_fallback_expand = 1.5, ga_roi_max = 120.0;

    nh.getParam("localization/ndt/resolutions", ndt_resolutions);
    nh.getParam("localization/ndt/max_iterations", ndt_max_iterations);
    nh.param<double>("localization/ndt/step_size", ndt_step_size, 0.5);
    nh.param<double>("localization/ndt/trans_eps", ndt_trans_eps, 1e-3);
    nh.param<double>("localization/ndt/score_clip", ndt_score_clip, 3.0);
    nh.param<bool>("localization/ndt/use_imu_roll_pitch_init", ndt_use_imu_rp_init, true);

    nh.param<bool>("localization/gicp/enable", gicp_enable, true);
    nh.param<int>("localization/gicp/max_iterations", gicp_max_iterations, 40);
    nh.param<double>("localization/gicp/max_corr_dist", gicp_max_corr_dist, 1.5);
    nh.param<double>("localization/gicp/trans_eps", gicp_trans_eps, 1e-4);
    nh.param<double>("localization/gicp/euclidean_fitness_epsilon", gicp_fit_eps, 1e-5);
    nh.param<int>("localization/gicp/correspondence_randomness", gicp_corr_randomness, 20);
    nh.param<bool>("localization/gicp/use_reciprocal", gicp_use_recip, true);

    nh.param<double>("localization/global_align/fitness_thresh", ga_fitness_thresh, 1.5);
    nh.param<double>("localization/global_align/fallback_expand_roi", ga_fallback_expand, 1.5);
    nh.param<double>("localization/global_align/max_roi_radius", ga_roi_max, 120.0);

    nh.param<int>("localization/global_tree/submap_K", g_global_submap_K, 3);
    double global_submap_leaf = 0.5;
    nh.param<double>("localization/global_tree/submap_leaf", global_submap_leaf, 0.5);
    g_global_submap_leaf = static_cast<float>(global_submap_leaf);

    nh.param<int>("localization/global_point_residual/stack_k", g_gpr_stack_k, 3);
    
    // 注意：Noetic 的 param 不支持 size_t，先读 int 再转换
    int gpr_max_cache_points_int = 40000;
    nh.param<int>("localization/global_point_residual/max_cache_points", gpr_max_cache_points_int, 40000);
    if (gpr_max_cache_points_int < 0) gpr_max_cache_points_int = 0;
    g_gpr_max_cache_points = static_cast<size_t>(gpr_max_cache_points_int);

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;


    ros::Publisher pubGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("/global_map", 1, true);
    /* Load global map */
    g_global_map_ready = LoadGlobalMapAndBuildIndex();
    if (g_global_map_ready) {
        BuildGlobalIKDTreeFromMap(); // 新增：用全局地图一次性构建全局树
        publish_global_map(pubGlobalMap);
        ROS_INFO("[localization] Global map published to /global_map.");
    } else {
        ROS_WARN("[localization] Global map not ready, will run without global residuals.");
    }

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odom_lio", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);

    pubGlobalAligned = nh.advertise<sensor_msgs::PointCloud2>("/global_align/aligned_source", 1, false);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();

            GlobalTree_UpdateSubmap_WithCurrentFrames(state_point);
            /*** initialize the map kdtree ***/
            /*** initialize the map kdtree ***/
            if (ikdtree.Root_Node == nullptr)
            {
                if (feats_down_size > 5)
                {
                    bool curr_has_points = (feats_down_body && !feats_down_body->empty());
                    // --------- 初始对齐（仅执行一次），在建树前进行 ---------
                    if (!g_global_align_done && g_global_map_ready)
                    {
                        if (!curr_has_points) {
                            ROS_WARN("[global_align] skip: current frame has no points.");
                        } else {
                            // 源点：当前帧的去畸变、已降采样点（雷达坐标系）
                            pcl::PointCloud<PointType>::Ptr src_curr(new pcl::PointCloud<PointType>());
                            *src_curr = *feats_down_body;

                            // 初值：IMU的roll/pitch（若可用），yaw=0，平移=0
                            double r_init = 0.0, p_init = 0.0, y_init = 0.0;
                            if (ndt_use_imu_rp_init) {
                                Eigen::Vector3d rpy = SO3ToEuler(state_point.rot); // 你已有 SO3ToEuler
                                r_init = rpy[0];
                                p_init = rpy[1];
                                // yaw 初值可置 0 或用 rpy[2]，建议 0，避免引导错误
                            }
                            Eigen::Matrix4f T_guess = poseRPYToMat4f(0.0, 0.0, 0.0, r_init, p_init, y_init);
                            Eigen::Vector3f roi_center = T_guess.block<3,1>(0,3);

                            // 运行 NDT→GICP
                            AlignResult ar = RunNDTGICPOnce(
                                src_curr,
                                T_guess,
                                roi_center,
                                ga_roi_radius,
                                ndt_resolutions,
                                ndt_max_iterations,
                                ndt_step_size,
                                ndt_trans_eps,
                                ndt_score_clip,
                                gicp_enable,
                                gicp_max_iterations,
                                gicp_max_corr_dist,
                                gicp_trans_eps,
                                gicp_fit_eps,
                                gicp_corr_randomness,
                                gicp_use_recip,
                                (float)ga_roi_max
                            );

                            if (ar.ok && ar.score < ga_fitness_thresh) {
                                Eigen::Matrix3d Rm = ar.T.block<3,3>(0,0).cast<double>();
                                Eigen::Vector3d tm = ar.T.block<3,1>(0,3).cast<double>();

                                const Eigen::Matrix3d RL_I = state_point.offset_R_L_I.toRotationMatrix();
                                const Eigen::Vector3d tL_I = state_point.offset_T_L_I;

                                Eigen::Matrix3d RwI = Rm * RL_I.transpose();
                                Eigen::Vector3d twI = tm - RwI * tL_I;

                                if (!RwI.allFinite() || !twI.allFinite()) {
                                    ROS_ERROR("[global_align] RwI/twI invalid (nan/inf), abort applying alignment.");
                                } else {
                                    state_point.pos = twI;

                                    Eigen::Quaterniond q(RwI);
                                    q.normalize();
                                    Eigen::Vector4d c; c << q.x(), q.y(), q.z(), q.w();
                                    state_point.rot.coeffs() = c;

                                    try {
                                        kf.change_x(state_point);
                                    } catch (...) {
                                        ROS_ERROR("[global_align] kf.change_x threw exception. Will continue without change.");
                                    }
                                }

                                if (pubGlobalAligned) {
                                    pcl::PointCloud<PointType>::Ptr src_curr(new pcl::PointCloud<PointType>());
                                    if (feats_down_body) *src_curr = *feats_down_body;
                                    if (src_curr->empty()) {
                                        ROS_WARN("src_curr empty, skip transform.");
                                        continue;
                                    }
                                    pcl::PointCloud<PointType>::Ptr src_aligned(new pcl::PointCloud<PointType>());
                                    try {
                                        pcl::transformPointCloud(*src_curr, *src_aligned, ar.T);
                                        if (!ar.T.allFinite()) {
                                            ROS_ERROR("[global_align] Transformation contains nan/inf, skip.");
                                            continue;
                                        }
                                        if (!src_aligned->empty()) {
                                            sensor_msgs::PointCloud2 msg;
                                            pcl::toROSMsg(*src_aligned, msg);
                                            msg.header.frame_id = "camera_init";
                                            msg.header.stamp = ros::Time::now();
                                            pubGlobalAligned.publish(msg);
                                        }
                                    } catch (const std::exception& e) {
                                        ROS_WARN("[global_align] transformPointCloud failed: %s", e.what());
                                    }
                                }

                                g_global_align_done = true;
                                ROS_INFO("[global_align] initial pose set by %s. fitness=%.3f", ar.method.c_str(), ar.score);
                            } else {
                                ROS_WARN("[global_align] initial alignment failed or poor. ok=%d score=%.3f (th=%.3f)",
                                        int(ar.ok), ar.score, ga_fitness_thresh);
                            }
                        }
                    }
                    // --------- 初始对齐结束 ---------

                    // 用（可能已重置的）状态，将本帧点转换到全局（世界）坐标，再建树
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->clear();
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; ++i) {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }

                    if (!feats_down_world->empty()) {
                        ikdtree.Build(feats_down_world->points);
                    } else {
                        ROS_WARN("[ikdtree] feats_down_world empty at build.");
                    }
                }
                continue; // 保持原逻辑
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
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

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
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
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
