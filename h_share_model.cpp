void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    const double t_match_begin = omp_get_wtime();

    // 清空局部容器
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

    // 全局容器
    static PointCloudXYZI::Ptr laserCloudOri_global(new PointCloudXYZI(100000, 1));
    static PointCloudXYZI::Ptr corr_normvect_global(new PointCloudXYZI(100000, 1));

    // 选择标记与残差缓存（局部 + 全局）
    static bool  point_selected_surf_local[100000];
    static bool  point_selected_surf_global[100000];
    static float res_last_local[100000];
    static float res_last_global[100000];

    memset(point_selected_surf_local,  false, sizeof(point_selected_surf_local));
    memset(point_selected_surf_global, false, sizeof(point_selected_surf_global));

    // 本地默认参数
    const float  dist_th_local   = 5.0f;
    const float  dist_th_global  = 12.0f;
    const float  plane_eps       = 0.1f;
    const float  s_gate          = 0.7f;
    const float  r_abs_max_global = (float)g_loc.gpr_max_point2plane_dist;
    const double alpha_global    = 3.0;
    const double w_global_base   = 2.0;
    const double huber_delta     = 0.5;
    const bool   use_global_residual = g_loc.gpr_enable && g_global_ikdtree_ready;
    const int    global_max_keep     = (g_loc.gpr_max_points > 0) ? g_loc.gpr_max_points : 4000;

    const float tau_global_better = 0.85f;

    static int s_boost_frames = 0;
    double boost_gain = (s_boost_frames > 0) ? 2.0 : 1.0;

    // === 新增：用于判定是否持续清空 local 树的状态（带滞回） ===
    static bool keep_clearing_local_tree = false; // true 表示本帧结束要清空 local 树
    // 设定滞回阈值：global 残差 <= good_thresh 视为“好”，> bad_thresh 视为“差”
    const double good_thresh = std::max(0.15, 0.35 * (double)g_loc.gpr_max_point2plane_dist);
    const double bad_thresh  = std::max(0.2 , 0.60 * (double)g_loc.gpr_max_point2plane_dist);

    // 容器大小
    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);

    auto huber_weight = [&](double r_abs, double delta)->double {
        if (delta <= 0) return 1.0;
        return (r_abs <= delta) ? 1.0 : (delta / r_abs);
    };

    // 新增：统计“将要进入观测的全局残差”的累加器
    double sum_rg_kept = 0.0;
    int    rg_kept_cnt = 0;

    // 第一轮：坐标转换与邻域搜索、平面估计
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        // body -> world
        const V3D p_body(point_body.x, point_body.y, point_body.z);
        const V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        // 局部匹配
        {
            if (!g_local_tree_empty || true) {
                std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
                auto &points_near_local = Nearest_Points[i];

                ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near_local, pointSearchSqDis);

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
                            point_selected_surf_local[i] = true;
                            normvec->points[i].x = pabcd(0);
                            normvec->points[i].y = pabcd(1);
                            normvec->points[i].z = pabcd(2);
                            normvec->points[i].intensity = pd2;
                            res_last_local[i] = std::fabs(pd2);
                        }
                    }
                }
            }
            // 如果 g_local_tree_empty 为真，这一帧直接跳过局部匹配
        }

        // 全局匹配
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
                        point_selected_surf_global[i] = true;
                        corr_normvect_global->points[i].x = pabcd_g(0);
                        corr_normvect_global->points[i].y = pabcd_g(1);
                        corr_normvect_global->points[i].z = pabcd_g(2);
                        corr_normvect_global->points[i].intensity = pd2g;
                        res_last_global[i] = std::fabs(pd2g);
                    }
                }
            }
        }
    }

    // 决策阶段：全局显著更好 → 清除该索引的 local 观测
    if (use_global_residual)
    {
        int replaced_cnt = 0;
        double sum_rg = 0.0, sum_rl = 0.0;

        for (int i = 0; i < feats_down_size; i++)
        {
            if (!point_selected_surf_global[i]) continue;

            const float r_g = res_last_global[i];
            if (point_selected_surf_local[i])
            {
                const float r_l = res_last_local[i];
                if (r_g <= tau_global_better * r_l)
                {
                    point_selected_surf_local[i] = false;
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
                              (float)(sum_rg / std::max(1,replaced_cnt)),
                              (float)(sum_rl / std::max(1,replaced_cnt)));
        }
    }

    // 第二轮：收集到观测集合
    effct_feat_num = 0;
    int effct_feat_num_global = 0;
    int global_kept = 0;

    // 先收集 local
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf_local[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last_local[i];
            effct_feat_num++;
        }
    }

    // 再收集 global
    if (use_global_residual)
    {
        for (int i = 0; i < feats_down_size; i++)
        {
            if (!point_selected_surf_global[i]) continue;
            if (global_kept >= global_max_keep) break;

            laserCloudOri_global->points[effct_feat_num_global] = feats_down_body->points[i];
            const auto &np = corr_normvect_global->points[i];
            corr_normvect_global->points[effct_feat_num_global] = np;

            total_residual += res_last_global[i];

            // 统计真正进入观测的全局残差（用于判定是否清空 local 树）
            sum_rg_kept += res_last_global[i];
            rg_kept_cnt++;

            effct_feat_num_global++;
            global_kept++;
        }
    }

    const int total_eff = effct_feat_num + effct_feat_num_global;
    if (total_eff < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN_THROTTLE(0.2, "No Effective Points! (local=%d, global=%d)", effct_feat_num, effct_feat_num_global);
        return;
    }

    // 基于“全局观测均值残差”更新“是否持续清空 local 树”的滞回逻辑
    if (use_global_residual && rg_kept_cnt > 0)
    {
        const double mean_rg_kept = sum_rg_kept / (double)rg_kept_cnt;

        if (mean_rg_kept <= good_thresh) {
            // 全局配准已经好了：停止清空 local 树
            if (keep_clearing_local_tree) {
                ROS_INFO_THROTTLE(0.5, "[LOCAL TREE] global residual good (%.3f <= %.3f), stop clearing local tree.",
                                  (float)mean_rg_kept, (float)good_thresh);
            }
            keep_clearing_local_tree = false;
        } else if (mean_rg_kept > bad_thresh) {
            // 全局配准太差：开始/继续清空 local 树
            if (!keep_clearing_local_tree) {
                ROS_WARN_THROTTLE(0.5, "[LOCAL TREE] global residual bad (%.3f > %.3f), start clearing local tree.",
                                  (float)mean_rg_kept, (float)bad_thresh);
            }
            keep_clearing_local_tree = true;
        }
        // 中间区域：保持原判定，防抖
    }

    res_mean_last = total_residual / total_eff;
    match_time += omp_get_wtime() - t_match_begin;

    // 计算 Jacobian 与测量向量
    const double t_solve_begin = omp_get_wtime();
    ekfom_data.h_x = MatrixXd::Zero(total_eff, 12);
    ekfom_data.h.resize(total_eff);

    // 局部残差观测
    for (int r = 0; r < effct_feat_num; r++)
    {
        const PointType &laser_p  = laserCloudOri->points[r];
        const PointType &norm_p   = corr_normvect->points[r];

        const V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat; point_be_crossmat << SKEW_SYM_MATRX(point_this_be);

        const V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat; point_crossmat << SKEW_SYM_MATRX(point_this);

        V3D n_world(norm_p.x, norm_p.y, norm_p.z);
        const double pd2 = norm_p.intensity;
        const double w_robust = huber_weight(std::fabs(pd2), huber_delta);

        n_world *= w_robust;

        const V3D C(s.rot.conjugate() * n_world);
        const V3D A(point_crossmat * C);

        if (extrinsic_est_en)
        {
            const V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);
            ekfom_data.h_x.block<1, 12>(r, 0) << n_world(0), n_world(1), n_world(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(r, 0) << n_world(0), n_world(1), n_world(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }
        ekfom_data.h(r) = -pd2 * w_robust;
    }

    // 全局残差观测
    const double w_global_full = std::max(1.0, alpha_global * w_global_base * boost_gain);
    for (int j = 0; j < effct_feat_num_global; j++)
    {
        const int r = effct_feat_num + j;

        const PointType &laser_p  = laserCloudOri_global->points[j];
        const PointType &norm_p_g = corr_normvect_global->points[j];

        const V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat; point_be_crossmat << SKEW_SYM_MATRX(point_this_be);

        const V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat; point_crossmat << SKEW_SYM_MATRX(point_this);

        V3D n_world_g(norm_p_g.x, norm_p_g.y, norm_p_g.z);
        const double pd2_g_raw = norm_p_g.intensity;
        const double w_robust_g = huber_weight(std::fabs(pd2_g_raw), huber_delta);

        const double w_all = w_global_full * w_robust_g;
        n_world_g *= w_all;
        const double pd2_g = pd2_g_raw * w_all;

        const V3D C(s.rot.conjugate() * n_world_g);
        const V3D A(point_crossmat * C);

        if (extrinsic_est_en)
        {
            const V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);
            ekfom_data.h_x.block<1, 12>(r, 0) << n_world_g(0), n_world_g(1), n_world_g(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(r, 0) << n_world_g(0), n_world_g(1), n_world_g(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }
        ekfom_data.h(r) = -pd2_g;
    }

    solve_time += omp_get_wtime() - t_solve_begin;

    // === 关键：在求解完成后，根据 keep_clearing_local_tree 决定是否清空本地 IKD-Tree ===
    if (keep_clearing_local_tree)
    {
        // 用当前帧有效特征点重建local tree
        using TreeT = KD_TREE<PointType>;
        typename TreeT::PointVector new_pts;
        new_pts.clear();

        // 注意：这里只添加有效点（已transform到world坐标），可根据你的需求选取
        for (int i = 0; i < feats_down_size; ++i)
        {
            // 保留当前帧的所有降采样点，可根据点筛选进一步过滤
            new_pts.push_back(feats_down_world->points[i]);
        }
        ikdtree.Build(new_pts);        // 用当前帧特征点重建local-tree!
        g_local_tree_empty = false;    // 标记为非空

        s_boost_frames = std::max(s_boost_frames, 8);

        ROS_WARN_THROTTLE(0.5, "[LOCAL TREE] cleared & rebuilt with current frame (%ld pts).", new_pts.size());
    } 

    // 递减 boost
    if (s_boost_frames > 0) s_boost_frames--;
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
