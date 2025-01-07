#include "cloud_matching.h"

CloudMatching::CloudMatching()
{
    nearest_point_range = 0.5;
    voxelGridFilter.setLeafSize(0.2, 0.2, 0.2);
}

CloudMatching::~CloudMatching()
{
}

/**
 * @brief 设置点云匹配的参数，包括分辨率、范围、滤波器等
 *
 * @param _nearest_point_range 最近点范围，用于过滤最近点
 * @param _voxel_size 体素滤波的叶子大小
 * @param hor_fov_ 激光雷达的水平视场角 (度)
 * @param ver_max_ 激光雷达的最大垂直角 (度)
 * @param ver_min_ 激光雷达的最小垂直角 (度)
 * @param hor_pixel_num_ 水平方向的像素数
 * @param ver_pixel_num_ 垂直方向的像素数
 * @param lidar_to_imu_ 激光雷达到IMU的外参矩阵
 * @param imu_to_lidar_ IMU到激光雷达的外参矩阵
 * @param show_img_ 是否显示图像的标志，1表示显示，其他值表示不显示
 */
void CloudMatching::SetParams(float _nearest_point_range,
                              float _voxel_size,
                              float hor_fov_,
                              float ver_max_,
                              float ver_min_,
                              int hor_pixel_num_,
                              int ver_pixel_num_,
                              const Eigen::Matrix4d &lidar_to_imu_,
                              const Eigen::Matrix4d &imu_to_lidar_,
                              int show_img_)
{
    // 设置是否显示图像
    show_img = show_img_ == 1;

    // 设置最近点范围和体素滤波器叶子大小
    nearest_point_range = _nearest_point_range;
    voxel_size = _voxel_size;

    // 设置激光雷达视场角和像素分辨率
    hor_fov = hor_fov_;
    ver_max = ver_max_;
    ver_min = ver_min_;
    hor_pixel_num = hor_pixel_num_;
    ver_pixel_num = ver_pixel_num_;
    hor_resolution = (hor_fov * M_PI / 180.0f) / float(hor_pixel_num);             // 水平方向分辨率
    ver_resolution = ((ver_max - ver_min) * M_PI / 180.0f) / float(ver_pixel_num); // 垂直方向分辨率

    // 设置激光雷达到IMU和IMU到激光雷达的外参
    lidar_to_imu = lidar_to_imu_;
    imu_to_lidar = imu_to_lidar_;

    // 配置体素滤波器
    voxelGridFilter.setLeafSize(voxel_size, voxel_size, voxel_size); // 普通体素滤波器
    voxelGridFilterLoop.setLeafSize(0.2, 0.2, 0.2);                  // 用于闭环检测的滤波器
}

void CloudMatching::TransformPoints(const PointsWithNormals &po, PointsWithNormals &pm, const Eigen::Matrix4d T)
{
    pm = po;
    pm.x = T(0, 0) * po.x + T(0, 1) * po.y + T(0, 2) * po.z + T(0, 3);
    pm.y = T(1, 0) * po.x + T(1, 1) * po.y + T(1, 2) * po.z + T(1, 3);
    pm.z = T(2, 0) * po.x + T(2, 1) * po.y + T(2, 2) * po.z + T(2, 3);
    pm.normal_x = T(0, 0) * po.normal_x + T(0, 1) * po.normal_y + T(0, 2) * po.normal_z;
    pm.normal_y = T(1, 0) * po.normal_x + T(1, 1) * po.normal_y + T(1, 2) * po.normal_z;
    pm.normal_z = T(2, 0) * po.normal_x + T(2, 1) * po.normal_y + T(2, 2) * po.normal_z;
    return;
}

/**
 * @brief 匹配两帧点云以估计它们的相对位姿，并计算残差和退化信息
 *
 * @param keyi 第一个关键帧的点云及相关信息
 * @param keyj 第二个关键帧的点云及相关信息
 * @param rel_poseT 输出的相对位姿（从`keyj`到`keyi`的变换矩阵）
 * @param residual_sum 输出的残差和
 * @param eigenvalues 输出的特征值，用于评估匹配的退化程度
 * @param eigenvectors 输出的特征向量，用于评估匹配的退化方向
 * @return 是否匹配成功，true表示成功，false表示失败
 */
bool CloudMatching::MatchNormalClouds(const KeyFrame &keyi, const KeyFrame &keyj,
                                      Eigen::Matrix4d &rel_poseT, double &residual_sum,
                                      Eigen::Vector3d &eigenvalues, Eigen::Matrix3d &eigenvectors)
{
    std::vector<std::pair<int, int>> matches; // 用于存储点对匹配结果

    // 初始化两帧之间的初始相对位姿，以`keyj`的坐标系为参考
    rel_poseT = (keyj.lidar_pose.inverse() * keyi.lidar_pose).matrix();

    // 准备点云
    pcl::PointCloud<PointsWithNormals>::Ptr cloud_i(new pcl::PointCloud<PointsWithNormals>());
    pcl::PointCloud<PointsWithNormals>::Ptr cloud_j(new pcl::PointCloud<PointsWithNormals>());

    // 根据体素大小对点云进行体素滤波
    if (voxel_size > 0.1)
    {
        voxelGridFilter.setInputCloud(keyi.normalPoints);
        voxelGridFilter.filter(*cloud_i);

        voxelGridFilter.setInputCloud(keyj.normalPoints);
        voxelGridFilter.filter(*cloud_j);
    }
    else
    {
        cloud_i.reset(new pcl::PointCloud<PointsWithNormals>(*keyi.normalPoints));
        cloud_j.reset(new pcl::PointCloud<PointsWithNormals>(*keyj.normalPoints));
    }

    // 如果目标点云为空，匹配失败
    if (cloud_j->empty())
    {
        return false;
    }

    // 使用k-d树加速最近邻搜索
    kdtreeNormalPoint.reset(new pcl::KdTreeFLANN<PointsWithNormals>());
    kdtreeNormalPoint->setInputCloud(cloud_j);

    // 高斯牛顿法优化，最多迭代30次
    for (int opt_iter = 0; opt_iter < 30; ++opt_iter)
    {
        std::vector<int> match_vec(cloud_i->size(), -1); // 存储匹配结果
        residual_sum = 0.0;                              // 初始化残差

#pragma omp parallel for num_threads(4) // 使用OpenMP加速处理
        for (size_t point_id = 0; point_id < cloud_i->size(); ++point_id)
        {
            PointsWithNormals point_i = cloud_i->points[point_id];
            PointsWithNormals point_ij;
            TransformPoints(point_i, point_ij, rel_poseT);

            std::vector<int> knn_index;
            std::vector<float> knn_distance;

            // 在目标点云中找到最近的3个点
            kdtreeNormalPoint->nearestKSearch(point_ij, 3, knn_index, knn_distance);

            // 检查法向量和距离条件，确定匹配点
            for (int i = 0; i < 3; ++i)
            {
                if (knn_distance[i] > nearest_point_range)
                    break;

                PointsWithNormals &point_j = cloud_j->points[knn_index[i]];
                float dot_product = point_ij.normal_x * point_j.normal_x +
                                    point_ij.normal_y * point_j.normal_y +
                                    point_ij.normal_z * point_j.normal_z;

                if (dot_product < 0.9848) // 法向量夹角超过10度
                    continue;

                match_vec[point_id] = knn_index[i];
                break;
            }
        }

        // 收集有效匹配对
        matches.clear();
        for (int i = 0; i < match_vec.size(); ++i)
        {
            if (match_vec[i] != -1)
                matches.emplace_back(i, match_vec[i]);
        }

        // 检查匹配对数量是否足够
        if (matches.size() < 15)
        {
            printf("Match Number: %d\n", int(matches.size()));
            return false;
        }

        // 构造优化问题的雅可比矩阵和右侧向量
        Eigen::MatrixXd A(6, 6);
        Eigen::VectorXd b(6);
        A.setZero();
        b.setZero();

        for (const auto &match : matches)
        {
            PointsWithNormals point_i = cloud_i->points[match.first];
            PointsWithNormals point_ij;
            TransformPoints(point_i, point_ij, rel_poseT);
            PointsWithNormals &point_j = cloud_j->points[match.second];

            Eigen::Vector3d X_ij(point_ij.x, point_ij.y, point_ij.z);
            Eigen::Vector3d X_j(point_j.x, point_j.y, point_j.z);
            Eigen::Vector3d N_j(point_j.normal_x, point_j.normal_y, point_j.normal_z);

            double residual = N_j.dot(X_ij - X_j);
            Eigen::Matrix<double, 1, 6> jacobian;
            jacobian << N_j[2] * X_ij[1] - N_j[1] * X_ij[2],
                N_j[0] * X_ij[2] - N_j[2] * X_ij[0],
                N_j[1] * X_ij[0] - N_j[0] * X_ij[1],
                N_j[0],
                N_j[1],
                N_j[2];

            A += jacobian.transpose() * jacobian;
            b += jacobian.transpose() * residual;

            residual_sum += residual * residual;
        }

        residual_sum /= matches.size();

        Eigen::VectorXd delta = A.ldlt().solve(-b);

        Eigen::Matrix4d update = Eigen::Matrix4d::Identity();
        update(0, 3) = delta(3);
        update(1, 3) = delta(4);
        update(2, 3) = delta(5);
        update.block(0, 0, 3, 3) = (Eigen::AngleAxisd(delta(2), Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(delta(1), Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(delta(0), Eigen::Vector3d::UnitX()))
                                       .matrix();

        rel_poseT = update * rel_poseT;

        float deltaR = delta.segment<3>(0).norm() * 180.0 / M_PI;
        float deltaT = delta.segment<3>(3).norm();

        if (deltaR < 0.01 && deltaT < 0.01)
            break;
    }

    // 检查匹配是否退化
    Eigen::Matrix3d covMatrix = Eigen::Matrix3d::Zero();
    for (const auto &match : matches)
    {
        PointsWithNormals &pt = cloud_i->points[match.first];
        Eigen::Vector3d normal(pt.normal_x, pt.normal_y, pt.normal_z);
        covMatrix += normal * normal.transpose();
    }
    covMatrix /= matches.size();
    covMatrix = imu_to_lidar.block(0, 0, 3, 3) * covMatrix * lidar_to_imu.block(0, 0, 3, 3);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covMatrix);
    eigenvalues = eigenSolver.eigenvalues();
    eigenvectors = eigenSolver.eigenvectors();

    return true;
}

bool CloudMatching::MatchNormalCloudsNoDegeneracy(const KeyFrame &keyi, const KeyFrame &keyj,
                                                  Eigen::Matrix4d &rel_poseT, double &residual_sum,
                                                  Eigen::Vector3d &eigenvalues, Eigen::Matrix3d &eigenvectors)
{

    std::vector<std::pair<int, int>> matches;
    rel_poseT = (keyj.lidar_pose.inverse() * keyi.lidar_pose).matrix();

    // Get relative pose between i and j (i coordinate system seen in j)
    pcl::PointCloud<PointsWithNormals>::Ptr cloud_i(new pcl::PointCloud<PointsWithNormals>());
    pcl::PointCloud<PointsWithNormals>::Ptr cloud_j(new pcl::PointCloud<PointsWithNormals>());

    if (voxel_size > 0.1)
    {
        voxelGridFilter.setInputCloud(keyi.normalPoints);
        voxelGridFilter.filter(*cloud_i);

        voxelGridFilter.setInputCloud(keyj.normalPoints);
        voxelGridFilter.filter(*cloud_j);
    }
    else
    {
        cloud_i.reset(new pcl::PointCloud<PointsWithNormals>(*keyi.normalPoints));
        cloud_j.reset(new pcl::PointCloud<PointsWithNormals>(*keyj.normalPoints));
    }

    if (cloud_j->empty())
    {
        return false;
    }

    kdtreeNormalPoint.reset(new pcl::KdTreeFLANN<PointsWithNormals>());
    kdtreeNormalPoint->setInputCloud(cloud_j);

    // Gauss Newton Method
    for (int opt_iter = 0; opt_iter < 30; ++opt_iter)
    {

        std::vector<int> match_vec(cloud_i->size(), -1);
        residual_sum = 0.0;

#pragma omp parallel for num_threads(4)
        for (size_t point_id = 0; point_id < cloud_i->size(); ++point_id)
        {
            PointsWithNormals point_i = cloud_i->points[point_id];
            PointsWithNormals point_ij;
            TransformPoints(point_i, point_ij, rel_poseT);

            std::vector<int> knn_index;
            std::vector<float> knn_distance;

            bool found_match = false;
            int match_id = -1;

            kdtreeNormalPoint->nearestKSearch(point_ij, 3, knn_index, knn_distance);

            for (int i = 0; i < 3; ++i)
            {
                if (knn_distance[i] > nearest_point_range)
                {
                    break;
                }
                // Check angle between normals
                PointsWithNormals &point_j = cloud_j->points[knn_index[i]];
                float dot_product = point_ij.normal_x * point_j.normal_x +
                                    point_ij.normal_y * point_j.normal_y +
                                    point_ij.normal_z * point_j.normal_z;

                if (dot_product < 0.9848)
                { // Approximately 10 degrees
                    continue;
                }

                found_match = true;
                match_id = knn_index[i];
                break;
            }

            if (!found_match)
            {
                continue;
            }
            match_vec[point_id] = match_id;
        }

        matches.clear();
        for (int i = 0; i < match_vec.size(); ++i)
        {
            if (match_vec[i] != -1)
            {
                std::pair<int, int> match_pair{i, match_vec[i]};
                matches.push_back(match_pair);
            }
        }

        // If match size too small, it failed
        if (matches.size() < 15)
        {
            printf("Match Number: %d\n", int(matches.size()));
            return false;
        }

        // Optimization part: Gauss Newton Method
        Eigen::MatrixXd A(6, 6);
        Eigen::VectorXd b(6);
        A.setZero();
        b.setZero();

        for (int match_id = 0; match_id < matches.size(); ++match_id)
        {
            PointsWithNormals point_i = cloud_i->points[matches[match_id].first];
            PointsWithNormals point_ij;
            TransformPoints(point_i, point_ij, rel_poseT);
            PointsWithNormals &point_j = cloud_j->points[matches[match_id].second];

            Eigen::Vector3d X_ij(point_ij.x, point_ij.y, point_ij.z);
            Eigen::Vector3d X_j(point_j.x, point_j.y, point_j.z);
            Eigen::Vector3d N_j(point_j.normal_x, point_j.normal_y, point_j.normal_z);

            double residual = N_j.dot(X_ij - X_j);
            Eigen::Matrix<double, 1, 6> jacobian;
            jacobian << N_j[2] * X_ij[1] - N_j[1] * X_ij[2],
                N_j[0] * X_ij[2] - N_j[2] * X_ij[0],
                N_j[1] * X_ij[0] - N_j[0] * X_ij[1],
                N_j[0],
                N_j[1],
                N_j[2];

            A += jacobian.transpose() * jacobian;
            b += jacobian.transpose() * residual;

            residual_sum += residual * residual;
        }

        residual_sum /= static_cast<float>(matches.size());

        Eigen::VectorXd delta = A.ldlt().solve(-b);
        Eigen::Matrix4d update = Eigen::Matrix4d::Identity();
        update(0, 3) = delta(3);
        update(1, 3) = delta(4);
        update(2, 3) = delta(5);
        update.block(0, 0, 3, 3) = (Eigen::AngleAxisd(delta(2), Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(delta(1), Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(delta(0), Eigen::Vector3d::UnitX()))
                                       .matrix();
        rel_poseT = update * rel_poseT;

        float deltaR = delta.segment<3>(0).norm() * 180.0 / M_PI;
        float deltaT = delta.segment<3>(3).norm();

        if (deltaR < 0.01 && deltaT < 0.01)
        {
            break;
        }
    }

    // Check for degeneracy
    Eigen::Matrix3d covMatrix;
    covMatrix.setZero();

    for (int i = 0; i < matches.size(); ++i)
    {
        PointsWithNormals &pt = cloud_i->points[matches[i].first];
        Eigen::Vector3d normal(pt.normal_x, pt.normal_y, pt.normal_z);
        covMatrix += normal * normal.transpose();
    }
    covMatrix /= static_cast<float>(matches.size());
    covMatrix = imu_to_lidar.block(0, 0, 3, 3) * covMatrix * lidar_to_imu.block(0, 0, 3, 3);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covMatrix);
    eigenvalues = eigenSolver.eigenvalues();
    eigenvectors = eigenSolver.eigenvectors();

    // if(eigenvalues(0) < 0.01) {
    //     std::cout<<"Eigen Values: "<<eigenvalues.transpose()<<std::endl;
    //     return false;
    // }

    return true;
}

/**
 * @brief 通过点云投影进行两帧点云的匹配
 *
 * @param keyi 第一帧关键帧的点云数据和位姿
 * @param keyj 第二帧关键帧的点云数据和位姿
 * @param rel_poseT 输出两帧之间的相对变换矩阵
 * @param residual_sum 输出残差和
 * @param projected_no 输出投影点的数量
 * @return 是否匹配成功
 */
bool CloudMatching::MatchByProjection(const KeyFrame &keyi,
                                      const KeyFrame &keyj,
                                      Eigen::Matrix4d &rel_poseT,
                                      double &residual_sum,
                                      int &projected_no)
{
    if (show_img)
    {
        ori_submap_img = cv::Mat(ver_pixel_num, hor_pixel_num, CV_8UC3, cv::Scalar(255, 255, 255));
        ori_matched_img = cv::Mat(ver_pixel_num, hor_pixel_num, CV_8UC3, cv::Scalar(255, 255, 255));
        now_img = cv::Mat(ver_pixel_num, hor_pixel_num, CV_8UC3, cv::Scalar(255, 255, 255));
        now_img_full = cv::Mat(ver_pixel_num, hor_pixel_num, CV_8UC3, cv::Scalar(255, 255, 255));
        submap_img = cv::Mat(ver_pixel_num, hor_pixel_num, CV_8UC3, cv::Scalar(255, 255, 255));
        matched_img = cv::Mat(ver_pixel_num, hor_pixel_num, CV_8UC3, cv::Scalar(255, 255, 255));
    }

    pcl::PointCloud<PointsWithNormals>::Ptr cloud_j(new pcl::PointCloud<PointsWithNormals>());

    if (voxel_size > 0.1)
    {
        // voxelGridFilter.setInputCloud(keyj.normalPoints);
        // voxelGridFilter.filter(*cloud_j);
        voxelGridFilterLoop.setInputCloud(keyj.normalPoints);
        voxelGridFilterLoop.filter(*cloud_j);
    }
    else
    {
        cloud_j.reset(new pcl::PointCloud<PointsWithNormals>(*keyj.normalPoints));
    }

    pcl::PointCloud<PointsWithNormals>::Ptr transformed_i(new pcl::PointCloud<PointsWithNormals>);
    pcl::PointCloud<PointsWithNormals>::Ptr projected_i(new pcl::PointCloud<PointsWithNormals>);
    rel_poseT = (keyj.lidar_pose.inverse() * keyi.lidar_pose).matrix();
    std::vector<std::pair<int, int>> matches;

    if (show_img)
    {
        pcl::transformPointCloudWithNormals(*keyi.normalPoints, *transformed_i, rel_poseT);
        ProjectCloud(transformed_i, projected_i);
        int ori_projection_no = 0;
        for (int i = 0; i < projected_i->size(); ++i)
        {
            if (projected_i->points[i].valid)
            {
                ++ori_projection_no;
            }
        }

        printf("original projection_no %d\n", ori_projection_no);

#pragma opm parallel for num_threads(4)
        for (int i = 0; i < ver_pixel_num * hor_pixel_num; ++i)
        {

            int v = i / hor_pixel_num;
            int u = i % hor_pixel_num;

            if (keyj.totalPoints->points[i].valid)
            {
                int r = 0;
                int g = 0;
                int b = 0;
                r = int((keyj.totalPoints->points[i].normal_x * 0.5 + 0.5) * 255);
                g = int((keyj.totalPoints->points[i].normal_y * 0.5 + 0.5) * 255);
                b = int((keyj.totalPoints->points[i].normal_z * 0.5 + 0.5) * 255);
                now_img_full.at<cv::Vec3b>(v, u) = cv::Vec3b(b, g, r);
            }

            if (projected_i->points[i].valid)
            {
                int r = 0;
                int g = 0;
                int b = 0;
                r = int((projected_i->points[i].normal_x * 0.5 + 0.5) * 255);
                g = int((projected_i->points[i].normal_y * 0.5 + 0.5) * 255);
                b = int((projected_i->points[i].normal_z * 0.5 + 0.5) * 255);
                ori_submap_img.at<cv::Vec3b>(v, u) = cv::Vec3b(b, g, r);
            }
        }

#pragma opm parallel for num_threads(4)
        for (int i = 0; i < cloud_j->size(); ++i)
        {
            PointsWithNormals &pt = cloud_j->points[i];
            int r = 0;
            int g = 0;
            int b = 0;
            r = int((pt.normal_x * 0.5 + 0.5) * 255);
            g = int((pt.normal_y * 0.5 + 0.5) * 255);
            b = int((pt.normal_z * 0.5 + 0.5) * 255);

            int u, v;
            get_uv(pt, u, v);

            now_img.at<cv::Vec3b>(v, u) = cv::Vec3b(b, g, r);

            int index = hor_pixel_num * v + u;
            PointsWithNormals &target_pt = projected_i->points[index];
            if (projected_i->points[index].valid == 0)
            {
                ori_matched_img.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 255, 0); // Green
                continue;
            }

            double dist_diff = std::sqrt((pt.x - target_pt.x) * (pt.x - target_pt.x) +
                                         (pt.y - target_pt.y) * (pt.y - target_pt.y) +
                                         (pt.z - target_pt.z) * (pt.z - target_pt.z));

            if (dist_diff > 0.5)
            {
                ori_matched_img.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 255); // Red
                continue;
            }

            double dot_product = pt.normal_x * target_pt.normal_x +
                                 pt.normal_y * target_pt.normal_y +
                                 pt.normal_z * target_pt.normal_z;

            if (dot_product < 0.9063)
            {                                                               // Approximately 20 degrees
                ori_matched_img.at<cv::Vec3b>(v, u) = cv::Vec3b(255, 0, 0); // Blue
                continue;
            }
            ori_matched_img.at<cv::Vec3b>(v, u) = cv::Vec3b(255, 255, 255);
        }

        cv::vconcat(ori_submap_img, now_img_full, ori_submap_img);
        cv::vconcat(ori_submap_img, now_img, ori_submap_img);
        cv::vconcat(ori_submap_img, ori_matched_img, ori_submap_img);
        cv::imshow("original_img", ori_submap_img);
        cv::waitKey(1);
    }

    // Gauss Newton Method

    for (int opt_iter = 0; opt_iter < 30; ++opt_iter)
    {
        if (show_img)
        {
            matched_img = cv::Mat(ver_pixel_num, hor_pixel_num, CV_8UC3, cv::Scalar(255, 255, 255));
        }
        projected_no = 0;
        pcl::transformPointCloudWithNormals(*keyi.normalPoints, *transformed_i, rel_poseT);
        ProjectCloud(transformed_i, projected_i);
        for (int i = 0; i < projected_i->size(); ++i)
        {
            if (projected_i->points[i].valid)
            {
                ++projected_no;
            }
        }

        if (projected_no < static_cast<int>(static_cast<float>(ver_pixel_num * hor_pixel_num) * 0.01))
        {
            printf("Projected Number: %d\n", projected_no);
            return false;
        }

        std::vector<int> match_vec(cloud_j->size(), -1);
        residual_sum = 0.0;

#pragma omp parallel for num_threads(4)
        for (int i = 0; i < cloud_j->size(); ++i)
        {
            PointsWithNormals &query_pt = cloud_j->points[i];
            int u, v;
            get_uv(query_pt, u, v);
            if (u == -1 || v == -1)
            {
                continue;
            }

            // 5 by 5 search

            double min_dist_diff = 20000.0;
            for (int j = -2; j <= 2; ++j)
            {
                if (u + j < 0 || u + j >= hor_pixel_num)
                {
                    continue;
                }
                for (int k = -2; k <= 2; ++k)
                {
                    if (v + k < 0 || v + k >= ver_pixel_num)
                    {
                        continue;
                    }
                    if (j == 0 && k == 0)
                    {
                        continue;
                    }

                    int index = hor_pixel_num * (v + k) + (u + j);

                    PointsWithNormals &target_pt = projected_i->points[index];
                    if (target_pt.valid == 0)
                    {
                        if (show_img)
                        {
                            // matched_img.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 255, 0); //Green
                        }
                        continue;
                    }

                    double dist_diff = std::sqrt((query_pt.x - target_pt.x) * (query_pt.x - target_pt.x) +
                                                 (query_pt.y - target_pt.y) * (query_pt.y - target_pt.y) +
                                                 (query_pt.z - target_pt.z) * (query_pt.z - target_pt.z));

                    if (dist_diff > 0.5)
                    {
                        if (show_img)
                        {
                            matched_img.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 255); // Red
                        }
                        continue;
                    }

                    double dot_product = query_pt.normal_x * target_pt.normal_x +
                                         query_pt.normal_y * target_pt.normal_y +
                                         query_pt.normal_z * target_pt.normal_z;

                    if (dot_product < 0.9063)
                    { // Approximately 20 degrees
                        if (show_img)
                        {
                            matched_img.at<cv::Vec3b>(v, u) = cv::Vec3b(255, 0, 0); // Blue
                        }
                        continue;
                    }
                    if (min_dist_diff > dist_diff)
                    {
                        min_dist_diff = dist_diff;
                        match_vec[i] = index;
                    }
                }
            }
        }

        matches.clear();
        for (int i = 0; i < match_vec.size(); ++i)
        {
            if (match_vec[i] != -1)
            {
                std::pair<int, int> match_pair{i, match_vec[i]};
                matches.push_back(match_pair);
            }
        }

        // If match size too small, it failed
        if (matches.size() < static_cast<float>(ver_pixel_num * hor_pixel_num) * 0.005)
        {
            // if(static_cast<int>(matches.size()) < 300) {
            printf("Match Number: %d\n", int(matches.size()));
            return false;
        }

        // Check for degeneracy
        Eigen::Matrix3d covMatrix;
        covMatrix.setZero();

        for (int i = 0; i < matches.size(); ++i)
        {
            int v = matches[i].second / hor_pixel_num;
            int u = matches[i].second % hor_pixel_num;
            if (show_img)
            {
                matched_img.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 255, 0);
            }

            PointsWithNormals &pt = cloud_j->points[matches[i].first];
            Eigen::Vector3d normal(pt.normal_x, pt.normal_y, pt.normal_z);
            covMatrix += normal * normal.transpose();
        }
        covMatrix /= static_cast<float>(matches.size());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covMatrix);
        Eigen::VectorXd eigenvalues = eigenSolver.eigenvalues();

        if (eigenvalues(0) < 0.01)
        {
            std::cout << "Eigen Values: " << eigenvalues.transpose() << std::endl;
            return false;
        }

        // Optimization part: Gauss Newton Method
        Eigen::MatrixXd A(6, 6);
        Eigen::VectorXd b(6);
        A.setZero();
        b.setZero();

        for (int match_id = 0; match_id < matches.size(); ++match_id)
        {
            PointsWithNormals &point_j = projected_i->points[matches[match_id].second];
            PointsWithNormals &point_i = cloud_j->points[matches[match_id].first];

            Eigen::Vector3d X_ji(point_j.x, point_j.y, point_j.z);
            Eigen::Vector3d X_i(point_i.x, point_i.y, point_i.z);
            Eigen::Vector3d N_i(point_i.normal_x, point_i.normal_y, point_i.normal_z);

            double residual = N_i.dot(X_ji - X_i);
            Eigen::Matrix<double, 1, 6> jacobian;
            jacobian << N_i[2] * X_ji[1] - N_i[1] * X_ji[2],
                N_i[0] * X_ji[2] - N_i[2] * X_ji[0],
                N_i[1] * X_ji[0] - N_i[0] * X_ji[1],
                N_i[0],
                N_i[1],
                N_i[2];

            A += jacobian.transpose() * jacobian;
            b += jacobian.transpose() * residual;

            residual_sum += residual * residual;
        }

        residual_sum /= static_cast<float>(matches.size());

        Eigen::VectorXd delta = A.ldlt().solve(-b);
        Eigen::Matrix4d update = Eigen::Matrix4d::Identity();
        update(0, 3) = delta(3);
        update(1, 3) = delta(4);
        update(2, 3) = delta(5);
        update.block(0, 0, 3, 3) = (Eigen::AngleAxisd(delta(2), Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(delta(1), Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(delta(0), Eigen::Vector3d::UnitX()))
                                       .matrix();
        rel_poseT = update * rel_poseT;

        float deltaR = delta.segment<3>(0).norm() * 180.0 / M_PI;
        float deltaT = delta.segment<3>(3).norm();

        if (deltaR < 0.01 && deltaT < 0.01)
        {
            break;
        }
    }
    printf("Loop Match Number: %d\n", int(matches.size()));

    if (show_img)
    {
#pragma opm parallel for num_threads(4)
        for (int i = 0; i < ver_pixel_num * hor_pixel_num; ++i)
        {

            int v = i / hor_pixel_num;
            int u = i % hor_pixel_num;

            if (projected_i->points[i].valid)
            {
                int r = 0;
                int g = 0;
                int b = 0;
                r = int((projected_i->points[i].normal_x * 0.5 + 0.5) * 255);
                g = int((projected_i->points[i].normal_y * 0.5 + 0.5) * 255);
                b = int((projected_i->points[i].normal_z * 0.5 + 0.5) * 255);
                submap_img.at<cv::Vec3b>(v, u) = cv::Vec3b(b, g, r);
            }
        }

        cv::imshow("submap projection", submap_img);
        cv::imshow("current projection", now_img);
        cv::imshow("matching projection", matched_img);

        cv::vconcat(now_img, submap_img, submap_img);
        cv::vconcat(submap_img, matched_img, submap_img);
        cv::imshow("now_img", submap_img);
        cv::waitKey(1);
    }

    return true;
}

/**
 * @brief 将点云投影到二维像素平面并对每个像素选择最近的点，同时移除错误或冗余点
 *
 * @param cloud_in 输入的三维点云数据
 * @param cloud_out 输出的二维投影点云数据
 */
void CloudMatching::ProjectCloud(const pcl::PointCloud<PointsWithNormals>::Ptr &cloud_in,
                                 pcl::PointCloud<PointsWithNormals>::Ptr &cloud_out)
{
    // 初始化输出点云，每个像素点对应一个容器
    cloud_out.reset(new pcl::PointCloud<PointsWithNormals>(ver_pixel_num, hor_pixel_num));

    // 遍历输入点云中的每个点，进行投影
#pragma omp parallel for num_threads(4) // 使用OpenMP进行多线程并行加速
    for (size_t i = 0; i < cloud_in->size(); ++i)
    {
        PointsWithNormals &pt = cloud_in->points[i];
        int u, v;               // u: 水平像素索引，v: 垂直像素索引
        get_uv(pt, u, v);       // 根据点的坐标计算其在二维投影平面的像素坐标
        if (u == -1 || v == -1) // 如果像素坐标无效，跳过
        {
            continue;
        }

        int index = v * hor_pixel_num + u;       // 计算点在二维数组中的索引
        if (cloud_out->points[index].valid == 1) // 如果该像素已有点
        {
            // 检查当前点与已有点的距离，保留距离更近的点
            if (get_range(pt) >= get_range(cloud_out->points[index]))
            {
                continue;
            }
        }
        // 如果该像素尚未被占用或当前点更近，则更新该像素的点信息
        cloud_out->points[index] = pt;
        cloud_out->points[index].valid = 1;
    }

    // 对点云进行进一步清理，移除可能的错误点或冗余点
    for (int u = 0; u < hor_pixel_num; ++u) // 遍历水平像素
    {
        for (int v = 0; v < ver_pixel_num; ++v) // 遍历垂直像素
        {
            int index = v * hor_pixel_num + u;
            if (cloud_out->points[index].valid == 0) // 如果该像素无效，跳过
            {
                continue;
            }

            PointsWithNormals &query_pt = cloud_out->points[index];                                     // 获取当前像素点信息
            Eigen::Vector3d ray_dir = Eigen::Vector3d(query_pt.x, query_pt.y, query_pt.z).normalized(); // 当前点的方向向量

            // 如果当前点的方向向量与法向量夹角过大，则跳过
            if (ray_dir.dot(Eigen::Vector3d(query_pt.normal_x, query_pt.normal_y, query_pt.normal_z)) < 0)
            {
                continue;
            }

            // 检查当前点的邻域像素，移除可能的冗余点
            for (int j = -2; j <= 2; ++j)
            {
                if (u + j < 0 || u + j >= hor_pixel_num) // 检查水平像素是否越界
                {
                    continue;
                }
                for (int k = -2; k <= 2; ++k)
                {
                    if (v + k < 0 || v + k >= ver_pixel_num) // 检查垂直像素是否越界
                    {
                        continue;
                    }
                    if (j == 0 && k == 0) // 跳过当前点本身
                    {
                        continue;
                    }
                    int target_index = hor_pixel_num * (v + k) + (u + j);

                    PointsWithNormals &target_pt = cloud_out->points[target_index]; // 获取邻域像素点信息
                    if (target_pt.valid == 0)                                       // 如果邻域像素无效，跳过
                    {
                        continue;
                    }
                    if (target_pt.range > query_pt.range) // 如果邻域点距离大于当前点，跳过
                    {
                        continue;
                    }

                    Eigen::Vector3d target_ray_dir = Eigen::Vector3d(target_pt.x, target_pt.y, target_pt.z).normalized(); // 邻域点的方向向量

                    // 如果邻域点的方向向量与其法向量夹角过大，跳过
                    if (target_ray_dir.dot(Eigen::Vector3d(target_pt.normal_x, target_pt.normal_y, target_pt.normal_z)) > 0)
                    {
                        continue;
                    }
                    // 标记当前点为无效点
                    query_pt.valid = 0;
                }
            }
        }
    }
}

void CloudMatching::get_uv(const PointsWithNormals &point, int &u, int &v)
{

    float xy_range = std::sqrt(point.x * point.x + point.y * point.y);
    if (xy_range < 1e-3)
    {
        u = -1;
        v = -1;
        return;
    }

    float azimuth = M_PI - atan2(point.y, point.x);
    float elevation = (ver_max * M_PI / 180.0) - atan2(point.z, xy_range);

    int u_ = (azimuth / hor_resolution);
    int v_ = (elevation / ver_resolution);

    if (u_ < 0 || u_ >= hor_pixel_num || v_ < 0 || v_ >= ver_pixel_num)
    {
        u = -1;
        v = -1;
        return;
    }

    u = u_;
    v = v_;
    return;
}

float CloudMatching::get_range(PointsWithNormals &pt)
{
    pt.range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    return pt.range;
}
