#include <scan_pose_gt_generation/ScanPoseGtGeneration.h>

#include <fstream>

#include <boost/filesystem.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <beam_utils/log.h>
#include <beam_utils/math.h>
#include <beam_utils/time.h>

namespace scan_pose_gt_gen {

void ScanPoseGtGeneration::run() {
  LoadConfig();
  LoadExtrinsics();
  LoadGtCloud();
  LoadTrajectory();
  SetupRegistration();
  RegisterScans();
  SaveResults();
}

void ScanPoseGtGeneration::LoadConfig() {
  BEAM_INFO("Loading config file: {}", inputs_.config);
  nlohmann::json J;
  if (!beam::ReadJson(inputs_.config, J)) {
    throw std::runtime_error{"Invalid config json"};
  }

  if (!J.contains("map_max_size") || !J.contains("save_map") ||
      !J.contains("rotation_threshold_deg") ||
      !J.contains("translation_threshold_m") || !J.contains("scan_filters") ||
      !J.contains("gt_cloud_filters")) {
    throw std::runtime_error{
        "missing one or more parameter in the config file"};
  }

  params_.map_max_size = J["map_max_size"].get<int>();
  params_.save_map = J["save_map"].get<bool>();
  params_.rotation_threshold_deg = J["rotation_threshold_deg"].get<double>();
  params_.translation_threshold_m = J["translation_threshold_m"].get<double>();

  if (params_.save_map) {
    map_save_dir_ = beam::CombinePaths(inputs_.output_directory, "gt_maps");
    boost::filesystem::create_directory(map_save_dir_);
  }

  scan_filters_ = beam_filtering::LoadFilterParamsVector(J["scan_filters"]);
  gt_cloud_filters_ =
      beam_filtering::LoadFilterParamsVector(J["gt_cloud_filters"]);
}

void ScanPoseGtGeneration::LoadExtrinsics() {
  beam_calibration::TfTree extrinsics;
  extrinsics.LoadJSON(inputs_.extrinsics);
  T_MOVING_LIDAR_ =
      extrinsics.GetTransformEigen(moving_frame_id_, lidar_frame_id_).matrix();
}

void ScanPoseGtGeneration::LoadGtCloud() {
  BEAM_INFO("Loading gt cloud: {}", inputs_.gt_cloud);
  PointCloud gt_cloud_in;
  pcl::io::loadPCDFile(inputs_.gt_cloud, gt_cloud_in);
  if (gt_cloud_in.empty()) {
    BEAM_ERROR("empty input gt cloud.");
    throw std::runtime_error{"empty input cloud"};
  }

  BEAM_INFO("Loading gt cloud pose: {}", inputs_.gt_cloud_pose);
  nlohmann::json J;
  if (!beam::ReadJson(inputs_.gt_cloud_pose, J)) {
    throw std::runtime_error{"Invalid gt_cloud_pose json"};
  }

  if (!J.contains("T_World_GtCloud")) {
    throw std::runtime_error{
        "cannot load gt cloud pose, missing T_World_GtCloud field"};
  }

  std::vector<double> v = J["T_World_GtCloud"];
  if (v.size() != 16) {
    throw std::runtime_error{"invalid T_World_GtCloud, size must be 16 (4x4)"};
  }

  Eigen::Matrix4d T_World_GtCloud;
  T_World_GtCloud << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9],
      v[10], v[11], v[12], v[13], v[14], v[15];
  if (!beam::IsTransformationMatrix(T_World_GtCloud)) {
    throw std::runtime_error{
        "T_World_GtCloud is not a valid transformation matrix"};
  }
  T_World_GtCloud_ = T_World_GtCloud;
  PointCloud gt_cloud_in_world;
  pcl::transformPointCloud(gt_cloud_in, gt_cloud_in_world, T_World_GtCloud);

  gt_cloud_in_world_ = std::make_shared<PointCloud>(
      beam_filtering::FilterPointCloud<pcl::PointXYZ>(gt_cloud_in_world,
                                                      gt_cloud_filters_));
}

void ScanPoseGtGeneration::LoadTrajectory() {
  beam_mapping::Poses poses;
  BEAM_INFO("Loading initial trajectory: {}", inputs_.initial_trajectory);
  if (!poses.LoadFromFile(inputs_.initial_trajectory)) {
    throw std::invalid_argument{"Invalid pose file"};
  }

  moving_frame_id_ = poses.GetMovingFrame();
  world_frame_id_ = poses.GetFixedFrame();

  for (size_t k = 0; k < poses.GetTimeStamps().size(); k++) {
    Eigen::Affine3d T_WORLD_MOVINGFRAME(poses.GetPoses()[k]);
    trajectory_.AddTransform(T_WORLD_MOVINGFRAME, world_frame_id_,
                             moving_frame_id_, poses.GetTimeStamps()[k]);
  }
}

void ScanPoseGtGeneration::SetupRegistration() {
  BEAM_INFO("Setting up registration");
  icp_.setInputTarget(gt_cloud_in_world_);
  icp_.setMaxCorrespondenceDistance(params_.icp_params.max_corr_dist);
  icp_.setMaximumIterations(params_.icp_params.max_iterations);
  icp_.setTransformationEpsilon(params_.icp_params.transform_eps);
  icp_.setEuclideanFitnessEpsilon(params_.icp_params.fitness_eps);
  BEAM_INFO("Done setting up registration, elapsed time: {}s",
            timer_.elapsed());
}

void ScanPoseGtGeneration::RegisterScans() {
  rosbag::Bag bag;
  BEAM_INFO("Opening bag: {}", inputs_.bag);
  bag.open(inputs_.bag, rosbag::bagmode::Read);
  if (!bag.isOpen()) { throw std::runtime_error{"unable to open ROS bag"}; }

  rosbag::View view(bag, rosbag::TopicQuery(inputs_.topic), ros::TIME_MIN,
                    ros::TIME_MAX, true);
  int total_messages = view.size();
  BEAM_INFO("Read a total of {} pointcloud messages", total_messages);
  for (auto iter = view.begin(); iter != view.end(); iter++) {
    scan_counter_++;
    auto sensor_msg = iter->instantiate<sensor_msgs::PointCloud2>();
    ros::Time timestamp = sensor_msg->header.stamp;
    pcl::PCLPointCloud2::Ptr pcl_pc2_tmp =
        std::make_shared<pcl::PCLPointCloud2>();
    PointCloud cloud;
    beam::pcl_conversions::toPCL(*sensor_msg, *pcl_pc2_tmp);
    pcl::fromPCLPointCloud2(*pcl_pc2_tmp, cloud);
    RegisterSingleScan(cloud, timestamp);
  }
}

void ScanPoseGtGeneration::RegisterSingleScan(
    const PointCloud& cloud_in_lidar_frame, const ros::Time& timestamp) {
  // filter cloud and transform to estimated world frame
  PointCloud cloud_filtered_in_lidar_frame =
      beam_filtering::FilterPointCloud<pcl::PointXYZ>(cloud_in_lidar_frame,
                                                      scan_filters_);
  PointCloudPtr cloud_in_WorldEst = std::make_shared<PointCloud>();

  // if first scan, get estimated pose straight from poses. Otherwise, get only
  // relative pose from poses
  Eigen::Matrix4d T_WorldEst_Lidar;
  if (current_map_size_ == 0 && results_.saved_cloud_names.empty()) {
    T_WorldEst_Lidar = GetT_WorldEst_Lidar(timestamp);
  } else {
    T_WorldEst_Lidar = T_World_MovingLast_ *
                       GetT_MovingLast_MovingCurrent(timestamp) *
                       T_MOVING_LIDAR_;
  }

  pcl::transformPointCloud(cloud_filtered_in_lidar_frame, *cloud_in_WorldEst,
                           T_WorldEst_Lidar);

  // run ICP
  BEAM_INFO("Running registration for scan {}, timestamp: {} Ns", scan_counter_,
            timestamp.toNSec());
  timer_.restart();
  PointCloud registered_cloud;
  icp_.setInputSource(cloud_in_WorldEst);
  icp_.align(registered_cloud);
  Eigen::Matrix4d T_World_WorldEst =
      icp_.getFinalTransformation().cast<double>();
  BEAM_INFO("Registration time: {}s", timer_.elapsed());

  if (!icp_.hasConverged()) {
    results_.invalid_scan_stamps.push_back(timestamp.toNSec());
    results_.invalid_scan_translations.push_back(0);
    results_.invalid_scan_angles.push_back(0);
    return;
  }

  double trans = T_World_WorldEst.block(0, 3, 3, 1).norm();
  if (trans > params_.translation_threshold_m) {
    results_.invalid_scan_stamps.push_back(timestamp.toNSec());
    results_.invalid_scan_translations.push_back(trans);
    results_.invalid_scan_angles.push_back(0);
    return;
  }

  Eigen::Matrix3d R = T_World_WorldEst.block(0, 0, 3, 3);
  Eigen::AngleAxis<double> aa(R);
  double angle = aa.angle() * 180 / M_PI;
  if (angle > params_.rotation_threshold_deg) {
    results_.invalid_scan_stamps.push_back(timestamp.toNSec());
    results_.invalid_scan_translations.push_back(0);
    results_.invalid_scan_angles.push_back(angle);
    return;
  }

  Eigen::Matrix4d T_WORLD_LIDAR = T_World_WorldEst * T_WorldEst_Lidar;
  SaveSuccessfulRegistration(cloud_in_lidar_frame, T_WORLD_LIDAR, timestamp);
}

Eigen::Matrix4d
    ScanPoseGtGeneration::GetT_WorldEst_Lidar(const ros::Time& timestamp) {
  Eigen::Matrix4d T_WORLD_MOVING =
      trajectory_
          .GetTransformEigen(world_frame_id_, moving_frame_id_, timestamp)
          .matrix();
  return T_WORLD_MOVING * T_MOVING_LIDAR_;
}

Eigen::Matrix4d ScanPoseGtGeneration::GetT_MovingLast_MovingCurrent(
    const ros::Time& timestamp_current) {
  Eigen::Matrix4d T_World_MovingCurrent =
      trajectory_
          .GetTransformEigen(world_frame_id_, moving_frame_id_,
                             timestamp_current)
          .matrix();
  Eigen::Matrix4d T_MovingLast_World =
      trajectory_
          .GetTransformEigen(moving_frame_id_, world_frame_id_, timestamp_last_)
          .matrix();
  return T_MovingLast_World * T_World_MovingCurrent;
}

void ScanPoseGtGeneration::SaveSuccessfulRegistration(
    const PointCloud& cloud_in_lidar_frame,
    const Eigen::Matrix4d& T_WORLD_LIDAR, const ros::Time& timestamp) {
  PointCloud cloud_in_world;
  pcl::transformPointCloud(cloud_in_lidar_frame, cloud_in_world, T_WORLD_LIDAR);
  map_ += cloud_in_world;
  current_map_size_++;

  if (current_map_size_ == params_.map_max_size) {
    std::string err;
    std::string map_filename =
        "map_" + std::to_string(results_.saved_cloud_names.size() + 1) + ".pcd";
    std::string map_filepath = beam::CombinePaths(map_save_dir_, map_filename);
    if (beam::SavePointCloud<pcl::PointXYZ>(
            map_filepath, map_, beam::PointCloudFileType::PCDBINARY, err)) {
      BEAM_CRITICAL("unable to save map, reason: {}", err);
      throw std::runtime_error{"unable to save map"};
    }
    map_ = PointCloud();
    current_map_size_ = 0;
    results_.saved_cloud_names.push_back(map_filename);
  }

  results_.valid_scan_stamps.push_back(timestamp.toNSec());

  Eigen::Matrix4d T_WORLD_MOVING =
      T_WORLD_LIDAR * beam::InvertTransform(T_MOVING_LIDAR_);

  // update last successful measurements
  T_World_MovingLast_ = T_WORLD_MOVING;
  timestamp_last_ = timestamp;

  // add to poses
  results_.poses.AddSingleTimeStamp(timestamp);
  results_.poses.AddSinglePose(T_WORLD_MOVING);
}

void ScanPoseGtGeneration::SaveResults() {
  // save map
  std::string err;
  std::string map_filename =
      "map_" + std::to_string(results_.saved_cloud_names.size() + 1) + ".pcd";
  std::string map_filepath = beam::CombinePaths(map_save_dir_, map_filename);
  if (beam::SavePointCloud<pcl::PointXYZ>(
          map_filepath, map_, beam::PointCloudFileType::PCDBINARY, err)) {
    BEAM_CRITICAL("unable to save map, reason: {}", err);
    throw std::runtime_error{"unable to save map"};
  }
  results_.saved_cloud_names.push_back(map_filename);

  nlohmann::json J;
  J["map_names"] = results_.saved_cloud_names;
  J["invalid_scan_stamps"] = results_.invalid_scan_stamps;
  J["invalid_scan_angles"] = results_.invalid_scan_angles;
  J["invalid_scan_translations"] = results_.invalid_scan_translations;
  J["valid_scan_stamps"] = results_.valid_scan_stamps;
  J["num_total_scans"] =
      results_.valid_scan_stamps.size() + results_.invalid_scan_stamps.size();
  J["num_valid_scans"] = results_.valid_scan_stamps.size();
  J["num_invalid_scans"] = results_.invalid_scan_stamps.size();

  std::string filename =
      beam::CombinePaths(inputs_.output_directory, "results_summary.json");
  std::ofstream o(filename);
  o << std::setw(4) << J << std::endl;

  // save poses.json
  results_.poses.SetFixedFrame(world_frame_id_);
  results_.poses.SetMovingFrame(moving_frame_id_);
  results_.poses.WriteToJSON(inputs_.output_directory);
}

} // namespace scan_pose_gt_gen