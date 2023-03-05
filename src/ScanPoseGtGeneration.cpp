#include <scan_pose_gt_generation/ScanPoseGtGeneration.h>

#include <fstream>

#include <boost/filesystem.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <beam_filtering/VoxelDownsample.h>
#include <beam_matching/loam/LoamFeatureExtractor.h>
#include <beam_utils/log.h>
#include <beam_utils/math.h>
#include <beam_utils/time.h>

namespace scan_pose_gt_gen {

void ScanPoseGtGeneration::run() {
  LoadConfig();
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
      !J.contains("gt_cloud_filters") || !J.contains("point_size") ||
      !J.contains("extract_loam_points")) {
    throw std::runtime_error{
        "missing one or more parameter in the config file"};
  }

  params_.map_max_size = J["map_max_size"].get<int>();
  params_.save_map = J["save_map"].get<bool>();
  params_.rotation_threshold_deg = J["rotation_threshold_deg"].get<double>();
  params_.translation_threshold_m = J["translation_threshold_m"].get<double>();
  params_.point_size = J["point_size"].get<int>();
  params_.extract_loam_points = J["extract_loam_points"].get<bool>();

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
  BEAM_INFO("Extracted T_MOVING_LIDAR:");
  std::cout << T_MOVING_LIDAR_ << "\n";
}

void ScanPoseGtGeneration::LoadGtCloud() {
  BEAM_INFO("Loading gt cloud: {}", inputs_.gt_cloud);
  PointCloudIRT gt_cloud_in;
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
  BEAM_INFO("loaded T_World_GtCloud: ");
  std::cout << T_World_GtCloud << "\n";
  T_World_GtCloud_ = T_World_GtCloud;
  PointCloudIRT gt_cloud_in_world;
  pcl::transformPointCloud(gt_cloud_in, gt_cloud_in_world, T_World_GtCloud);

  gt_cloud_in_world_ = std::make_shared<PointCloudIRT>(
      beam_filtering::FilterPointCloud<PointXYZIRT>(gt_cloud_in_world,
                                                    gt_cloud_filters_));

  if (inputs_.visualize) {
    beam_filtering::VoxelDownsample<PointXYZIRT> filter(
        Eigen::Vector3f(0.05, 0.05, 0.05));
    filter.SetInputCloud(gt_cloud_in_world_);
    filter.Filter();
    PointCloudIRT::Ptr cloud_filtererd = std::make_shared<PointCloudIRT>();
    *cloud_filtererd = filter.GetFilteredCloud();
    viewer_ = std::make_unique<pcl::visualization::PCLVisualizer>();
    pcl::visualization::PointCloudColorHandlerCustom<PointXYZIRT> col(
        cloud_filtererd, 255, 255, 255);
    viewer_->addPointCloud<PointXYZIRT>(cloud_filtererd, col, "GTMap");
    viewer_->addCoordinateSystem(1.0);

    std::function<void(const pcl::visualization::KeyboardEvent&)> keyboard_cb =
        [this](const pcl::visualization::KeyboardEvent& event) {
          keyboardEventOccurred(event);
        };

    viewer_->registerKeyboardCallback(keyboard_cb);
  }
}

void ScanPoseGtGeneration::keyboardEventOccurred(
    const pcl::visualization::KeyboardEvent& event) {
  if (event.getKeySym() == "n" && event.keyDown()) { next_scan_ = true; }
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
    if (!beam::IsTransformationMatrix(T_WORLD_MOVINGFRAME.matrix())) {
      BEAM_ERROR("Invalid transformation matrix read from poses, skipping");
      continue;
    }

    trajectory_.AddTransform(T_WORLD_MOVINGFRAME, world_frame_id_,
                             moving_frame_id_, poses.GetTimeStamps()[k]);
  }
}

void ScanPoseGtGeneration::SetupRegistration() {
  BEAM_INFO("Setting up registration");
  icp_ = std::make_unique<IcpType>();
  icp_->setInputTarget(gt_cloud_in_world_);
  icp_->setMaxCorrespondenceDistance(params_.icp_params.max_corr_dist);
  icp_->setMaximumIterations(params_.icp_params.max_iterations);
  icp_->setTransformationEpsilon(params_.icp_params.transform_eps);
  icp_->setEuclideanFitnessEpsilon(params_.icp_params.fitness_eps);
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

  // get lidar sensor frame from first message to calculate extrinsics
  {
    auto first_msg = view.begin()->instantiate<sensor_msgs::PointCloud2>();
    lidar_frame_id_ = first_msg->header.frame_id;
  }
  LoadExtrinsics();

  // iterate through bag and run registration
  for (auto iter = view.begin(); iter != view.end(); iter++) {
    scan_counter_++;
    auto sensor_msg = iter->instantiate<sensor_msgs::PointCloud2>();
    ros::Time timestamp = sensor_msg->header.stamp;
    pcl::PCLPointCloud2::Ptr pcl_pc2_tmp =
        std::make_shared<pcl::PCLPointCloud2>();
    PointCloudIRT cloud;
    beam::pcl_conversions::toPCL(*sensor_msg, *pcl_pc2_tmp);
    pcl::fromPCLPointCloud2(*pcl_pc2_tmp, cloud);
    RegisterSingleScan(cloud, timestamp);
  }
}

void ScanPoseGtGeneration::RegisterSingleScan(
    const PointCloudIRT& cloud_in_lidar_frame, const ros::Time& timestamp) {
  PointCloudIRT cloud2_in_lidar_frame =
      ExtractStrongLoamPoints(cloud_in_lidar_frame);

  // filter cloud and transform to estimated world frame
  PointCloudIRT cloud_filtered_in_lidar_frame =
      beam_filtering::FilterPointCloud<PointXYZIRT>(cloud2_in_lidar_frame,
                                                    scan_filters_);
  PointCloudIRT::Ptr cloud_in_WorldEst = std::make_shared<PointCloudIRT>();

  // if first scan, get estimated pose straight from poses. Otherwise, get only
  // relative pose from poses
  Eigen::Matrix4d T_WorldEst_Lidar;
  if (is_first_scan_) {
    if (!GetT_WorldEst_Lidar(timestamp, T_WorldEst_Lidar)) { return; }
    is_first_scan_ = false;
  } else {
    Eigen::Matrix4d T_MovingLast_MovingCurrent;
    if (!GetT_MovingLast_MovingCurrent(timestamp, T_MovingLast_MovingCurrent)) {
      return;
    }
    T_WorldEst_Lidar =
        T_World_MovingLast_ * T_MovingLast_MovingCurrent * T_MOVING_LIDAR_;
  }

  pcl::transformPointCloud(cloud_filtered_in_lidar_frame, *cloud_in_WorldEst,
                           T_WorldEst_Lidar);

  // run ICP
  BEAM_INFO("Running registration for scan {}, timestamp: {} Ns", scan_counter_,
            timestamp.toNSec());
  timer_.restart();
  PointCloudIRT registered_cloud;
  icp_->setInputSource(cloud_in_WorldEst);
  icp_->align(registered_cloud);
  Eigen::Matrix4d T_World_WorldEst =
      icp_->getFinalTransformation().cast<double>();
  BEAM_INFO("Registration time: {}s", timer_.elapsed());

  bool failed_registration{false};
  std::string icp_results_str;
  if (!icp_->hasConverged()) {
    results_.invalid_scan_stamps.push_back(timestamp.toNSec());
    results_.invalid_scan_translations.push_back(0);
    results_.invalid_scan_angles.push_back(0);
    icp_results_str = "No convergence";
    failed_registration = true;
  } else {
    double trans = T_World_WorldEst.block(0, 3, 3, 1).norm();
    Eigen::Matrix3d R = T_World_WorldEst.block(0, 0, 3, 3);
    Eigen::AngleAxis<double> aa(R);
    double angle = aa.angle() * 180 / M_PI;
    if (trans > params_.translation_threshold_m ||
        angle > params_.rotation_threshold_deg) {
      results_.invalid_scan_stamps.push_back(timestamp.toNSec());
      results_.invalid_scan_translations.push_back(trans);
      results_.invalid_scan_angles.push_back(angle);
      icp_results_str = "Invalid result: t =  " + std::to_string(trans) +
                        ", r = " + std::to_string(angle);
      failed_registration = true;
    }
  }

  if (failed_registration) {
    DisplayResults(cloud2_in_lidar_frame, Eigen::Matrix4d(), T_WorldEst_Lidar,
                   false, icp_results_str);
    return;
  }

  Eigen::Matrix4d T_WORLD_LIDAR = T_World_WorldEst * T_WorldEst_Lidar;
  DisplayResults(cloud2_in_lidar_frame, T_WORLD_LIDAR, T_WorldEst_Lidar, true);
  SaveSuccessfulRegistration(cloud_in_lidar_frame, T_WORLD_LIDAR, timestamp);
}

PointCloudIRT ScanPoseGtGeneration::ExtractStrongLoamPoints(
    const PointCloudIRT& cloud_in) {
  if (!params_.extract_loam_points) { return cloud_in; }
  beam_matching::LoamParamsPtr params =
      std::make_shared<beam_matching::LoamParams>();
  beam_matching::LoamFeatureExtractor extractor(params);
  beam_matching::LoamPointCloud loam_cloud =
      extractor.ExtractFeatures(cloud_in);
  loam_cloud.edges.weak.Clear();
  loam_cloud.surfaces.weak.Clear();
  PointCloudIRT cloud_out;
  pcl::copyPointCloud(loam_cloud.GetCombinedCloud(), cloud_out);
  return cloud_out;
}

bool ScanPoseGtGeneration::GetT_WorldEst_Lidar(const ros::Time& timestamp,
                                               Eigen::Matrix4d& T_WORLD_LIDAR) {
  Eigen::Matrix4d T_WORLD_MOVING;
  try {
    T_WORLD_MOVING =
        trajectory_
            .GetTransformEigen(world_frame_id_, moving_frame_id_, timestamp)
            .matrix();
  } catch (...) {
    BEAM_WARN("skipping scan");
    return false;
  }

  T_WORLD_LIDAR = T_WORLD_MOVING * T_MOVING_LIDAR_;
  return true;
}

bool ScanPoseGtGeneration::GetT_MovingLast_MovingCurrent(
    const ros::Time& timestamp_current,
    Eigen::Matrix4d& T_MovingLast_MovingCurrent) {
  try {
    Eigen::Matrix4d T_World_MovingCurrent =
        trajectory_
            .GetTransformEigen(world_frame_id_, moving_frame_id_,
                               timestamp_current)
            .matrix();
    Eigen::Matrix4d T_MovingLast_World =
        trajectory_
            .GetTransformEigen(moving_frame_id_, world_frame_id_,
                               timestamp_last_)
            .matrix();

    T_MovingLast_MovingCurrent = T_MovingLast_World * T_World_MovingCurrent;
    return true;
  } catch (...) { return false; }
}

void ScanPoseGtGeneration::SaveSuccessfulRegistration(
    const PointCloudIRT& cloud_in_lidar_frame,
    const Eigen::Matrix4d& T_WORLD_LIDAR, const ros::Time& timestamp) {
  if (!params_.save_map) { return; }
  PointCloudIRT cloud_in_world;
  pcl::transformPointCloud(cloud_in_lidar_frame, cloud_in_world, T_WORLD_LIDAR);
  map_ += cloud_in_world;
  current_map_size_++;

  if (current_map_size_ == params_.map_max_size) {
    std::string map_filename =
        "map_" + std::to_string(results_.saved_cloud_names.size() + 1) + ".pcd";
    std::string map_filepath = beam::CombinePaths(map_save_dir_, map_filename);
    BEAM_INFO("saving map to: {}", map_filepath);
    std::string err;
    if (!beam::SavePointCloud<PointXYZIRT>(
            map_filepath, map_, beam::PointCloudFileType::PCDBINARY, err)) {
      BEAM_CRITICAL("unable to save map, reason: {}", err);
      throw std::runtime_error{"unable to save map"};
    }

    map_ = PointCloudIRT();
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
  BEAM_INFO("saving map to: {}", map_filepath);
  if (!beam::SavePointCloud<PointXYZIRT>(
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

  // copy over files to output
  std::string output_config = beam::CombinePaths(inputs_.output_directory, "config_copy.json");
  BEAM_INFO("copying config file to: {}", output_config);
  boost::filesystem::copy_file(inputs_.config, output_config);
  std::string output_gt_cloud = beam::CombinePaths(inputs_.output_directory, "gt_cloud_copy.pcd");
  BEAM_INFO("copying gt cloud to: {}", output_gt_cloud);
  boost::filesystem::copy_file(inputs_.gt_cloud, output_gt_cloud);
  std::string output_gt_cloud_pose = beam::CombinePaths(inputs_.output_directory, "gt_cloud_pose_copy.json");
  BEAM_INFO("copying gt cloud pose file to: {}", output_gt_cloud_pose);
  boost::filesystem::copy_file(inputs_.gt_cloud_pose, output_gt_cloud_pose);
  
}

void ScanPoseGtGeneration::DisplayResults(
    const PointCloudIRT& cloud_in_lidar,
    const Eigen::Matrix4d& T_WorldOpt_Lidar,
    const Eigen::Matrix4d& T_WorldEst_Lidar, bool successful,
    const std::string& icp_results) {
  if (!inputs_.visualize) { return; }

  viewer_->removePointCloud("ScanAligned");
  viewer_->removePointCloud("ScanInitial");

  PointCloudIRT::Ptr cloud_initial = std::make_shared<PointCloudIRT>();
  pcl::transformPointCloud(cloud_in_lidar, *cloud_initial, T_WorldEst_Lidar);
  pcl::visualization::PointCloudColorHandlerCustom<PointXYZIRT> init_col(
      cloud_initial, 255, 0, 0);
  viewer_->addPointCloud<PointXYZIRT>(cloud_initial, init_col, "ScanInitial");
  viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, params_.point_size,
      "ScanInitial");
  if (successful) {
    std::cout << "Showing successful ICP results\n"
              << "Press 'n' to skip to next scan\n";
    PointCloudIRT::Ptr cloud_aligned = std::make_shared<PointCloudIRT>();
    pcl::transformPointCloud(cloud_in_lidar, *cloud_aligned, T_WorldOpt_Lidar);
    pcl::visualization::PointCloudColorHandlerCustom<PointXYZIRT> fin_col(
        cloud_aligned, 0, 255, 0);
    viewer_->addPointCloud<PointXYZIRT>(cloud_aligned, fin_col, "ScanAligned");
    viewer_->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, params_.point_size,
        "ScanAligned");
  } else {
    std::cout << "Showing unsuccessful ICP results.\n"
              << "Reason for failure: " << icp_results << "\n"
              << "Press 'n' to skip to next scan\n";
  }

  while (!viewer_->wasStopped() && !next_scan_) { viewer_->spinOnce(); }
  next_scan_ = false;
}

} // namespace scan_pose_gt_gen