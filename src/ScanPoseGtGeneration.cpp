#include <scan_pose_gt_generation/ScanPoseGtGeneration.h>

#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <beam_filtering/VoxelDownsample.h>
#include <beam_matching/loam/LoamFeatureExtractor.h>
#include <beam_utils/bspline.h>
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

  if (!J.contains("save_map") || !J.contains("scan_filters") ||
      !J.contains("gt_cloud_filters") || !J.contains("point_size") ||
      !J.contains("extract_loam_points") ||
      !J.contains("max_spline_length_m") ||
      !J.contains("min_spline_measurements")) {
    throw std::runtime_error{
        "missing one or more parameter in the config file"};
  }

  params_.save_map = J["save_map"].get<bool>();
  params_.point_size = J["point_size"].get<int>();
  params_.extract_loam_points = J["extract_loam_points"].get<bool>();
  params_.max_spline_length_m = J["max_spline_length_m"].get<double>();
  params_.min_spline_measurements = J["min_spline_measurements"].get<int>();

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

  Eigen::Matrix4d T_WORLD_GTCLOUD;
  T_WORLD_GTCLOUD << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9],
      v[10], v[11], v[12], v[13], v[14], v[15];
  if (!beam::IsTransformationMatrix(T_WORLD_GTCLOUD)) {
    throw std::runtime_error{
        "T_WORLD_GTCLOUD is not a valid transformation matrix"};
  }
  BEAM_INFO("loaded T_WORLD_GTCLOUD ");

  BEAM_INFO("filtering input cloud");
  PointCloudIRT gt_cloud_in_world;
  pcl::transformPointCloud(gt_cloud_in, gt_cloud_in_world, T_WORLD_GTCLOUD);

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

    input_trajectory_.AddTransform(T_WORLD_MOVINGFRAME, world_frame_id_,
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
  total_scans_ = view.size();
  BEAM_INFO("Read a total of {} pointcloud messages", total_scans_);

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
  Eigen::Matrix4d T_WORLDEST_MOVING;
  if (!GetT_WORLDEST_MOVING(timestamp, T_WORLDEST_MOVING)) { return; }
  Eigen::Matrix4d T_WORLDEST_LIDAR = T_WORLDEST_MOVING * T_MOVING_LIDAR_;

  pcl::transformPointCloud(cloud_filtered_in_lidar_frame, *cloud_in_WorldEst,
                           T_WORLDEST_LIDAR);

  // run ICP
  BEAM_INFO("Running registration for scan {}/{}", scan_counter_, total_scans_);
  timer_.restart();
  PointCloudIRT registered_cloud;
  icp_->setInputSource(cloud_in_WorldEst);
  icp_->align(registered_cloud);
  Eigen::Matrix4d T_WORLD_WORLDEST =
      icp_->getFinalTransformation().cast<double>();
  BEAM_INFO("Registration time: {}s", timer_.elapsed());

  if (!icp_->hasConverged()) {
    DisplayResults(cloud2_in_lidar_frame, Eigen::Matrix4d(), T_WORLDEST_LIDAR,
                   false);
    return;
  }

  Eigen::Matrix4d T_WORLD_LIDAR = T_WORLD_WORLDEST * T_WORLDEST_LIDAR;
  DisplayResults(cloud2_in_lidar_frame, T_WORLD_LIDAR, T_WORLDEST_LIDAR, true);
  AddRegistrationResult(cloud_in_lidar_frame, T_WORLD_LIDAR, timestamp);
  if (IsMapFull()) {
    FitSplineToTrajectory();
    SaveMaps();
    UpdateT_INIT_SPLINE();
    trajectories_raw_.push_back(Trajectory());
    trajectories_spline_.push_back(Trajectory());
  }
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

void ScanPoseGtGeneration::UpdateT_INIT_SPLINE() {
  const Trajectory& traj_last = trajectories_spline_.back();
  const std::vector<beam::Pose>& poses_last = traj_last.GetPoses();
  const beam::Pose& pose_last = poses_last.back();
  Eigen::Matrix4d T_WORLD_MOVINGLASTSPLINE = pose_last.T_FIXED_MOVING;
  ros::Time timestamp_last_spline;
  timestamp_last_spline.fromNSec(pose_last.timestampInNs);
  Eigen::Matrix4d T_WORLD_MOVINGLASTINIT;
  if (!GetT_WORLDESTINIT_MOVING(timestamp_last_spline,
                                T_WORLD_MOVINGLASTINIT)) {
    throw std::runtime_error{"cannot update T_INIT_SPLINE"};
  }
  T_INIT_SPLINE_ =
      beam::InvertTransform(T_WORLD_MOVINGLASTINIT) * T_WORLD_MOVINGLASTSPLINE;
}

bool ScanPoseGtGeneration::GetT_WORLDEST_MOVING(
    const ros::Time& timestamp, Eigen::Matrix4d& T_WORLD_MOVING) {
  Eigen::Matrix4d T_WORLD_MOVING_INIT;
  bool successful = GetT_WORLDESTINIT_MOVING(timestamp, T_WORLD_MOVING_INIT);
  if (!successful) { return false; }

  if (trajectories_raw_.size() < 2) {
    T_WORLD_MOVING = T_WORLD_MOVING_INIT;
    return true;
  }

  // get pose relative to end of last trajectory
  T_WORLD_MOVING = T_WORLD_MOVING_INIT * T_INIT_SPLINE_;
  return true;
}

bool ScanPoseGtGeneration::GetT_WORLDESTINIT_MOVING(
    const ros::Time& timestamp, Eigen::Matrix4d& T_WORLD_MOVING) {
  try {
    T_WORLD_MOVING =
        input_trajectory_
            .GetTransformEigen(world_frame_id_, moving_frame_id_, timestamp)
            .matrix();
  } catch (...) {
    BEAM_WARN("skipping scan");
    return false;
  }
  return true;
}

void ScanPoseGtGeneration::AddRegistrationResult(
    const PointCloudIRT& cloud_in_lidar_frame,
    const Eigen::Matrix4d& T_WORLD_LIDAR, const ros::Time& timestamp) {
  int64_t timeInNs = timestamp.toNSec();
  Eigen::Matrix4d T_WORLD_MOVING =
      T_WORLD_LIDAR * beam::InvertTransform(T_MOVING_LIDAR_);
  if (trajectories_raw_.empty()) { trajectories_raw_.push_back(Trajectory()); }
  trajectories_raw_.rbegin()->AddPose(timeInNs, T_WORLD_MOVING);
  if (!params_.save_map) { return; }
  current_traj_scans_in_lidar_.emplace(timeInNs, cloud_in_lidar_frame);
}

bool ScanPoseGtGeneration::IsMapFull() {
  const Trajectory& curr_traj =
      trajectories_raw_.at(trajectories_raw_.size() - 1);

  // check minimum number of measurements
  if (curr_traj.Size() < params_.min_spline_measurements) { return false; }

  // check trajectory length
  if (curr_traj.Length() < params_.max_spline_length_m) { return false; }

  return true;
}

void ScanPoseGtGeneration::FitSplineToTrajectory() {
  const Trajectory& traj_raw = trajectories_raw_.back();
  BEAM_INFO("fitting spline to trajectory with {} poses", traj_raw.Size());
  timer_.restart();
  beam::BsplineSE3 spline;
  spline.feed_trajectory(traj_raw.GetPoses());
  BEAM_INFO("done fitting spline in {}s", timer_.elapsed());

  Trajectory new_traj;
  for (const beam::Pose& p : traj_raw.GetPoses()) {
    double time = static_cast<double>(p.timestampInNs * 1e-9);
    Eigen::Matrix4d T_FIXED_MOVING;
    spline.get_pose(time, T_FIXED_MOVING);
    new_traj.AddPose(p.timestampInNs, T_FIXED_MOVING);
  }
  trajectories_spline_.push_back(new_traj);
  BEAM_INFO("created spline trajectory with {} pose", new_traj.Size());
}

void ScanPoseGtGeneration::SaveMaps() {
  std::string map_filename1 =
      "map_raw_" + std::to_string(trajectories_raw_.size());
  SaveMap(*trajectories_raw_.rbegin(), map_filename1);
  trajectories_raw_.rbegin()->map_filename = map_filename1;

  std::string map_filename2 =
      "map_spline_" + std::to_string(trajectories_spline_.size());
  SaveMap(*trajectories_spline_.rbegin(), map_filename2);
  trajectories_spline_.rbegin()->map_filename = map_filename2;
  current_traj_scans_in_lidar_.clear();
}

void ScanPoseGtGeneration::SaveMap(const Trajectory& trajectory,
                                   const std::string& name) {
  if (!params_.save_map) { return; }

  PointCloudIRT map;
  for (const beam::Pose& pose : trajectory.GetPoses()) {
    const PointCloudIRT& cloud_in_lidar =
        current_traj_scans_in_lidar_.at(pose.timestampInNs);
    PointCloudIRT cloud_in_world;
    Eigen::Matrix4d T_WORLD_LIDAR = pose.T_FIXED_MOVING * T_MOVING_LIDAR_;
    pcl::transformPointCloud(cloud_in_lidar, cloud_in_world, T_WORLD_LIDAR);
    map += cloud_in_world;
  }

  std::string err;
  std::string map_filepath = beam::CombinePaths(map_save_dir_, name + ".pcd");
  BEAM_INFO("saving map of size {} to: {}", map.size(), map_filepath);
  if (!beam::SavePointCloud<PointXYZIRT>(
          map_filepath, map, beam::PointCloudFileType::PCDBINARY, err)) {
    BEAM_CRITICAL("unable to save map, reason: {}", err);
    throw std::runtime_error{"unable to save map"};
  }
}

void ScanPoseGtGeneration::SaveTrajectories(
    const std::vector<Trajectory>& trajectory, const std::string& name) {
  std::vector<std::string> trajectory_names;
  beam_mapping::Poses poses_all;
  int counter{0};
  for (const Trajectory& t : trajectory) {
    if (t.Size() == 0) { continue; }
    counter++;
    beam_mapping::Poses poses_current;
    for (const beam::Pose& p : t.GetPoses()) {
      ros::Time time_ros;
      time_ros.fromNSec(p.timestampInNs);
      poses_current.AddSingleTimeStamp(time_ros);
      poses_current.AddSinglePose(p.T_FIXED_MOVING);
      poses_all.AddSingleTimeStamp(time_ros);
      poses_all.AddSinglePose(p.T_FIXED_MOVING);
    }
    poses_current.SetFixedFrame(world_frame_id_);
    poses_current.SetMovingFrame(moving_frame_id_);
    std::string poses_name = beam::CombinePaths(inputs_.output_directory,
                                                t.map_filename + "_poses.json");
    BEAM_INFO("saving {} poses for trajectory {}",
              poses_current.GetTimeStamps().size(), counter);
    poses_current.WriteToFile(poses_name, "JSON");
    trajectory_names.push_back(poses_name);
    poses_name = beam::CombinePaths(inputs_.output_directory,
                                    t.map_filename + "_poses.pcd");
    poses_current.WriteToFile(poses_name, "PCD");
  }

  // save list of trajectories
  nlohmann::json J;
  J["trajectories"] = trajectory_names;
  std::string filename =
      beam::CombinePaths(inputs_.output_directory, name + "_list.json");
  std::ofstream o(filename);
  o << std::setw(4) << J << std::endl;

  // save combined
  poses_all.SetFixedFrame(world_frame_id_);
  poses_all.SetMovingFrame(moving_frame_id_);
  std::string combined_name =
      beam::CombinePaths(inputs_.output_directory, name + "_poses_combined");
  poses_all.WriteToFile(combined_name + ".json", "JSON");
  poses_all.WriteToFile(combined_name + ".pcd", "PCD");
}

void ScanPoseGtGeneration::SaveResults() {
  SaveTrajectories(trajectories_raw_, "trajectory_raw");
  SaveTrajectories(trajectories_spline_, "trajectories_spline");

  // copy over files to output
  std::string output_config =
      beam::CombinePaths(inputs_.output_directory, "config_copy.json");
  BEAM_INFO("copying config file to: {}", output_config);
  boost::filesystem::copy_file(
      inputs_.config, output_config,
      boost::filesystem::copy_option::overwrite_if_exists);
  std::string output_gt_cloud =
      beam::CombinePaths(inputs_.output_directory, "gt_cloud_copy.pcd");
  BEAM_INFO("copying gt cloud to: {}", output_gt_cloud);
  boost::filesystem::copy_file(
      inputs_.gt_cloud, output_gt_cloud,
      boost::filesystem::copy_option::overwrite_if_exists);
  std::string output_gt_cloud_pose =
      beam::CombinePaths(inputs_.output_directory, "gt_cloud_pose_copy.json");
  BEAM_INFO("copying gt cloud pose file to: {}", output_gt_cloud_pose);
  boost::filesystem::copy_file(
      inputs_.gt_cloud_pose, output_gt_cloud_pose,
      boost::filesystem::copy_option::overwrite_if_exists);
}

void ScanPoseGtGeneration::DisplayResults(
    const PointCloudIRT& cloud_in_lidar,
    const Eigen::Matrix4d& T_WORLDOPT_LIDAR,
    const Eigen::Matrix4d& T_WORLDEST_LIDAR, bool successful) {
  if (!inputs_.visualize) { return; }

  viewer_->removePointCloud("ScanAligned");
  viewer_->removePointCloud("ScanInitial");

  PointCloudIRT::Ptr cloud_initial = std::make_shared<PointCloudIRT>();
  pcl::transformPointCloud(cloud_in_lidar, *cloud_initial, T_WORLDEST_LIDAR);
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
    pcl::transformPointCloud(cloud_in_lidar, *cloud_aligned, T_WORLDOPT_LIDAR);
    pcl::visualization::PointCloudColorHandlerCustom<PointXYZIRT> fin_col(
        cloud_aligned, 0, 255, 0);
    viewer_->addPointCloud<PointXYZIRT>(cloud_aligned, fin_col, "ScanAligned");
    viewer_->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, params_.point_size,
        "ScanAligned");
  } else {
    std::cout << "Showing unsuccessful ICP results.\n"
              << "Press 'n' to skip to next scan\n";
  }

  while (!viewer_->wasStopped() && !next_scan_) { viewer_->spinOnce(); }
  next_scan_ = false;
}

} // namespace scan_pose_gt_gen