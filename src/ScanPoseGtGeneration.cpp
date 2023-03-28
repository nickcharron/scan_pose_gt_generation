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
      !J.contains("scan_filters_refinement") ||
      !J.contains("gt_cloud_filters") ||
      !J.contains("gt_cloud_filters_refinement") ||
      !J.contains("output_filters") || !J.contains("point_size") ||
      !J.contains("extract_loam_points") ||
      !J.contains("max_spline_length_m") ||
      !J.contains("min_spline_measurements") || !J.contains("run_refinement")) {
    throw std::runtime_error{
        "missing one or more parameter in the config file"};
  }

  params_.save_map = J["save_map"].get<bool>();
  params_.point_size = J["point_size"].get<int>();
  params_.extract_loam_points = J["extract_loam_points"].get<bool>();
  params_.max_spline_length_m = J["max_spline_length_m"].get<double>();
  params_.min_spline_measurements = J["min_spline_measurements"].get<int>();
  params_.run_refinement = J["run_refinement"].get<bool>();

  std::string date = beam::ConvertTimeToDate(std::chrono::system_clock::now());
  root_save_dir_ = beam::CombinePaths(inputs_.output_directory, date);
  boost::filesystem::create_directory(root_save_dir_);
  poses_save_dir_ = beam::CombinePaths(root_save_dir_, "submap_poses");
  boost::filesystem::create_directory(poses_save_dir_);

  if (params_.save_map) {
    map_save_dir_ = beam::CombinePaths(root_save_dir_, "submaps");
    boost::filesystem::create_directory(map_save_dir_);
  }

  scan_filters_ = beam_filtering::LoadFilterParamsVector(J["scan_filters"]);
  scan_filters_refinement_ =
      beam_filtering::LoadFilterParamsVector(J["scan_filters_refinement"]);
  gt_cloud_filters_ =
      beam_filtering::LoadFilterParamsVector(J["gt_cloud_filters"]);
  gt_cloud_filters_refinement_ =
      beam_filtering::LoadFilterParamsVector(J["gt_cloud_filters_refinement"]);
  output_filters_ = beam_filtering::LoadFilterParamsVector(J["output_filters"]);
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
  gt_cloud_refinement_in_world_ = std::make_shared<PointCloudIRT>(
      beam_filtering::FilterPointCloud<PointXYZIRT>(
          gt_cloud_in_world, gt_cloud_filters_refinement_));

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

  start_time_ns =
      poses.GetTimeStamps().front().toNSec() + inputs_.start_offset_s;
  end_time_ns = poses.GetTimeStamps().back().toNSec() - inputs_.end_offset_s;
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
  trajectories_raw_.push_back(Trajectory());
  trajectories_spline_.push_back(Trajectory());

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
    ProcessSingleScan(cloud, timestamp);
  }
  BEAM_INFO("done adding scans, registering final trajectory");

  // process remainder of scans
  if (inputs_.visualize) {
    RegisterCurrentTrajectoryScans();
  } else {
    RegisterCurrentTrajectoryScansInParallel();
  }
  FitSplineToTrajectory();
  SaveMaps();
}

void ScanPoseGtGeneration::ProcessSingleScan(
    const PointCloudIRT& cloud_in_lidar_frame, const ros::Time& timestamp) {
  int64_t timeInNs = timestamp.toNSec();
  if (timeInNs < start_time_ns || timeInNs > end_time_ns) { return; }

  Eigen::Matrix4d T_WORLDEST_MOVING;
  if (!GetT_WORLD_MOVING(timestamp, T_WORLDEST_MOVING)) { return; }

  trajectories_raw_.rbegin()->AddPose(timeInNs, T_WORLDEST_MOVING);
  current_traj_scans_in_lidar_.emplace(timeInNs, cloud_in_lidar_frame);
  if (IsMapFull()) {
    if (inputs_.visualize) {
      RegisterCurrentTrajectoryScans();
    } else {
      RegisterCurrentTrajectoryScansInParallel();
    }
    FitSplineToTrajectory();
    RunRefinement();
    FitSplineToTrajectory();
    SaveMaps();
    trajectories_raw_.push_back(Trajectory());
    trajectories_spline_.push_back(Trajectory());
  }
}

void ScanPoseGtGeneration::RegisterCurrentTrajectoryScans() {
  auto& curr_traj = *trajectories_raw_.rbegin();
  for (int i = 0; i < current_traj_scans_in_lidar_.size(); i++) {
    // setup registration (needed for parallelization)
    auto icp = std::make_unique<IcpType>();
    icp->setInputTarget(gt_cloud_in_world_);
    icp->setMaxCorrespondenceDistance(params_.icp_params.max_corr_dist);
    icp->setMaximumIterations(params_.icp_params.max_iterations);
    icp->setTransformationEpsilon(params_.icp_params.transform_eps);
    icp->setEuclideanFitnessEpsilon(params_.icp_params.fitness_eps);

    // get current scan
    auto it = current_traj_scans_in_lidar_.begin();
    std::advance(it, i);
    int64_t timestampInNs = it->first;
    const auto& cloud_in_lidar_frame = it->second;

    // get current scan pose
    Eigen::Matrix4d T_WORLDEST_MOVING;
    curr_traj.GetPose(timestampInNs, T_WORLDEST_MOVING);
    Eigen::Matrix4d T_WORLDEST_LIDAR = T_WORLDEST_MOVING * T_MOVING_LIDAR_;

    // filter cloud and transform to estimated world frame
    PointCloudIRT cloud2_in_lidar_frame =
        ExtractStrongLoamPoints(cloud_in_lidar_frame);
    PointCloudIRT cloud_filtered_in_lidar_frame =
        beam_filtering::FilterPointCloud<PointXYZIRT>(cloud2_in_lidar_frame,
                                                      scan_filters_);
    PointCloudIRT::Ptr cloud_in_WorldEst = std::make_shared<PointCloudIRT>();
    pcl::transformPointCloud(cloud_filtered_in_lidar_frame, *cloud_in_WorldEst,
                             T_WORLDEST_LIDAR);

    // run ICP
    BEAM_INFO("Running registration for scan {}/{}", scan_counter_,
              total_scans_);
    beam::HighResolutionTimer timer;
    timer.restart();
    PointCloudIRT registered_cloud;
    icp->setInputSource(cloud_in_WorldEst);
    icp->align(registered_cloud);
    Eigen::Matrix4d T_WORLD_WORLDEST =
        icp->getFinalTransformation().cast<double>();
    BEAM_INFO("Registration time: {}s", timer.elapsed());

    if (!icp->hasConverged()) {
      DisplayResults(cloud_filtered_in_lidar_frame, Eigen::Matrix4d(),
                     T_WORLDEST_LIDAR, false);
      curr_traj.RemovePose(timestampInNs);
      continue;
    }

    Eigen::Matrix4d T_WORLD_MOVING = T_WORLD_WORLDEST * T_WORLDEST_MOVING;
    curr_traj.UpdatePose(timestampInNs, T_WORLD_MOVING);

    Eigen::Matrix4d T_WORLD_LIDAR = T_WORLD_WORLDEST * T_WORLDEST_LIDAR;
    DisplayResults(cloud_filtered_in_lidar_frame, T_WORLD_LIDAR,
                   T_WORLDEST_LIDAR, true);
  }
}

void ScanPoseGtGeneration::RegisterCurrentTrajectoryScansInParallel() {
  auto& curr_traj = *trajectories_raw_.rbegin();

#pragma omp parallel
#pragma omp for
  for (int i = 0; i < current_traj_scans_in_lidar_.size(); i++) {
    // setup registration (needed for parallelization)
    auto icp = std::make_unique<IcpType>();
    icp->setInputTarget(gt_cloud_in_world_);
    icp->setMaxCorrespondenceDistance(params_.icp_params.max_corr_dist);
    icp->setMaximumIterations(params_.icp_params.max_iterations);
    icp->setTransformationEpsilon(params_.icp_params.transform_eps);
    icp->setEuclideanFitnessEpsilon(params_.icp_params.fitness_eps);

    // get current scan
    auto it = current_traj_scans_in_lidar_.begin();
    std::advance(it, i);
    int64_t timestampInNs = it->first;
    const auto& cloud_in_lidar_frame = it->second;

    // get current scan pose
    Eigen::Matrix4d T_WORLDEST_MOVING;
    curr_traj.GetPose(timestampInNs, T_WORLDEST_MOVING);
    Eigen::Matrix4d T_WORLDEST_LIDAR = T_WORLDEST_MOVING * T_MOVING_LIDAR_;

    // filter cloud and transform to estimated world frame
    PointCloudIRT cloud2_in_lidar_frame =
        ExtractStrongLoamPoints(cloud_in_lidar_frame);
    PointCloudIRT cloud_filtered_in_lidar_frame =
        beam_filtering::FilterPointCloud<PointXYZIRT>(cloud2_in_lidar_frame,
                                                      scan_filters_);
    PointCloudIRT::Ptr cloud_in_WorldEst = std::make_shared<PointCloudIRT>();
    pcl::transformPointCloud(cloud_filtered_in_lidar_frame, *cloud_in_WorldEst,
                             T_WORLDEST_LIDAR);

    // run ICP
    BEAM_INFO("Running registration for scan {}/{}", i, total_scans_);
    beam::HighResolutionTimer timer;
    timer.restart();
    PointCloudIRT registered_cloud;
    icp->setInputSource(cloud_in_WorldEst);
    icp->align(registered_cloud);
    Eigen::Matrix4d T_WORLD_WORLDEST =
        icp->getFinalTransformation().cast<double>();
    BEAM_INFO("Registration time: {}s", timer.elapsed());

    if (!icp->hasConverged()) {
      BEAM_ERROR("current scan registration did not converge (t = {}Ns)",
                 timestampInNs);
      curr_traj.RemovePose(timestampInNs);
      continue;
    }

    Eigen::Matrix4d T_WORLD_MOVING = T_WORLD_WORLDEST * T_WORLDEST_MOVING;
    curr_traj.UpdatePose(timestampInNs, T_WORLD_MOVING);
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
  // loam_cloud.edges.weak.Clear();
  // loam_cloud.surfaces.weak.Clear();
  PointCloudIRT cloud_out;
  pcl::copyPointCloud(loam_cloud.GetCombinedCloud(), cloud_out);
  return cloud_out;
}

bool ScanPoseGtGeneration::GetT_WORLD_MOVING(const ros::Time& timestamp,
                                             Eigen::Matrix4d& T_WORLD_MOVING) {
  Eigen::Matrix4d T_WORLDINIT_MOVING;
  bool successful = GetT_WORLD_MOVING_INIT(timestamp, T_WORLDINIT_MOVING);
  if (!successful) { return false; }
  T_WORLD_MOVING = T_WORLD_WORLDINIT_AVG_ * T_WORLDINIT_MOVING;
  return true;
}

bool ScanPoseGtGeneration::GetT_WORLD_MOVING_INIT(
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
  Trajectory& traj_spline = trajectories_spline_.back();
  traj_spline = Trajectory();
  BEAM_INFO("fitting spline to trajectory with {} poses", traj_raw.Size());
  timer_.restart();
  beam::BsplineSE3 spline;
  spline.feed_trajectory(traj_raw.GetPoses());
  BEAM_INFO("done fitting spline in {}s", timer_.elapsed());

  for (const auto& [t, p] : traj_raw.GetPoses()) {
    double time = static_cast<double>(p.timestampInNs * 1e-9);
    Eigen::Matrix4d T_FIXED_MOVING;
    if (spline.get_pose(time, T_FIXED_MOVING)) {
      traj_spline.AddPose(p.timestampInNs, T_FIXED_MOVING);
    }
  }
  BEAM_INFO("created/updated spline trajectory with {} pose",
            traj_spline.Size());

  // update drift: T_WORLD_WORLDINIT_AVG_
  std::vector<Eigen::Matrix4d, beam::AlignMat4d> T_WORLD_WORLDINIT_VEC;
  for (const auto& [t, p] : traj_spline.GetPoses()) {
    ros::Time rt;
    rt.fromNSec(p.timestampInNs);
    const Eigen::Matrix4d& T_WORLD_MOVING = p.T_FIXED_MOVING;
    Eigen::Matrix4d T_WORLDINIT_MOVING;
    if (GetT_WORLD_MOVING_INIT(rt, T_WORLDINIT_MOVING)) {
      T_WORLD_WORLDINIT_VEC.push_back(
          T_WORLD_MOVING * beam::InvertTransform(T_WORLDINIT_MOVING));
    }
  }
  T_WORLD_WORLDINIT_AVG_ = beam::AverageTransforms(T_WORLD_WORLDINIT_VEC);
}

void ScanPoseGtGeneration::RunRefinement() {
  if (!params_.run_refinement) { return; }
  Trajectory& traj_raw = trajectories_raw_.back();
  const Trajectory& traj_spline = trajectories_spline_.back();
  int counter{0};
  beam::HighResolutionTimer timer1;
  timer1.restart();

#pragma omp parallel
#pragma omp for
  for (int i = 0; i < current_traj_scans_in_lidar_.size(); i++) {
    beam::HighResolutionTimer timer2;
    timer2.restart();

    // setup registration (needed for parallelization)
    auto icp = std::make_unique<IcpType>();
    icp->setInputTarget(gt_cloud_refinement_in_world_);
    icp->setMaxCorrespondenceDistance(
        params_.icp_params_refinement.max_corr_dist);
    icp->setMaximumIterations(params_.icp_params_refinement.max_iterations);
    icp->setTransformationEpsilon(params_.icp_params_refinement.transform_eps);
    icp->setEuclideanFitnessEpsilon(params_.icp_params_refinement.fitness_eps);

    // get current scan
    auto it = current_traj_scans_in_lidar_.begin();
    std::advance(it, i);
    int64_t timestampInNs = it->first;
    const auto& scan_in_lidar = it->second;

    // get pose from spline first, if not then use raw ICP result
    Eigen::Matrix4d T_WORLDINIT_MOVING;
    if (!traj_spline.GetPose(timestampInNs, T_WORLDINIT_MOVING)) {
      traj_raw.GetPose(timestampInNs, T_WORLDINIT_MOVING);
    }

    // filter cloud
    PointCloudIRT cloud_filtered_in_lidar_frame =
        beam_filtering::FilterPointCloud<PointXYZIRT>(scan_in_lidar,
                                                      scan_filters_refinement_);

    // transform cloud
    PointCloudIRT::Ptr cloud_in_WorldEst = std::make_shared<PointCloudIRT>();
    Eigen::Matrix4d T_WORLDINIT_LIDAR = T_WORLDINIT_MOVING * T_MOVING_LIDAR_;
    pcl::transformPointCloud(cloud_filtered_in_lidar_frame, *cloud_in_WorldEst,
                             T_WORLDINIT_LIDAR);

    BEAM_INFO("Running registration refinement for scan {}/{}", i,
              current_traj_scans_in_lidar_.size());
    PointCloudIRT registered_cloud;
    icp->setInputSource(cloud_in_WorldEst);
    icp->align(registered_cloud);
    BEAM_INFO("Registration time for scan {}: {}s", i, timer2.elapsed());
    if (!icp->hasConverged()) { continue; }
    Eigen::Matrix4d T_WORLD_WORLDEST =
        icp->getFinalTransformation().cast<double>();
    Eigen::Matrix4d T_WORLD_MOVING = T_WORLD_WORLDEST * T_WORLDINIT_MOVING;
    traj_raw.UpdatePose(timestampInNs, T_WORLD_MOVING);
  }
  BEAM_INFO("Completed refinement in {}s", timer1.elapsed());
}

void ScanPoseGtGeneration::SaveMaps() {
  std::string map_filename1 =
      "map_raw_" + std::to_string(trajectories_raw_.size());
  SaveMap(*trajectories_raw_.rbegin(), map_filename1);
  trajectories_raw_.rbegin()->map_filename = map_filename1;
  SaveTrajectory(*trajectories_raw_.rbegin());

  std::string map_filename2 =
      "map_spline_" + std::to_string(trajectories_spline_.size());
  SaveMap(*trajectories_spline_.rbegin(), map_filename2);
  trajectories_spline_.rbegin()->map_filename = map_filename2;
  SaveTrajectory(*trajectories_spline_.rbegin());
  current_traj_scans_in_lidar_.clear();
}

void ScanPoseGtGeneration::SaveMap(const Trajectory& trajectory,
                                   const std::string& name) {
  if (!params_.save_map) { return; }

  PointCloudIRT map;
  for (const auto& [t, p] : trajectory.GetPoses()) {
    const PointCloudIRT& cloud_in_lidar =
        current_traj_scans_in_lidar_.at(p.timestampInNs);
    PointCloudIRT cloud_filtered_in_lidar_frame =
        beam_filtering::FilterPointCloud<PointXYZIRT>(cloud_in_lidar,
                                                      output_filters_);
    PointCloudIRT cloud_in_world;
    Eigen::Matrix4d T_WORLD_LIDAR = p.T_FIXED_MOVING * T_MOVING_LIDAR_;
    pcl::transformPointCloud(cloud_filtered_in_lidar_frame, cloud_in_world,
                             T_WORLD_LIDAR);
    map += cloud_in_world;
  }

  std::string err;
  std::string map_filepath = beam::CombinePaths(map_save_dir_, name + ".pcd");
  BEAM_INFO("saving map of size {} to: {}", map.size(), map_filepath);
  if (!beam::SavePointCloud<PointXYZIRT>(
          map_filepath, map, beam::PointCloudFileType::PCDBINARY, err)) {
    BEAM_ERROR("unable to save map, reason: {}", err);
  }
}

void ScanPoseGtGeneration::SaveTrajectory(const Trajectory& trajectory) {
  if (trajectory.Size() == 0) { return; }

  beam_mapping::Poses poses_current;
  for (const auto& [t, p] : trajectory.GetPoses()) {
    ros::Time time_ros;
    time_ros.fromNSec(p.timestampInNs);
    poses_current.AddSingleTimeStamp(time_ros);
    poses_current.AddSinglePose(p.T_FIXED_MOVING);
  }
  poses_current.SetFixedFrame(world_frame_id_);
  poses_current.SetMovingFrame(moving_frame_id_);
  std::string poses_name = beam::CombinePaths(
      poses_save_dir_, trajectory.map_filename + "_poses.json");
  poses_current.WriteToFile(poses_name, "JSON");
  poses_name = beam::CombinePaths(poses_save_dir_,
                                  trajectory.map_filename + "_poses.pcd");
  BEAM_INFO("saving {} poses for trajectory to: {}",
            poses_current.GetTimeStamps().size(), poses_name);
  poses_current.WriteToFile(poses_name, "PCD");
}

void ScanPoseGtGeneration::SaveTrajectories(
    const std::vector<Trajectory>& trajectory, const std::string& name) {
  // save combined
  beam_mapping::Poses poses_all;
  for (const Trajectory& t : trajectory) {
    if (t.Size() == 0) { continue; }
    beam_mapping::Poses poses_current;
    for (const auto& [t, p] : t.GetPoses()) {
      ros::Time time_ros;
      time_ros.fromNSec(p.timestampInNs);
      poses_all.AddSingleTimeStamp(time_ros);
      poses_all.AddSinglePose(p.T_FIXED_MOVING);
    }
  }
  poses_all.SetFixedFrame(world_frame_id_);
  poses_all.SetMovingFrame(moving_frame_id_);
  std::string combined_name =
      beam::CombinePaths(root_save_dir_, name + "_poses_combined");
  poses_all.WriteToFile(combined_name + ".json", "JSON");
  poses_all.WriteToFile(combined_name + ".pcd", "PCD");
}

void ScanPoseGtGeneration::SaveResults() {
  SaveTrajectories(trajectories_raw_, "trajectory_raw");
  SaveTrajectories(trajectories_spline_, "trajectory_spline");

  // save list of trajectories & maps
  nlohmann::json J;
  std::vector<std::string> raw_names;
  for (const auto& t : trajectories_raw_) {
    raw_names.push_back(t.map_filename);
  }
  J["raw"] = raw_names;
  std::vector<std::string> spline_names;
  for (const auto& t : trajectories_spline_) {
    spline_names.push_back(t.map_filename);
  }
  J["spline"] = spline_names;
  std::string filename = beam::CombinePaths(root_save_dir_, "output_list.json");
  std::ofstream o(filename);
  o << std::setw(4) << J << std::endl;

  // copy over files to output
  std::string output_config =
      beam::CombinePaths(root_save_dir_, "config_copy.json");
  BEAM_INFO("copying config file to: {}", output_config);
  boost::filesystem::copy_file(
      inputs_.config, output_config,
      boost::filesystem::copy_option::overwrite_if_exists);
  std::string output_gt_cloud =
      beam::CombinePaths(root_save_dir_, "gt_cloud_copy.pcd");
  BEAM_INFO("copying gt cloud to: {}", output_gt_cloud);
  boost::filesystem::copy_file(
      inputs_.gt_cloud, output_gt_cloud,
      boost::filesystem::copy_option::overwrite_if_exists);
  std::string output_gt_cloud_pose =
      beam::CombinePaths(root_save_dir_, "gt_cloud_pose_copy.json");
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