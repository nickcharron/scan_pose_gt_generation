#include <scan_pose_gt_generation/ScanPoseGtGeneration.h>

#include <fstream>

#include <boost/filesystem.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <beam_utils/log.h>
#include <beam_utils/math.h>

namespace scan_pose_gt_gen {

void ScanPoseGtGeneration::run() {
  LoadConfig();
  LoadExtrinsics();
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

  if(params_.save_map){
      map_save_dir_ = beam::CombinePaths(inputs_.output, "gt_maps");
      boost::filesystem::create_directory(map_save_dir_);
  }

  scan_filters_ = beam_filtering::LoadFilterParamsVector(J["scan_filters"]);
  gt_cloud_filters_ =
      beam_filtering::LoadFilterParamsVector(J["gt_cloud_filters"]);
}

void ScanPoseGtGeneration::LoadExtrinsics() {
  extrinsics_.LoadJSON(inputs_.extrinsics);
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

void ScanPoseGtGeneration::RegisterScans() {
  rosbag::Bag bag;
  BEAM_INFO("Opening bag: {}", inputs_.bag);
  bag.open(inputs_.bag, rosbag::bagmode::Read);
  if (!bag.isOpen()) { throw std::runtime_error{"unable to open ROS bag"}; }

  rosbag::View view(bag, rosbag::TopicQuery(inputs_.topic), ros::TIME_MIN,
                    ros::TIME_MAX, true);
  int total_messages = view.size();
  BEAM_INFO("Read a total of {} pointcloud messages", total_messages);
  int message_counter{0};
  std::string output_message = "registering scans";
  for (auto iter = view.begin(); iter != view.end(); iter++) {
    message_counter++;
    beam::OutputPercentComplete(message_counter, total_messages,
                                output_message);
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
  using namespace beam_matching;

  // filter cloud and transform to estimated world frame
  PointCloud cloud_filtered_in_lidar_frame =
      beam_filtering::FilterPointCloud<pcl::PointXYZ>(cloud_in_lidar_frame,
                                                      scan_filters_);
  PointCloudPtr cloud_in_WorldEst = std::make_shared<PointCloud>();
  Eigen::Matrix4d T_WorldEst_Lidar = GetT_WorldEst_Lidar(timestamp);
  pcl::transformPointCloud(cloud_filtered, *cloud_in_WorldEst,
                           T_WorldEst_Lidar);

  // run ICP
  IcpMatcherParams icp_params;
  icp_params.max_corr = 0.1;
  icp_params.max_iter = 0.1;
  icp_params.fit_eps = 1e-3;

  IcpMatcher icp_matcher(icp_params);
  icp_matcher.SetRef(gt_cloud_in_world_);
  icp_matcher.SetTarget(cloud_in_WorldEst);
  icp_matcher.Match();
  Eigen::Matrix4d T_WorldEst_World = icp_matcher.GetResult();

  double trans = T_WorldEst_World.block(0, 3, 3, 1).norm();
  if (trans > params_.translation_threshold_m) {
    invalid_scans.push_back(timestamp.toNSec());
    invalid_scan_translations.push_back(trans);
    invalid_scan_angles.push_back(0);
    return;
  }

  Eigen::Matrix3d R = T_WorldEst_World.block(0, 0, 3, 3);
  Eigen::AxisAngled aa(R);
  double angle = aa.angle() * 180 / M_PI;
  if (angle > params_.rotation_threshold_deg) {
    invalid_scans.push_back(timestamp.toNSec());
    invalid_scan_translations.push_back(0);
    invalid_scan_angles.push_back(angle);
    return;
  }

  Eigen::Matrix4d T_WORLD_LIDAR =
      beam::InvertTransform(T_WorldEst_World) * T_WorldEst_Lidar;
  SaveSuccessfulRegistration(cloud_in_lidar_frame, T_WORLD_LIDAR, timestamp);
}

Eigen::Matrix4d
    ScanPoseGtGeneration::GetT_WorldEst_Lidar(const ros::Time& timestamp) {
  // todo
}

void ScanPoseGtGeneration::SaveSuccessfulRegistration(
    const PointCloud& cloud_in_lidar_frame,
    const Eigen::Matrix4d& T_WORLD_LIDAR, const ros::TIme& timestamp) {
  PointCloud cloud_in_world;
  pcl::transformPointCloud(cloud_in_lidar_frame, cloud_in_world, T_WORLD_LIDAR);
  map_ += cloud_in_world;
  current_map_size_++;

  if (current_map_size_ == params_.map_max_size) {
    std::string err;
    std::string filename = "map_" + std::to_string(results_.saved_cloud_names.size() + 1) + ".pcd";
    std::string filepath = beam::CombinePaths(map_save_dir_, filename);
    if (beam::SavePointCloud<pcl::PointXYZ>(
            filename, map_, beam::PointCloudFileType::PCDBINARY, err)) {
      BEAM_CRITICAL("unable to save map, reason: {}", err);
      throw std::runtime_error{"unable to save map"};
    }
    map_ = PointCloud();
    current_map_size_ = 0;
    saved_cloud_names.push_back(filename);
  }
}

void ScanPoseGtGeneration::SaveResults() {
  // todo
  // save map_names.json
  // save poses.json
}

} // namespace scan_pose_gt_gen