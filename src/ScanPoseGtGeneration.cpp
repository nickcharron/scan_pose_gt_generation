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
  LoadExtrinsics();
  LoadGtCloud();
  LoadTrajectory();
  RegisterScans();
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

  pcl::transformPointCloud(gt_cloud_in, gt_cloud_in_world, T_World_GtCloud);
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

void ScanPoseGtGeneration::RegisterSingleScan(const PointCloud& cloud,
                                              const ros::Time& timestamp) {
  //
}

} // namespace scan_pose_gt_gen