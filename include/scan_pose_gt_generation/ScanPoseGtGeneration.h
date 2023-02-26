#pragma once

#include <beam_calibration/TfTree.h>
#include <beam_filtering/Utils.h>
#include <beam_mapping/Poses.h>
#include <beam_mapping/Utils.h>
#include <beam_utils/pointclouds.h>

namespace scan_pose_gt_gen {

class ScanPoseGtGeneration {
public:
  struct Inputs {
    std::string bag;
    std::string gt_cloud;
    std::string gt_cloud_pose;
    std::string initial_trajectory;
    std::string output_directory;
    std::string extrinsics;
    std::string topic;
    std::string config;
  };

  struct Params {
    bool save_map{true};
    int map_max_size{100};
  };

  ScanPoseGtGeneration() = delete;

  ScanPoseGtGeneration(const Inputs& inputs) : inputs_(inputs) {}

  ScanPoseGtGeneration(const Inputs& inputs, const Params& params)
      : inputs_(inputs), params_(params) {}

  void run();

private:
  void LoadConfig();
  void LoadExtrinsics();
  void LoadGtCloud();
  void LoadTrajectory();
  void SetInputFilters();
  void RegisterScans();
  void RegisterSingleScan(const PointCloud& cloud, const ros::Time& timestamp);

  Inputs inputs_;
  Params params_;
  beam_calibration::TfTree trajectory_;
  beam_calibration::TfTree extrinsics_;
  PointCloud gt_cloud_in_world_;
  Eigen::Matrix4d T_World_GtCloud_;
  std::string world_frame_id_;
  std::string moving_frame_id_;
  std::string lidar_frame_id_;

  std::vector<beam_filtering::FilterParamsType> scan_filters_;
  std::vector<beam_filtering::FilterParamsType> gt_cloud_filters_;
};

} // namespace scan_pose_gt_gen