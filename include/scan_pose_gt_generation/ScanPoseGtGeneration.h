#pragma once

#include <beam_calibration/TfTree.h>
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
  };

  struct Params {
    bool save_map{true};
    int map_size{100};
  };

  ScanPoseGtGeneration() = delete;

  ScanPoseGtGeneration(const Inputs& inputs) : inputs_(inputs) {}

  ScanPoseGtGeneration(const Inputs& inputs, const Params& params)
      : inputs_(inputs), params_(params) {}

  void run();

private:
  void LoadExtrinsics();
  void LoadGtCloud();
  void LoadTrajectory();
  void RegisterScans();
  void LoadTrajectory();

  Inputs inputs_;
  Params params_;
  beam_calibration::TfTree trajectory_;
  beam_calibration::TfTree extrinsics_;
  PointCloud gt_cloud_in_world;
  Eigen::Matrix4d T_World_GtCloud_;
  std::string world_frame_id_;
  std::string moving_frame_id_;
  std::string lidar_frame_id_;
};

} // namespace scan_pose_gt_gen