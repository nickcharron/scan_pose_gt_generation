#pragma once

#define PCL_NO_PRECOMPILE

#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <beam_calibration/TfTree.h>
#include <beam_filtering/Utils.h>
#include <beam_mapping/Poses.h>
#include <beam_mapping/Utils.h>
#include <beam_utils/pointclouds.h>
#include <beam_utils/se3.h>

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
    bool visualize;
  };

  struct IcpParams {
    double max_corr_dist{0.4};
    int max_iterations{40};
    double transform_eps{1e-8};
    double fitness_eps{1e-2};
  };

  struct Params {
    bool save_map{true};
    int map_max_size{100};
    int point_size{3};
    bool extract_loam_points{true};
    IcpParams icp_params;
  };

  struct Trajectory {
    std::string map_filename;
    std::vector<beam::Pose> poses;
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
  void SetupRegistration();
  void RegisterScans();
  void RegisterSingleScan(const PointCloudIRT& cloud_in_lidar_frame,
                          const ros::Time& timestamp);
  bool GetT_WorldEst_Lidar(const ros::Time& timestamp,
                           Eigen::Matrix4d& T_WORLD_LIDAR);
  bool GetT_MovingLast_MovingCurrent(
      const ros::Time& timestamp_current,
      Eigen::Matrix4d& T_MovingLast_MovingCurrent);
  void SaveSuccessfulRegistration(const PointCloudIRT& cloud_in_lidar_frame,
                                  const Eigen::Matrix4d& T_WORLD_LIDAR,
                                  const ros::Time& timestamp);
  void SaveResults();
  PointCloudIRT ExtractStrongLoamPoints(const PointCloudIRT& cloud_in);
  void DisplayResults(const PointCloudIRT& cloud_in_lidar,
                      const Eigen::Matrix4d& T_WorldOpt_Lidar,
                      const Eigen::Matrix4d& T_WorldEst_Lidar, bool successful);

  void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event);

  Inputs inputs_;
  Params params_;
  std::vector<Trajectory> trajectories_;
  beam_calibration::TfTree input_trajectory_;
  PointCloudIRT::Ptr gt_cloud_in_world_;
  Eigen::Matrix4d T_World_GtCloud_;
  Eigen::Matrix4d T_MOVING_LIDAR_;
  std::string world_frame_id_;
  std::string moving_frame_id_;
  std::string lidar_frame_id_;

  std::vector<beam_filtering::FilterParamsType> scan_filters_;
  std::vector<beam_filtering::FilterParamsType> gt_cloud_filters_;

  PointCloudIRT map_;
  int current_map_size_{0};
  std::string map_save_dir_;
  ros::Time timestamp_last_;
  Eigen::Matrix4d T_World_MovingLast_;
  beam::HighResolutionTimer timer_;
  int scan_counter_{0};
  std::unique_ptr<pcl::visualization::PCLVisualizer> viewer_;
  bool next_scan_{false};
  bool is_first_scan_{true};

  // registration
  using IcpType = pcl::IterativeClosestPoint<PointXYZIRT, PointXYZIRT>;
  std::unique_ptr<IcpType> icp_;
};

} // namespace scan_pose_gt_gen