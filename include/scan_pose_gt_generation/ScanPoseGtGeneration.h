#pragma once

#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

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
    bool visualize;
  };

  struct IcpParams {
    double max_corr_dist{0.1};
    int max_iterations{40};
    double transform_eps{1e-8};
    double fitness_eps{1e-2};
  };

  struct Params {
    bool save_map{true};
    int map_max_size{100};
    double rotation_threshold_deg{15};
    double translation_threshold_m{0.5};
    int point_size{3};
    IcpParams icp_params;
  };

  struct Results {
    std::vector<int64_t> invalid_scan_stamps; // timestamps in NS
    std::vector<double> invalid_scan_angles;
    std::vector<double> invalid_scan_translations;
    std::vector<int64_t> valid_scan_stamps; // timestamps in NS
    std::vector<std::string> saved_cloud_names;
    beam_mapping::Poses poses;
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
  void RegisterSingleScan(const PointCloud& cloud_in_lidar_frame,
                          const ros::Time& timestamp);
  Eigen::Matrix4d GetT_WorldEst_Lidar(const ros::Time& timestamp);
  Eigen::Matrix4d
      GetT_MovingLast_MovingCurrent(const ros::Time& timestamp_current);
  void SaveSuccessfulRegistration(const PointCloud& cloud_in_lidar_frame,
                                  const Eigen::Matrix4d& T_WORLD_LIDAR,
                                  const ros::Time& timestamp);
  void SaveResults();

  void DisplayResults(const PointCloud& cloud_in_lidar,
                      const Eigen::Matrix4d& T_WorldOpt_Lidar,
                      const Eigen::Matrix4d& T_WorldEst_Lidar, bool successful,
                      const std::string& icp_results = "");

  void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event);

  Inputs inputs_;
  Params params_;
  Results results_;
  beam_calibration::TfTree trajectory_;
  PointCloudPtr gt_cloud_in_world_;
  Eigen::Matrix4d T_World_GtCloud_;
  Eigen::Matrix4d T_MOVING_LIDAR_;
  std::string world_frame_id_;
  std::string moving_frame_id_;
  std::string lidar_frame_id_;

  std::vector<beam_filtering::FilterParamsType> scan_filters_;
  std::vector<beam_filtering::FilterParamsType> gt_cloud_filters_;

  PointCloud map_;
  int current_map_size_{0};
  std::string map_save_dir_;
  ros::Time timestamp_last_;
  Eigen::Matrix4d T_World_MovingLast_;
  beam::HighResolutionTimer timer_;
  int scan_counter_{0};
  std::unique_ptr<pcl::visualization::PCLVisualizer> viewer_;
  bool next_scan_{false};

  // registration
  using IcpType = pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>;
  std::unique_ptr<IcpType> icp_;
};

} // namespace scan_pose_gt_gen