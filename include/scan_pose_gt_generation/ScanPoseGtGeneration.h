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
    int start_offset_s{0};
    int end_offset_s{0};
    bool visualize;
  };

  struct IcpParamsInit {
    double max_corr_dist{0.4};
    int max_iterations{40};
    double transform_eps{1e-8};
    double fitness_eps{1e-2};
  };

  struct IcpParamsRef {
    double max_corr_dist{0.05};
    int max_iterations{20};
    double transform_eps{1e-9};
    double fitness_eps{1e-3};
  };

  struct Params {
    bool save_map{true};
    int point_size{3};
    bool extract_loam_points{true};
    double max_spline_length_m{5};
    double min_spline_measurements{10};
    bool run_refinement{true};
    IcpParamsInit icp_params;
    IcpParamsRef icp_params_refinement;
  };

  class Trajectory {
  public:
    Trajectory() = default;

    std::string map_filename;

    const std::map<int64_t, beam::Pose>& GetPoses() const { return poses_; }

    int Size() const { return poses_.size(); }

    double Length() const { return length_; }

    bool GetPose(int64_t timestampInNs, Eigen::Matrix4d& T_FIXED_MOVING) const {
      auto iter = poses_.find(timestampInNs);
      if (iter == poses_.end()) { return false; }
      T_FIXED_MOVING = iter->second.T_FIXED_MOVING;
      return true;
    }

    bool UpdatePose(int64_t timestampInNs, Eigen::Matrix4d& T_FIXED_MOVING) {
      auto iter = poses_.find(timestampInNs);
      if (iter == poses_.end()) { return false; }
      iter->second.T_FIXED_MOVING = T_FIXED_MOVING;
      return true;
    }

    void RemovePose(int64_t timestampInNs) { poses_.erase(timestampInNs); }

    void AddPose(int64_t timestampInNs, const Eigen::Matrix4d& T_FIXED_MOVING) {
      beam::Pose pose;
      pose.timestampInNs = timestampInNs;
      pose.T_FIXED_MOVING = T_FIXED_MOVING;

      if (poses_.size() > 1) {
        Eigen::Vector3d t_last =
            poses_.rbegin()->second.T_FIXED_MOVING.block(0, 3, 3, 1);
        Eigen::Vector3d t_cur = T_FIXED_MOVING.block(0, 3, 3, 1);
        length_ += (t_last - t_cur).norm();
      }

      poses_.emplace(timestampInNs, pose);
    }

  private:
    std::map<int64_t, beam::Pose> poses_;
    double length_{0};
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
  void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event);
  void LoadTrajectory();
  void RegisterScans();
  void ProcessSingleScan(const PointCloudIRT& cloud_in_lidar_frame,
                         const ros::Time& timestamp);
  void RegisterCurrentTrajectoryScansInParallel();
  void RegisterCurrentTrajectoryScans();
  PointCloudIRT ExtractStrongLoamPoints(const PointCloudIRT& cloud_in);
  bool GetT_WORLD_MOVING(const ros::Time& timestamp,
                         Eigen::Matrix4d& T_WORLD_MOVING);
  bool GetT_WORLD_MOVING_INIT(const ros::Time& timestamp,
                              Eigen::Matrix4d& T_WORLD_MOVING);
  bool IsMapFull();
  void FitSplineToTrajectory();
  void RunRefinement();
  void SaveMaps();
  void SaveMap(const Trajectory& trajectory, const std::string& name);
  void SaveTrajectory(const Trajectory& trajectory);
  void SaveTrajectories(const std::vector<Trajectory>& trajectory,
                        const std::string& name);
  void SaveResults();
  void DisplayResults(const PointCloudIRT& cloud_in_lidar,
                      const Eigen::Matrix4d& T_WORLDOPT_LIDAR,
                      const Eigen::Matrix4d& T_WORLDEST_LIDAR, bool successful);

  Inputs inputs_;
  Params params_;
  std::unordered_map<int64_t, PointCloudIRT> current_traj_scans_in_lidar_;
  std::vector<Trajectory> trajectories_raw_;
  std::vector<Trajectory> trajectories_spline_;
  Eigen::Matrix4d T_WORLD_WORLDINIT_AVG_{Eigen::Matrix4d::Identity()};
  beam_calibration::TfTree input_trajectory_;
  PointCloudIRT::Ptr gt_cloud_in_world_;
  PointCloudIRT::Ptr gt_cloud_refinement_in_world_;
  Eigen::Matrix4d T_MOVING_LIDAR_;
  std::string world_frame_id_;
  std::string moving_frame_id_;
  std::string lidar_frame_id_;
  int64_t start_time_ns;
  int64_t end_time_ns;

  std::vector<beam_filtering::FilterParamsType> scan_filters_;
  std::vector<beam_filtering::FilterParamsType> scan_filters_refinement_;
  std::vector<beam_filtering::FilterParamsType> gt_cloud_filters_;
  std::vector<beam_filtering::FilterParamsType> gt_cloud_filters_refinement_;
  std::vector<beam_filtering::FilterParamsType> output_filters_;

  std::string map_save_dir_;
  std::string poses_save_dir_;
  std::string root_save_dir_;
  // std::vector<std::string> trajectory_names_;
  // std::vector<std::string> trajectory_names_raw_;
  // std::vector<std::string> trajectory_names_spline_;
  // std::vector<std::string> map_names_raw_;
  // std::vector<std::string> map_names_spline_;

  beam::HighResolutionTimer timer_;
  int scan_counter_{0};
  int total_scans_;
  std::unique_ptr<pcl::visualization::PCLVisualizer> viewer_;
  bool next_scan_{false};

  // registration
  using IcpType = pcl::IterativeClosestPoint<PointXYZIRT, PointXYZIRT>;
};

} // namespace scan_pose_gt_gen