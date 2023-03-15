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
    int point_size{3};
    bool extract_loam_points{true};
    double max_spline_length_m{5};
    double min_spline_measurements{10};
    IcpParams icp_params;
  };

  class Trajectory {
  public:
    Trajectory() = default;

    std::string map_filename;

    const std::vector<beam::Pose>& GetPoses() const { return poses_; }

    int Size() const { return poses_.size(); }

    double Length() const { return length_; }

    void AddPose(int64_t timestampInNs, const Eigen::Matrix4d& T_FIXED_MOVING) {
      beam::Pose pose;
      pose.timestampInNs = timestampInNs;
      pose.T_FIXED_MOVING = T_FIXED_MOVING;

      if (poses_.size() > 1) {
        Eigen::Vector3d t_last =
            poses_.rbegin()->T_FIXED_MOVING.block(0, 3, 3, 1);
        Eigen::Vector3d t_cur = T_FIXED_MOVING.block(0, 3, 3, 1);
        length_ += (t_last - t_cur).norm();
      }

      poses_.push_back(pose);
    }

  private:
    std::vector<beam::Pose> poses_;
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
  void SetupRegistration();
  void RegisterScans();
  void RegisterSingleScan(const PointCloudIRT& cloud_in_lidar_frame,
                          const ros::Time& timestamp);
  PointCloudIRT ExtractStrongLoamPoints(const PointCloudIRT& cloud_in);
  void UpdateT_INIT_SPLINE();
  bool GetT_WORLDEST_MOVING(const ros::Time& timestamp,
                            Eigen::Matrix4d& T_WORLD_MOVING);
  bool GetT_WORLDESTINIT_MOVING(const ros::Time& timestamp,
                                Eigen::Matrix4d& T_WORLD_MOVING);
  void AddRegistrationResult(const PointCloudIRT& cloud_in_lidar_frame,
                             const Eigen::Matrix4d& T_WORLD_LIDAR,
                             const ros::Time& timestamp);
  bool IsMapFull();
  void FitSplineToTrajectory();
  void SaveMaps();
  void SaveMap(const Trajectory& trajectory, const std::string& name);
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
  beam_calibration::TfTree input_trajectory_;
  PointCloudIRT::Ptr gt_cloud_in_world_;
  Eigen::Matrix4d T_MOVING_LIDAR_;
  Eigen::Matrix4d T_INIT_SPLINE_;
  std::string world_frame_id_;
  std::string moving_frame_id_;
  std::string lidar_frame_id_;

  std::vector<beam_filtering::FilterParamsType> scan_filters_;
  std::vector<beam_filtering::FilterParamsType> gt_cloud_filters_;

  std::string map_save_dir_;
  beam::HighResolutionTimer timer_;
  int scan_counter_{0};
  int total_scans_;
  std::unique_ptr<pcl::visualization::PCLVisualizer> viewer_;
  bool next_scan_{false};

  // registration
  using IcpType = pcl::IterativeClosestPoint<PointXYZIRT, PointXYZIRT>;
  std::unique_ptr<IcpType> icp_;
};

} // namespace scan_pose_gt_gen