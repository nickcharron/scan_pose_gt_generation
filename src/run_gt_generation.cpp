#include <gflags/gflags.h>

#include <scan_pose_gt_generation/ScanPoseGtGeneration.h>

#include <beam_utils/gflags.h>

DEFINE_string(bag_file, "",
              "Full file path to bag file containing the lidar data. ");
DEFINE_validator(bag_file, &beam::gflags::ValidateBagFileMustExist);
DEFINE_string(gt_cloud, "",
              "Full file path to ground truth cloud (pcd) used to register "
              "scans against. ");
DEFINE_validator(gt_cloud, &beam::gflags::ValidateFileMustExist);
DEFINE_string(
    gt_cloud_pose, "",
    "Full file path to the ground truth cloud pose (json). This transform from "
    "the GT cloud's frame to the world frame (T_World_GtCloud), or the fixed "
    "frame in the initial poses. This can be generated by aligning the GT "
    "Cloud to a map generated with the intial estimated poses. This can be "
    "done easily in CloudCompare. See example file: "
    "T_World_GtCloud_example.json");
DEFINE_validator(gt_cloud_pose, &beam::gflags::ValidateJsonFileMustExist);
DEFINE_string(initial_trajectory, "",
              "full path to initial trajectory pose file. For format, see "
              "config/initial_trajectory_example2. You can create "
              "this using the bag_to_poses executable, or running beam_slam");
DEFINE_validator(initial_trajectory, &beam::gflags::ValidateFileMustExist);
DEFINE_string(output_directory, "",
              "Full path to output directory to save results. This directory "
              "must exist.");
DEFINE_validator(output_directory, &beam::gflags::ValidateDirMustExist);
DEFINE_string(extrinsics, "",
              "Full file path to extrinsics json config file. For format, see "
              "map_builder/config/examples/EXAMPLE_EXTRINSICS.json");
DEFINE_validator(extrinsics, &beam::gflags::ValidateJsonFileMustExist);
DEFINE_string(topic, "",
              "topic name for the lidar data you want to generate GT for");
DEFINE_validator(topic, &beam::gflags::ValidateCannotBeEmpty);
DEFINE_string(config, "",
              "full path to config file, example file found in config/config_example.json");
DEFINE_validator(config, &beam::gflags::ValidateFileMustExist);

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  scan_pose_gt_gen::ScanPoseGtGeneration::Inputs inputs;
  inputs.bag = FLAGS_bag_file;
  inputs.gt_cloud = FLAGS_gt_cloud;
  inputs.gt_cloud_pose = FLAGS_gt_cloud_pose;
  inputs.initial_trajectory = FLAGS_initial_trajectory;
  inputs.output_directory = FLAGS_output_directory;
  inputs.extrinsics = FLAGS_extrinsics;
  inputs.topic = FLAGS_topic;
  inputs.config = FLAGS_config;
  scan_pose_gt_gen::ScanPoseGtGeneration gt_generator(inputs);
  gt_generator.run();

  return 0;
}
