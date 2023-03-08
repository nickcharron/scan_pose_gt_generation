#include <gflags/gflags.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <beam_filtering/Utils.h>
#include <beam_utils/gflags.h>
#include <beam_utils/kdtree.h>
#include <beam_utils/pointclouds.h>

DEFINE_string(input_directory, "",
              "Full path to output directory of run_gt_generation");
DEFINE_validator(input_directory, &beam::gflags::ValidateDirMustExist);

class ResultsVisualizer {
public:
  struct ResultsData {
    PointCloudPtr gt_cloud_in_world{std::make_shared<PointCloud>()};
    std::vector<std::string> maps;
  };

  explicit ResultsVisualizer(const std::string& input_dir) {
    LoadResultsData(input_dir);
    Setup();
  }

  void Run() {
    for (const auto& map_path : data_.maps) {
      PointCloud map;
      BEAM_INFO("loading map: {}", map_path);
      pcl::io::loadPCDFile(map_path, map);
      if (map.empty()) {
        BEAM_ERROR("empty input map.");
        throw std::runtime_error{"empty input map"};
      }

      viewer_.removePointCloud("CurrentMap");
      PointCloudPtr map_filtered = std::make_shared<PointCloud>();
      *map_filtered = FilterPointsCloseToGt(map);
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col(
          map_filtered, 0, 255, 0);
      viewer_.addPointCloud<pcl::PointXYZ>(map_filtered, col, "CurrentMap");
      viewer_.setPointCloudRenderingProperties(
          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "CurrentMap");
      next_scan_ = false;
      while (!viewer_.wasStopped() && !next_scan_) { viewer_.spinOnce(); }
    }
  }

private:
  void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
    if (event.getKeySym() == "n" && event.keyDown()) {
      BEAM_INFO("displaying next scan...");
      next_scan_ = true;
    }
  }

  PointCloud FilterPointsCloseToGt(const PointCloud& map) {
    PointCloud map_out;
    for (int i = 0; i < map.size(); i++) {
      std::vector<uint32_t> point_ids;
      std::vector<float> point_distances;
      kdtree_->nearestKSearch(map.at(i), 1, point_ids, point_distances);
      if (!point_distances.empty() &&
          point_distances.at(0) < max_distance_to_gt_) {
        map_out.push_back(map.at(i));
      }
    }
    return map_out;
  }

  void Setup() {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> col(
        data_.gt_cloud_in_world, 255, 255, 255);
    viewer_.addPointCloud<pcl::PointXYZ>(data_.gt_cloud_in_world, col, "GTMap");
    viewer_.setPointCloudRenderingProperties(
          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "GTMap");
    viewer_.addCoordinateSystem(1.0);
    std::function<void(const pcl::visualization::KeyboardEvent&)> keyboard_cb =
        [this](const pcl::visualization::KeyboardEvent& event) {
          keyboardEventOccurred(event);
        };
    viewer_.registerKeyboardCallback(keyboard_cb);
    kdtree_ =
        std::make_unique<beam::KdTree<pcl::PointXYZ>>(*data_.gt_cloud_in_world);
  }

  void LoadResultsData(const std::string& dir) {
    std::string gt_cloud_p = beam::CombinePaths(dir, "gt_cloud_copy.pcd");
    BEAM_INFO("Loading gt cloud: {}", gt_cloud_p);
    PointCloud gt_cloud_in;
    pcl::io::loadPCDFile(gt_cloud_p, gt_cloud_in);
    if (gt_cloud_in.empty()) {
      BEAM_ERROR("empty input gt cloud.");
      throw std::runtime_error{"empty input cloud"};
    }

    std::string gt_cloud_pose =
        beam::CombinePaths(dir, "gt_cloud_pose_copy.json");
    BEAM_INFO("Loading gt cloud pose: {}", gt_cloud_pose);
    nlohmann::json J;
    if (!beam::ReadJson(gt_cloud_pose, J)) {
      throw std::runtime_error{"Invalid gt_cloud_pose json"};
    }

    if (!J.contains("T_World_GtCloud")) {
      throw std::runtime_error{
          "cannot load gt cloud pose, missing T_World_GtCloud field"};
    }

    std::vector<double> v = J["T_World_GtCloud"];
    if (v.size() != 16) {
      throw std::runtime_error{
          "invalid T_World_GtCloud, size must be 16 (4x4)"};
    }

    Eigen::Matrix4d T_World_GtCloud;
    T_World_GtCloud << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
        v[9], v[10], v[11], v[12], v[13], v[14], v[15];

    // filter input cloud
    beam_filtering::FilterParamsType filter{
        beam_filtering::FilterType::VOXEL,
        std::vector<double>{0.05, 0.05, 0.05}};
    std::vector<beam_filtering::FilterParamsType> filters{filter};
    PointCloud cloud_filtered =
        beam_filtering::FilterPointCloud<pcl::PointXYZ>(gt_cloud_in, filters);

    pcl::transformPointCloud(cloud_filtered, *data_.gt_cloud_in_world,
                             T_World_GtCloud);

    // load results
    nlohmann::json J2;
    std::string results_file = beam::CombinePaths(dir, "results_summary.json");
    BEAM_INFO("Reading results json: {}", results_file);
    if (!beam::ReadJson(results_file, J2)) {
      throw std::runtime_error{"Invalid gt_cloud_pose json"};
    }
    if (!J2.contains("map_names")) {
      throw std::runtime_error{
          "invalid results file, no field named map_names"};
    }
    std::vector<std::string> map_names = J2["map_names"];
    std::string map_dir = beam::CombinePaths(dir, "gt_maps");
    for (const auto& name : map_names) {
      data_.maps.push_back(beam::CombinePaths(map_dir, name));
    }
  }

  ResultsData data_;
  bool next_scan_{false};
  double max_distance_to_gt_{2};
  std::unique_ptr<beam::KdTree<pcl::PointXYZ>> kdtree_;
  pcl::visualization::PCLVisualizer viewer_;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ResultsVisualizer viz(FLAGS_input_directory);
  viz.Run();
  return 0;
}
