cmake_minimum_required(VERSION 3.14)
project(scan_pose_gt_generation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(
  beam REQUIRED
  COMPONENTS
  utils
  mapping
  calibration
)

find_package(OpenMP)


## Declare a C++ library
add_library(
  ${PROJECT_NAME}
  src/ScanPoseGtGeneration.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    include
)
## Specify libraries to link a library or executable target against
target_link_libraries(
  ${PROJECT_NAME}
    beam::utils
    beam::matching
    beam::filtering
    beam::mapping
    beam::calibration
    OpenMP::OpenMP_CXX
)

################ Executables ###################
add_executable(${PROJECT_NAME}_run_gt_generation
  src/run_gt_generation.cpp
)

target_include_directories(${PROJECT_NAME}_run_gt_generation
  PUBLIC
)

target_link_libraries(${PROJECT_NAME}_run_gt_generation
  ${PROJECT_NAME}
  beam::utils
)

add_executable(${PROJECT_NAME}_results_visualizer
  src/results_visualizer.cpp
)

target_include_directories(${PROJECT_NAME}_results_visualizer
  PUBLIC
)

target_link_libraries(${PROJECT_NAME}_results_visualizer
  ${PROJECT_NAME}
  beam::utils
)
