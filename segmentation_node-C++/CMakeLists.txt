cmake_minimum_required(VERSION 2.8.3)
project(segmentation)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  std_srvs
  cv_bridge
  image_transport
  image_geometry
  

)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OPENNI_INCLUDE_DIRS}
  
)

add_executable(segmentation_node
  src/segmentation_node.cpp
  include/segmentation/segmentation_node.h
)



target_link_libraries(segmentation_node
  ${catkin_LIBRARIES}
  ${PCL_LIBARIES}
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
