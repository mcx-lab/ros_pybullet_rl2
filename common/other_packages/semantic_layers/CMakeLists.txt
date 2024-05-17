cmake_minimum_required(VERSION 2.8.3)
project(semantic_layers)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  std_msgs
  costmap_2d
  dynamic_reconfigure
  roscpp
  tf2_ros
)

add_message_files(
   DIRECTORY msg
   FILES
    People.msg
    Person.msg
)

generate_messages(
   DEPENDENCIES geometry_msgs std_msgs
)

generate_dynamic_reconfigure_options(
  cfg/SemanticLayerPlugin.cfg
  cfg/ProxemicLayer.cfg
)


catkin_package(
    CATKIN_DEPENDS 
      angles 
      costmap_2d 
      dynamic_reconfigure
      geometry_msgs
      message_runtime
      std_msgs
      roscpp
      people_msgs
      pluginlib
      tf2_ros
)

include_directories(${catkin_INCLUDE_DIRS} include)

add_library(semantic_layers
            src/semantic_inflation_layer.cpp
            src/semantic_obstacle_layer.cpp
            src/semantic_layer.cpp
            src/semantic_proxemic_layer.cpp
            src/semantic_social_layer.cpp)
add_dependencies(semantic_layers ${PROJECT_NAME}_gencfg ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
