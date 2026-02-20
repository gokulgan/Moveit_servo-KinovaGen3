

cmake_minimum_required(VERSION 3.8)
project(servo_keyboard_kinova)

# Default to C++17 - Required for 'auto' in lambda parameters
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# --- Android Servo Node ---
#add_executable(servo_android_kinova 
	#  src/servo_android_kinova.cpp
  #)

# CRITICAL: This line tells the compiler to look in 'src' for "json.hpp"
#target_include_directories(servo_android_kinova PRIVATE src)

# --- Keyboard Servo Node ---
add_executable(servo_keyboard_kinova 
  src/servo_keyboard_kinova.cpp
)

# Link dependencies for Android Node
#ament_target_dependencies(servo_android_kinova
	#  rclcpp
  #  geometry_msgs
  #  control_msgs
  #  moveit_msgs
  #  std_msgs
  #  std_srvs
  #)

# Link dependencies for Keyboard Node
ament_target_dependencies(servo_keyboard_kinova
  rclcpp
  geometry_msgs
  control_msgs
  moveit_msgs
  std_msgs
  std_srvs
)

# Install both executables
install(TARGETS 
	#  servo_android_kinova
  servo_keyboard_kinova
  DESTINATION lib/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(
  rclcpp
  geometry_msgs
  control_msgs
  moveit_msgs
  std_msgs
  std_srvs
)

ament_package()
