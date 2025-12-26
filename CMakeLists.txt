cmake_minimum_required(VERSION 3.16)
project(mujoco_experience)

# MuJoCo requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add MuJoCo submodule
add_subdirectory(mujoco)

# You can add your own C++ examples here later
# add_executable(my_demo src/my_demo.cc)
# target_link_libraries(my_demo mujoco::mujoco)
