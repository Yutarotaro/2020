cmake_minimum_required(VERSION 3.1)
project(MyOpenCVProject)
SET( CMAKE_CXX_COMPILER /usr/bin/clang++ )
# Copy resouce
#file(COPY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${PROJECT_BINARY_DIR}/include/)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/include/")
add_definitions( -std=c++17 -Wall )

add_executable(main main.cpp pos/module.cpp read/read.cpp read/difference.cpp common/init.cpp)
add_executable(setting setting.cpp)

# Include directories for this project
set(INCLUDE_DIR
  ${OPENCV_INCLUDE_DIRS}
  # ${PROJECT_SOURCE_DIR}/include
  -DRESOURCE_DIR
  ${CMAKE_SOURCE_DIR}/include/
)

# Add a include files
include_directories("${INCLUDE_DIR}")

# For Eigen
find_package( Eigen3 REQUIRED )
include_directories(EIGEN3_INCLUDE_DIR)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(main ${OpenCV_LIBS})
    target_include_directories(setting PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(setting ${OpenCV_LIBS})
endif()


