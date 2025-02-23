cmake_minimum_required(VERSION 3.1)
project(MyOpenCVProject)
#SET( CMAKE_CXX_COMPILER /usr/bin/clang++ )
SET( CMAKE_CXX_COMPILER /usr/bin/g++ )
set(CMAKE_CXX_FLAG "${CMAKE_C_FLAGS} -g -O0 -fmax-errors=5 -Wfatal-erros -std=c++17")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories( external/* modules )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/include/")

file(GLOB MY_SOURCE modules/pos/*.cpp modules/read/*.cpp common/*.cpp external/AdaptiveIntegralThresholding/thresh.cpp)

add_executable(main main.cpp ${MY_SOURCE})
add_executable(test test.cpp  ${MY_SOURCE})

add_executable(setting setting.cpp)
add_executable(prepareTemplate prepareTemplate.cpp)




# Include directories for this project
set(INCLUDE_DIR
  ${OPENCV_INCLUDE_DIRS}
  # ${PROJECT_SOURCE_DIR}/include
  -DRESOURCE_DIR
  ${CMAKE_SOURCE_DIR}/include/
)

# Add a include files

# For Eigen
#find_package( Eigen3 REQUIRED )
#include_directories(EIGEN3_INCLUDE_DIR)
include_directories("/usr/local/Cellar/eigen/3.3.7/include/eigen3")
include_directories("/usr/local/Cellar/boost/1.73.0/include")

# For OpenCV
#find_package(OpenCV 4.3.0 EXACT REQUIRED)
find_package(OpenCV 3.4.10  REQUIRED)
if(OpenCV_FOUND)
  #target_include_directories(tmp PUBLIC ${OpenCV_INCLUDE_DIRS})
  #  target_link_libraries(tmp ${OpenCV_LIBS})
    target_include_directories(test PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test ${OpenCV_LIBS})
    target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(main ${OpenCV_LIBS})

    target_include_directories(setting PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(setting ${OpenCV_LIBS})
    target_include_directories(prepareTemplate PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(prepareTemplate ${OpenCV_LIBS})
endif()

include_directories("/usr/local/Cellar/tesseract/4.1.1/include")
include_directories("/usr/local/Cellar/leptonica/1.80.0/include")
