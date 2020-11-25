cmake_minimum_required(VERSION 3.1)
project(MyOpenCVProject)
SET( CMAKE_CXX_COMPILER /usr/bin/clang++ )
set(CMAKE_CXX_FLAG "${CMAKE_C_FLAGS} -g -O0 -fmax-errors=5 -Wfatal-erros -std=c++17")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories( external/* modules )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/include/")

file(GLOB MY_SOURCE modules/pos/*.cpp modules/read/*.cpp common/*.cpp external/AdaptiveIntegralThresholding/thresh.cpp)

add_executable(main main.cpp ${MY_SOURCE})
add_executable(movie movie.cpp ${MY_SOURCE})
add_executable(final_main final_main.cpp ${MY_SOURCE})
add_executable(minimumColorDifference minimumColorDifference.cpp ${MY_SOURCE})
add_executable(test test.cpp  ${MY_SOURCE})

#add_executable(tmp tmp.cpp  modules/pos/Homography.cpp modules/pos/calib.cpp modules/read/difference.cpp modules/read/template.cpp common/init.cpp)
#add_executable(main main.cpp pos/Homography.cpp pos/calib.cpp read/difference.cpp read/readability.cpp read/template.cpp common/init.cpp)
#add_executable(opt opt.cpp pos/fromtwo.cpp sub/fit.cpp pos/Homography.cpp pos/calib.cpp read/difference.cpp read/readability.cpp read/template.cpp common/init.cpp)
#add_executable(makeMask makeMask.cpp  pos/Homography.cpp pos/calib.cpp read/difference.cpp read/readability.cpp read/template.cpp common/init.cpp)
#add_executable(justRead AdaptiveIntegralThresholding/thresh.cpp justRead.cpp pos/fromtwo.cpp sub/fit.cpp pos/Homography.cpp pos/calib.cpp read/difference.cpp read/readability.cpp read/template.cpp common/init.cpp)
#add_executable(minimumColorDifference AdaptiveIntegralThresholding/thresh.cpp minimumColorDifference.cpp  pos/Homography.cpp pos/calib.cpp read/difference.cpp read/readability.cpp read/template.cpp common/init.cpp)
#add_executable(fitEllipseToRoi AdaptiveIntegralThresholding/thresh.cpp testsuite/fitEllipseToRoi.cpp pos/fromtwo.cpp sub/fit.cpp pos/Homography.cpp pos/calib.cpp read/difference.cpp read/readability.cpp read/template.cpp common/init.cpp)
#add_executable(newHomography newHomography.cpp pos/fromtwo.cpp sub/fit.cpp pos/Homography.cpp pos/calib.cpp read/difference.cpp read/readability.cpp read/template.cpp common/init.cpp)
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
    target_include_directories(movie PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(movie ${OpenCV_LIBS})
    target_include_directories(final_main PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(final_main ${OpenCV_LIBS})
    #  target_include_directories(justRead PUBLIC ${OpenCV_INCLUDE_DIRS})
    #target_link_libraries(justRead ${OpenCV_LIBS})
    target_include_directories(minimumColorDifference PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(minimumColorDifference ${OpenCV_LIBS})

    #target_link_libraries(opt ${OpenCV_LIBS})
    #target_link_libraries(fitEllipseToRoi ${OpenCV_LIBS})
    #target_link_libraries(makeMask ${OpenCV_LIBS})
    #target_include_directories(newHomography PUBLIC ${OpenCV_INCLUDE_DIRS})
    #target_link_libraries(newHomography ${OpenCV_LIBS})
    target_include_directories(setting PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(setting ${OpenCV_LIBS})
    target_include_directories(prepareTemplate PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(prepareTemplate ${OpenCV_LIBS})
endif()

include_directories("/usr/local/Cellar/tesseract/4.1.1/include")
include_directories("/usr/local/Cellar/leptonica/1.80.0/include")
