cmake_minimum_required(VERSION 3.1)
project(MyOpenCVProject)

# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${PROJECT_BINARY_DIR}/include/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/include/")

# Create Main project
set(SOURCES
    main.cpp
)

set(HEADERS
    # main.h
)

add_executable(Main
    ${SOURCES}
    ${HEADERS}
)


# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    target_include_directories(Main PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(Main ${OpenCV_LIBS})
endif()


