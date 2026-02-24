cmake_minimum_required(VERSION 3.16)

project(VSLAM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/Users/Laksh/OpenCV/opencv/build")
find_package(OpenCV REQUIRED)

add_executable(vslam
    src/main.cpp
)

add_executable(test
    src/test.cpp
    src/features.cpp
)

target_link_libraries(vslam
    ${OpenCV_LIBS}
)

target_link_libraries(test
    ${OpenCV_LIBS}
)
