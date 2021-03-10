cmake_minimum_required(VERSION 3.10)
project(ncnn_test)

set(CMAKE_CXX_STANDARD 11)

# 复制文件
file(COPY ./images DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ./models DESTINATION ${CMAKE_BINARY_DIR})

# 调用OpenCV
find_package( OpenCV 3 REQUIRED )
if (OPENCV_FOUND)
    message("OpenCV Found.")
endif ()

# 配置OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OpenMP Found.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 配置NCNN
set(ncnn_DIR ~/software/ncnn_with_vulkan/lib/cmake/ncnn)
find_package(ncnn)

add_executable(ncnn_test src/main.cpp)

target_link_libraries(ncnn_test ncnn ${OpenCV_LIBS})
