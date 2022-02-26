# cmake version to be used
cmake_minimum_required( VERSION 3.0 )

# project name
project(directAlign)

# external libs
find_package(Ceres REQUIRED)
# Or find_package(Ceres REQUIRED PATHS "/path/to/install/dir") if the install directory is not in your PATH

# external libs
find_package(OpenCV REQUIRED)

# target multiSalign_sphericalHomog_hyperspectral2gray
add_executable(multiSalign_sphericalHomog_hyperspectral2gray multiSalign_sphericalHomog_hyperspectral2gray.cpp )

#below is required for reading npy files
if(NOT DEFINED libnpy_INCLUDE_DIRS)
	MESSAGE("Please set libnpy_INCLUDE_DIRS variable to the include directory of libnpy.")
endif(NOT DEFINED libnpy_INCLUDE_DIRS)

#below is required for the RANSAC implementation
if(NOT DEFINED OpenCV_SRC_DIR)
	MESSAGE("Please set OpenCV_SRC_DIR variable to the source directory of opencv, i.e. the one containing the modules directory.")
endif(NOT DEFINED OpenCV_SRC_DIR)

SET(OpenCV_INCLUDE_SOURCE_DIRS "${OpenCV_INCLUDE_DIRS}/opencv2/"
																"${OpenCV_SRC_DIR}/modules/calib3d/src/"
																"${OpenCV_SRC_DIR}/modules/core/include/")

target_include_directories(multiSalign_sphericalHomog_hyperspectral2gray
  PRIVATE
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_SOURCE_DIRS}
    ${libnpy_INCLUDE_DIRS}
)

target_link_libraries(multiSalign_sphericalHomog_hyperspectral2gray
  PRIVATE
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS}
)

