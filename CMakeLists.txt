# cmake for OpenCV Cookbook 3rd edition chapter 07
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)
set(OpenCV_DIR "/home/lb/gitcode/opencv4.5/build")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# add executable
add_executable( face 09_haar_face_video.cpp)
add_executable( face1 facedetectcpu.cpp)
add_executable( face2 08_haar_face_gpu.cpp)
add_executable( test testgpu.cpp)
add_executable( test1 01_image_read.cpp)
add_executable( orb 06_orb_gpu.cpp)
add_executable( eye 10_haar_eye_gpu.cpp)
add_executable( test2 coaltest.cpp)

# link libraries
target_link_libraries( face ${OpenCV_LIBS})
target_link_libraries( face1 ${OpenCV_LIBS})
target_link_libraries( face2 ${OpenCV_LIBS})
target_link_libraries( test ${OpenCV_LIBS})
target_link_libraries( test1 ${OpenCV_LIBS})
target_link_libraries( orb ${OpenCV_LIBS})
target_link_libraries( eye ${OpenCV_LIBS})
target_link_libraries( test2 ${OpenCV_LIBS})

# Print some message showing some of them
message("##############################################")
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBRARIES}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    OpenCV_LIBS path: ${OpenCV_LIBS}")
message("##############################################")

# target_link_libraries( blobs ${OpenCV_LIBS})

# copy required images to every directory with executable
# SET (IMAGES ${CMAKE_SOURCE_DIR}/images/group.jpg 
#             ${CMAKE_SOURCE_DIR}/images/binaryGroup.bmp
#             ${CMAKE_SOURCE_DIR}/images/mser.bmp
#             ${CMAKE_SOURCE_DIR}/images/road.jpg
#             ${CMAKE_SOURCE_DIR}/images/chariot.jpg
#             ${CMAKE_SOURCE_DIR}/images/pidai002.png
#             )
# FILE(COPY ${IMAGES} DESTINATION .)
# FILE(COPY ${IMAGES} DESTINATION "Debug")
# FILE(COPY ${IMAGES} DESTINATION "Release")
