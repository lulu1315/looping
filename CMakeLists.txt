CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(looping)

FIND_PACKAGE(OpenCV 4.2.0 REQUIRED PATHS "/shared/foss-18/opencv-4.2.0_install/lib/cmake/opencv4")

ADD_EXECUTABLE(looping
"${CMAKE_SOURCE_DIR}/looping.cpp" 
)
ADD_EXECUTABLE(looping_ext
"${CMAKE_SOURCE_DIR}/looping_ext.cpp" 
)
#ADD_EXECUTABLE(seqmorph
#"${CMAKE_SOURCE_DIR}/seqmorph.cpp" 
#)
TARGET_LINK_LIBRARIES(looping ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(looping_ext ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(seqmorph ${OpenCV_LIBS})
