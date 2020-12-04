cmake_minimum_required(VERSION 3.17)
project(opencvtest)

set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )

# tell the build to include the headers from OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )
# tell it to link the executable target against OpenCV

add_executable(opencvtest main.cpp)

target_link_libraries(opencvtest ${OpenCV_LIBS} )