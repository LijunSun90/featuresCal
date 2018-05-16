# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project( featuresCal )

set(OpenCV_DIR /usr/local/share/OpenCV)

find_package( OpenCV 3.4.1 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB sources_list
    "src/*.h"
    "src/*.cpp"
)

set(LIB_SRC_FILES ${sources_list})

add_executable( featuresCal ${LIB_SRC_FILES} )

# Link your application with OpenCV libraries
target_link_libraries( featuresCal ${OpenCV_LIBS} )
