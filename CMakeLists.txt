cmake_minimum_required(VERSION 2.8)

##################
# CMAKE settings #
##################

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#########################
# Project Specification #
#########################

project(ImageToTkiZ)

#################
# Safety Checks #
#################

# Avoid in-source builds
if ("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
    message(FATAL_ERROR "Warning: In-source build! Aborting. Call cmake .. from a build folder.")
endif()

message(${CMAKE_CURRENT_BINARY_DIR})

#####################
# Compiler Settings #
#####################

message("Setting compiler settings:")
if( WIN32 )
    message("-- Windows detected")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
elseif( APPLE )
    message("-- OSX detected")
    # Set C++11
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    # Enable all warnings
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else()
    message("-- Linux detected")
    # Set C++11
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    # Enable all warnings
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

endif()
message("-- cxx Flags: " ${CMAKE_CXX_FLAGS})
message("-- Finished setting compiler settings.")
message("\n")

###################
# Linker settings #
###################

message("Setting linker settings:")
if( WIN32 )
    message("-- Windows detected -- setting corresponding linker settings")
elseif( APPLE )
    message("-- OSX detected -- setting corresponding linker settings")
else()
    message("-- Linux detected -- setting corresponding linker settings")
endif()
message("-- Finished setting linker settings.")
message("\n")

################
# Dependencies #
################

# OPENCV

set(OpenCV_STATIC OFF)
find_package( OpenCV REQUIRED )

include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

###############
# Directories #
###############

set(IMAGETOTKIZ_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(IMAGETOTKIZ_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

#########
# Files #
#########

set(IMAGETOTKIZ_SOURCES
        ${IMAGETOTKIZ_SOURCE_DIR}/main.cpp
    )

#############
# Libraries #
#############

##
##############
# Executable #
##############

add_executable(ImageToTkiZ ${IMAGETOTKIZ_SOURCES})
target_link_libraries(ImageToTkiZ ${OpenCV_LIBS})

MACRO (COPY_DLL trgt libname)
        ADD_CUSTOM_COMMAND (
        TARGET ${trgt}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy "${OPENCVDIR/bin}/${libname}.dll" ${CMAKE_CURRENT_BINARY_DIR}
    )
ENDMACRO (COPY_DLL)