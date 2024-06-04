# Install script for directory: /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/src/libncnn.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ncnn" TYPE FILE FILES
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/allocator.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/benchmark.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/blob.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/c_api.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/command.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/cpu.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/datareader.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/gpu.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/layer.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/layer_shader_type.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/layer_type.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/mat.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/modelbin.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/net.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/option.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/paramdict.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/pipeline.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/pipelinecache.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/simpleocv.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/simpleomp.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/simplestl.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/src/vulkan_header_fix.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/src/ncnn_export.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/src/layer_shader_type_enum.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/src/layer_type_enum.h"
    "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/src/platform.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake"
         "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/src/ncnnConfig.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/src/ncnn.pc")
endif()

