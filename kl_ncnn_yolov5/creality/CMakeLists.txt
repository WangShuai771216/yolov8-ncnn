macro(ncnn_add_example name)
    add_executable(${name} ${name}.cpp)
    if(OpenCV_FOUND)
        target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${name} PRIVATE ncnn ${OpenCV_LIBS})
    elseif(NCNN_SIMPLEOCV)
        target_compile_definitions(${name} PUBLIC USE_NCNN_SIMPLEOCV)
        target_link_libraries(${name} PRIVATE ncnn)
    endif()

    # add test to a virtual project group
    set_property(TARGET ${name} PROPERTY FOLDER "examples")
endmacro()

if(NCNN_PIXEL)
    if(NOT NCNN_SIMPLEOCV)
        find_package(OpenCV QUIET COMPONENTS opencv_world)
        # for opencv 2.4 on ubuntu 16.04, there is no opencv_world but OpenCV_FOUND will be TRUE
        if("${OpenCV_LIBS}" STREQUAL "")
            set(OpenCV_FOUND FALSE)
        endif()
        if(NOT OpenCV_FOUND)
            find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio calib3d)
        endif()
        if(NOT OpenCV_FOUND)
            find_package(OpenCV QUIET COMPONENTS core highgui imgproc calib3d)
        endif()
    endif()

    if(OpenCV_FOUND OR NCNN_SIMPLEOCV)
        if(OpenCV_FOUND)
            message(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
            message(STATUS "    version: ${OpenCV_VERSION}")
            message(STATUS "    libraries: ${OpenCV_LIBS}")
            message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

            if(${OpenCV_VERSION_MAJOR} GREATER 3)
                set(CMAKE_CXX_STANDARD 11)
            endif()
        endif()

        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
        include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

        # add_library(ncnn_yolov5 SHARED yolov5.cpp )
        add_library(ncnn_yolov5 SHARED yolov5.cpp  gcode_cut.cpp AI_property.cpp)

        target_include_directories(ncnn_yolov5 PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(ncnn_yolov5 PRIVATE ncnn ${OpenCV_LIBS})

        ncnn_add_example(detection)
        ncnn_add_example(yolo_attention)
        target_link_libraries(detection PRIVATE ncnn_yolov5)
    else()
        message(WARNING "OpenCV not found and NCNN_SIMPLEOCV disabled, examples won't be built")
    endif()
else()
    message(WARNING "NCNN_PIXEL not enabled, examples won't be built")
endif()

# 查找json-c库
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONC REQUIRED json-c)
include_directories(${JSONC_INCLUDE_DIRS})
target_link_libraries(ncnn_yolov5 PRIVATE ${JSONC_LIBRARIES})

# 查找并链接CURL库
find_package(CURL REQUIRED)
if(CURL_FOUND)
    include_directories(${CURL_INCLUDE_DIRS})
    target_link_libraries(ncnn_yolov5 PRIVATE ${CURL_LIBRARIES})
else()
    message(FATAL_ERROR "CURL library not found")
endif()