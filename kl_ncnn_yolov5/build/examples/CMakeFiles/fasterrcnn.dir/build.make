# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wangshuai/WS/new_kl/kl_ncnn_yolov5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/fasterrcnn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/fasterrcnn.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/fasterrcnn.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/fasterrcnn.dir/flags.make

examples/CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.o: examples/CMakeFiles/fasterrcnn.dir/flags.make
examples/CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.o: ../examples/fasterrcnn.cpp
examples/CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.o: examples/CMakeFiles/fasterrcnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.o"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.o -MF CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.o.d -o CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.o -c /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/examples/fasterrcnn.cpp

examples/CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.i"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/examples/fasterrcnn.cpp > CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.i

examples/CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.s"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/examples/fasterrcnn.cpp -o CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.s

# Object files for target fasterrcnn
fasterrcnn_OBJECTS = \
"CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.o"

# External object files for target fasterrcnn
fasterrcnn_EXTERNAL_OBJECTS =

examples/fasterrcnn: examples/CMakeFiles/fasterrcnn.dir/fasterrcnn.cpp.o
examples/fasterrcnn: examples/CMakeFiles/fasterrcnn.dir/build.make
examples/fasterrcnn: src/libncnn.a
examples/fasterrcnn: /usr/local/lib/libopencv_highgui.so.4.5.1
examples/fasterrcnn: /usr/local/lib/libopencv_videoio.so.4.5.1
examples/fasterrcnn: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
examples/fasterrcnn: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/fasterrcnn: /usr/local/lib/libopencv_imgcodecs.so.4.5.1
examples/fasterrcnn: /usr/local/lib/libopencv_imgproc.so.4.5.1
examples/fasterrcnn: /usr/local/lib/libopencv_core.so.4.5.1
examples/fasterrcnn: examples/CMakeFiles/fasterrcnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fasterrcnn"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fasterrcnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/fasterrcnn.dir/build: examples/fasterrcnn
.PHONY : examples/CMakeFiles/fasterrcnn.dir/build

examples/CMakeFiles/fasterrcnn.dir/clean:
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/fasterrcnn.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/fasterrcnn.dir/clean

examples/CMakeFiles/fasterrcnn.dir/depend:
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wangshuai/WS/new_kl/kl_ncnn_yolov5 /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/examples /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples/CMakeFiles/fasterrcnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/fasterrcnn.dir/depend

