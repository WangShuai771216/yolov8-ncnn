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
include examples/CMakeFiles/rfcn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/rfcn.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/rfcn.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/rfcn.dir/flags.make

examples/CMakeFiles/rfcn.dir/rfcn.cpp.o: examples/CMakeFiles/rfcn.dir/flags.make
examples/CMakeFiles/rfcn.dir/rfcn.cpp.o: ../examples/rfcn.cpp
examples/CMakeFiles/rfcn.dir/rfcn.cpp.o: examples/CMakeFiles/rfcn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/rfcn.dir/rfcn.cpp.o"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/rfcn.dir/rfcn.cpp.o -MF CMakeFiles/rfcn.dir/rfcn.cpp.o.d -o CMakeFiles/rfcn.dir/rfcn.cpp.o -c /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/examples/rfcn.cpp

examples/CMakeFiles/rfcn.dir/rfcn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rfcn.dir/rfcn.cpp.i"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/examples/rfcn.cpp > CMakeFiles/rfcn.dir/rfcn.cpp.i

examples/CMakeFiles/rfcn.dir/rfcn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rfcn.dir/rfcn.cpp.s"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/examples/rfcn.cpp -o CMakeFiles/rfcn.dir/rfcn.cpp.s

# Object files for target rfcn
rfcn_OBJECTS = \
"CMakeFiles/rfcn.dir/rfcn.cpp.o"

# External object files for target rfcn
rfcn_EXTERNAL_OBJECTS =

examples/rfcn: examples/CMakeFiles/rfcn.dir/rfcn.cpp.o
examples/rfcn: examples/CMakeFiles/rfcn.dir/build.make
examples/rfcn: src/libncnn.a
examples/rfcn: /usr/local/lib/libopencv_highgui.so.4.5.1
examples/rfcn: /usr/local/lib/libopencv_videoio.so.4.5.1
examples/rfcn: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
examples/rfcn: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/rfcn: /usr/local/lib/libopencv_imgcodecs.so.4.5.1
examples/rfcn: /usr/local/lib/libopencv_imgproc.so.4.5.1
examples/rfcn: /usr/local/lib/libopencv_core.so.4.5.1
examples/rfcn: examples/CMakeFiles/rfcn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable rfcn"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rfcn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/rfcn.dir/build: examples/rfcn
.PHONY : examples/CMakeFiles/rfcn.dir/build

examples/CMakeFiles/rfcn.dir/clean:
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/rfcn.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/rfcn.dir/clean

examples/CMakeFiles/rfcn.dir/depend:
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wangshuai/WS/new_kl/kl_ncnn_yolov5 /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/examples /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/examples/CMakeFiles/rfcn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/rfcn.dir/depend

