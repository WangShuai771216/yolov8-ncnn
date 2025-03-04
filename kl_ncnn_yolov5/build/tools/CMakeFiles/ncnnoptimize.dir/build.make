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
include tools/CMakeFiles/ncnnoptimize.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tools/CMakeFiles/ncnnoptimize.dir/compiler_depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/ncnnoptimize.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/ncnnoptimize.dir/flags.make

tools/CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.o: tools/CMakeFiles/ncnnoptimize.dir/flags.make
tools/CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.o: ../tools/ncnnoptimize.cpp
tools/CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.o: tools/CMakeFiles/ncnnoptimize.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.o"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tools/CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.o -MF CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.o.d -o CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.o -c /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/tools/ncnnoptimize.cpp

tools/CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.i"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/tools/ncnnoptimize.cpp > CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.i

tools/CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.s"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/tools/ncnnoptimize.cpp -o CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.s

# Object files for target ncnnoptimize
ncnnoptimize_OBJECTS = \
"CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.o"

# External object files for target ncnnoptimize
ncnnoptimize_EXTERNAL_OBJECTS =

tools/ncnnoptimize: tools/CMakeFiles/ncnnoptimize.dir/ncnnoptimize.cpp.o
tools/ncnnoptimize: tools/CMakeFiles/ncnnoptimize.dir/build.make
tools/ncnnoptimize: src/libncnn.a
tools/ncnnoptimize: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
tools/ncnnoptimize: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/ncnnoptimize: tools/CMakeFiles/ncnnoptimize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ncnnoptimize"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ncnnoptimize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/ncnnoptimize.dir/build: tools/ncnnoptimize
.PHONY : tools/CMakeFiles/ncnnoptimize.dir/build

tools/CMakeFiles/ncnnoptimize.dir/clean:
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/tools && $(CMAKE_COMMAND) -P CMakeFiles/ncnnoptimize.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/ncnnoptimize.dir/clean

tools/CMakeFiles/ncnnoptimize.dir/depend:
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wangshuai/WS/new_kl/kl_ncnn_yolov5 /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/tools /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/tools /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/tools/CMakeFiles/ncnnoptimize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/ncnnoptimize.dir/depend

