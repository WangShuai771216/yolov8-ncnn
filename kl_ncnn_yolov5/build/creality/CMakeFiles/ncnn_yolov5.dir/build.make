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
include creality/CMakeFiles/ncnn_yolov5.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include creality/CMakeFiles/ncnn_yolov5.dir/compiler_depend.make

# Include the progress variables for this target.
include creality/CMakeFiles/ncnn_yolov5.dir/progress.make

# Include the compile flags for this target's objects.
include creality/CMakeFiles/ncnn_yolov5.dir/flags.make

creality/CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.o: creality/CMakeFiles/ncnn_yolov5.dir/flags.make
creality/CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.o: ../creality/yolov5.cpp
creality/CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.o: creality/CMakeFiles/ncnn_yolov5.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object creality/CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.o"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT creality/CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.o -MF CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.o.d -o CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.o -c /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/creality/yolov5.cpp

creality/CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.i"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/creality/yolov5.cpp > CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.i

creality/CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.s"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/creality/yolov5.cpp -o CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.s

creality/CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.o: creality/CMakeFiles/ncnn_yolov5.dir/flags.make
creality/CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.o: ../creality/gcode_cut.cpp
creality/CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.o: creality/CMakeFiles/ncnn_yolov5.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object creality/CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.o"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT creality/CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.o -MF CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.o.d -o CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.o -c /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/creality/gcode_cut.cpp

creality/CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.i"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/creality/gcode_cut.cpp > CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.i

creality/CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.s"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/creality/gcode_cut.cpp -o CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.s

creality/CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.o: creality/CMakeFiles/ncnn_yolov5.dir/flags.make
creality/CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.o: ../creality/AI_property.cpp
creality/CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.o: creality/CMakeFiles/ncnn_yolov5.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object creality/CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.o"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT creality/CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.o -MF CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.o.d -o CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.o -c /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/creality/AI_property.cpp

creality/CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.i"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/creality/AI_property.cpp > CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.i

creality/CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.s"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/creality/AI_property.cpp -o CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.s

# Object files for target ncnn_yolov5
ncnn_yolov5_OBJECTS = \
"CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.o" \
"CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.o" \
"CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.o"

# External object files for target ncnn_yolov5
ncnn_yolov5_EXTERNAL_OBJECTS =

creality/libncnn_yolov5.so: creality/CMakeFiles/ncnn_yolov5.dir/yolov5.cpp.o
creality/libncnn_yolov5.so: creality/CMakeFiles/ncnn_yolov5.dir/gcode_cut.cpp.o
creality/libncnn_yolov5.so: creality/CMakeFiles/ncnn_yolov5.dir/AI_property.cpp.o
creality/libncnn_yolov5.so: creality/CMakeFiles/ncnn_yolov5.dir/build.make
creality/libncnn_yolov5.so: src/libncnn.a
creality/libncnn_yolov5.so: /usr/local/lib/libopencv_highgui.so.4.5.1
creality/libncnn_yolov5.so: /usr/local/lib/libopencv_videoio.so.4.5.1
creality/libncnn_yolov5.so: /usr/local/lib/libopencv_calib3d.so.4.5.1
creality/libncnn_yolov5.so: /usr/lib/x86_64-linux-gnu/libcurl.so
creality/libncnn_yolov5.so: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
creality/libncnn_yolov5.so: /usr/lib/x86_64-linux-gnu/libpthread.so
creality/libncnn_yolov5.so: /usr/local/lib/libopencv_imgcodecs.so.4.5.1
creality/libncnn_yolov5.so: /usr/local/lib/libopencv_features2d.so.4.5.1
creality/libncnn_yolov5.so: /usr/local/lib/libopencv_imgproc.so.4.5.1
creality/libncnn_yolov5.so: /usr/local/lib/libopencv_flann.so.4.5.1
creality/libncnn_yolov5.so: /usr/local/lib/libopencv_core.so.4.5.1
creality/libncnn_yolov5.so: creality/CMakeFiles/ncnn_yolov5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library libncnn_yolov5.so"
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ncnn_yolov5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
creality/CMakeFiles/ncnn_yolov5.dir/build: creality/libncnn_yolov5.so
.PHONY : creality/CMakeFiles/ncnn_yolov5.dir/build

creality/CMakeFiles/ncnn_yolov5.dir/clean:
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality && $(CMAKE_COMMAND) -P CMakeFiles/ncnn_yolov5.dir/cmake_clean.cmake
.PHONY : creality/CMakeFiles/ncnn_yolov5.dir/clean

creality/CMakeFiles/ncnn_yolov5.dir/depend:
	cd /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wangshuai/WS/new_kl/kl_ncnn_yolov5 /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/creality /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality /home/wangshuai/WS/new_kl/kl_ncnn_yolov5/build/creality/CMakeFiles/ncnn_yolov5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : creality/CMakeFiles/ncnn_yolov5.dir/depend

