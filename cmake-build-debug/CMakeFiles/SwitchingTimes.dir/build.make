# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/nlbr/CLionProjects/SwitchingTimes

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/nlbr/CLionProjects/SwitchingTimes/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/SwitchingTimes.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SwitchingTimes.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SwitchingTimes.dir/flags.make

CMakeFiles/SwitchingTimes.dir/main.cpp.o: CMakeFiles/SwitchingTimes.dir/flags.make
CMakeFiles/SwitchingTimes.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/nlbr/CLionProjects/SwitchingTimes/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SwitchingTimes.dir/main.cpp.o"
	/usr/local/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SwitchingTimes.dir/main.cpp.o -c /Users/nlbr/CLionProjects/SwitchingTimes/main.cpp

CMakeFiles/SwitchingTimes.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SwitchingTimes.dir/main.cpp.i"
	/usr/local/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nlbr/CLionProjects/SwitchingTimes/main.cpp > CMakeFiles/SwitchingTimes.dir/main.cpp.i

CMakeFiles/SwitchingTimes.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SwitchingTimes.dir/main.cpp.s"
	/usr/local/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nlbr/CLionProjects/SwitchingTimes/main.cpp -o CMakeFiles/SwitchingTimes.dir/main.cpp.s

CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.o: CMakeFiles/SwitchingTimes.dir/flags.make
CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.o: ../src/switching-times.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/nlbr/CLionProjects/SwitchingTimes/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.o"
	/usr/local/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.o -c /Users/nlbr/CLionProjects/SwitchingTimes/src/switching-times.cpp

CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.i"
	/usr/local/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nlbr/CLionProjects/SwitchingTimes/src/switching-times.cpp > CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.i

CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.s"
	/usr/local/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nlbr/CLionProjects/SwitchingTimes/src/switching-times.cpp -o CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.s

CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.o: CMakeFiles/SwitchingTimes.dir/flags.make
CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.o: ../src/switching-times-example.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/nlbr/CLionProjects/SwitchingTimes/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.o"
	/usr/local/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.o -c /Users/nlbr/CLionProjects/SwitchingTimes/src/switching-times-example.cpp

CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.i"
	/usr/local/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nlbr/CLionProjects/SwitchingTimes/src/switching-times-example.cpp > CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.i

CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.s"
	/usr/local/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nlbr/CLionProjects/SwitchingTimes/src/switching-times-example.cpp -o CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.s

# Object files for target SwitchingTimes
SwitchingTimes_OBJECTS = \
"CMakeFiles/SwitchingTimes.dir/main.cpp.o" \
"CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.o" \
"CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.o"

# External object files for target SwitchingTimes
SwitchingTimes_EXTERNAL_OBJECTS =

SwitchingTimes: CMakeFiles/SwitchingTimes.dir/main.cpp.o
SwitchingTimes: CMakeFiles/SwitchingTimes.dir/src/switching-times.cpp.o
SwitchingTimes: CMakeFiles/SwitchingTimes.dir/src/switching-times-example.cpp.o
SwitchingTimes: CMakeFiles/SwitchingTimes.dir/build.make
SwitchingTimes: /usr/local/Cellar/gcc/9.2.0_3/lib/gcc/9/libgomp.dylib
SwitchingTimes: CMakeFiles/SwitchingTimes.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/nlbr/CLionProjects/SwitchingTimes/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable SwitchingTimes"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SwitchingTimes.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SwitchingTimes.dir/build: SwitchingTimes

.PHONY : CMakeFiles/SwitchingTimes.dir/build

CMakeFiles/SwitchingTimes.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SwitchingTimes.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SwitchingTimes.dir/clean

CMakeFiles/SwitchingTimes.dir/depend:
	cd /Users/nlbr/CLionProjects/SwitchingTimes/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/nlbr/CLionProjects/SwitchingTimes /Users/nlbr/CLionProjects/SwitchingTimes /Users/nlbr/CLionProjects/SwitchingTimes/cmake-build-debug /Users/nlbr/CLionProjects/SwitchingTimes/cmake-build-debug /Users/nlbr/CLionProjects/SwitchingTimes/cmake-build-debug/CMakeFiles/SwitchingTimes.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SwitchingTimes.dir/depend

