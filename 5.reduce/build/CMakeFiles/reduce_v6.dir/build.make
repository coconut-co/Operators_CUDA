# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce/build

# Include any dependencies generated for this target.
include CMakeFiles/reduce_v6.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/reduce_v6.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reduce_v6.dir/flags.make

CMakeFiles/reduce_v6.dir/reduce_v6.cu.o: CMakeFiles/reduce_v6.dir/flags.make
CMakeFiles/reduce_v6.dir/reduce_v6.cu.o: ../reduce_v6.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/reduce_v6.dir/reduce_v6.cu.o"
	/usr/local/cuda-11.8/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce/reduce_v6.cu -o CMakeFiles/reduce_v6.dir/reduce_v6.cu.o

CMakeFiles/reduce_v6.dir/reduce_v6.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/reduce_v6.dir/reduce_v6.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/reduce_v6.dir/reduce_v6.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/reduce_v6.dir/reduce_v6.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target reduce_v6
reduce_v6_OBJECTS = \
"CMakeFiles/reduce_v6.dir/reduce_v6.cu.o"

# External object files for target reduce_v6
reduce_v6_EXTERNAL_OBJECTS =

reduce_v6: CMakeFiles/reduce_v6.dir/reduce_v6.cu.o
reduce_v6: CMakeFiles/reduce_v6.dir/build.make
reduce_v6: CMakeFiles/reduce_v6.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable reduce_v6"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduce_v6.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reduce_v6.dir/build: reduce_v6

.PHONY : CMakeFiles/reduce_v6.dir/build

CMakeFiles/reduce_v6.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reduce_v6.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reduce_v6.dir/clean

CMakeFiles/reduce_v6.dir/depend:
	cd /home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce /home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce /home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce/build /home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce/build /home/yst/文档/jwj/cuda/panada/my_cuda/5.reduce/build/CMakeFiles/reduce_v6.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reduce_v6.dir/depend

