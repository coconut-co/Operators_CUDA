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
CMAKE_SOURCE_DIR = /home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram/build

# Include any dependencies generated for this target.
include CMakeFiles/histogram_v1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/histogram_v1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/histogram_v1.dir/flags.make

CMakeFiles/histogram_v1.dir/histogram_v1.cu.o: CMakeFiles/histogram_v1.dir/flags.make
CMakeFiles/histogram_v1.dir/histogram_v1.cu.o: ../histogram_v1.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/histogram_v1.dir/histogram_v1.cu.o"
	/usr/local/cuda-11.8/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram/histogram_v1.cu -o CMakeFiles/histogram_v1.dir/histogram_v1.cu.o

CMakeFiles/histogram_v1.dir/histogram_v1.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/histogram_v1.dir/histogram_v1.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/histogram_v1.dir/histogram_v1.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/histogram_v1.dir/histogram_v1.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target histogram_v1
histogram_v1_OBJECTS = \
"CMakeFiles/histogram_v1.dir/histogram_v1.cu.o"

# External object files for target histogram_v1
histogram_v1_EXTERNAL_OBJECTS =

histogram_v1: CMakeFiles/histogram_v1.dir/histogram_v1.cu.o
histogram_v1: CMakeFiles/histogram_v1.dir/build.make
histogram_v1: CMakeFiles/histogram_v1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable histogram_v1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/histogram_v1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/histogram_v1.dir/build: histogram_v1

.PHONY : CMakeFiles/histogram_v1.dir/build

CMakeFiles/histogram_v1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/histogram_v1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/histogram_v1.dir/clean

CMakeFiles/histogram_v1.dir/depend:
	cd /home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram /home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram /home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram/build /home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram/build /home/yst/文档/jwj/cuda/panada/my_cuda/6.histogram/build/CMakeFiles/histogram_v1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/histogram_v1.dir/depend

