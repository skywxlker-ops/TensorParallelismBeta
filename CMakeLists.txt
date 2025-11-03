cmake_minimum_required(VERSION 3.20)
project(dtensor_pybind LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.5
)
FetchContent_MakeAvailable(pybind11)

find_package(MPI REQUIRED)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# ---- NCCL (adjust if you keep them in custom paths) ----
find_path(NCCL_INCLUDE_DIR nccl.h PATH_SUFFIXES include)
find_library(NCCL_LIBRARY NAMES nccl PATH_SUFFIXES lib lib64)
if (NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL not found. Set NCCL_INCLUDE_DIR and NCCL_LIBRARY.")
endif()
get_filename_component(NCCL_LIB_DIR "${NCCL_LIBRARY}" DIRECTORY)

# Make CUDA arch configurable; default 86 if not set
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
message(STATUS "CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

# ---- Python extension target ----
pybind11_add_module(dtensor_alloc
  bindings/dtensor_pybind_matmul.cpp
  bindings/dtensor_alloc.cpp          # (Work, ProcessGroup, DTensor + allocator helpers)
  bindings/cachingAllocator.cpp       # (caching allocator impl)
)

target_include_directories(dtensor_alloc
  PRIVATE
    ${NCCL_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings
)

target_link_libraries(dtensor_alloc
  PRIVATE
    CUDA::cudart
    CUDA::cublas
    ${NCCL_LIBRARY}
    MPI::MPI_CXX
)

# Name the module exactly how Python imports it: dtensor.dtensor_cpp
set_target_properties(dtensor_alloc PROPERTIES
  PREFIX ""                                # no "lib" prefix on the .so
  OUTPUT_NAME "dtensor_cpp"
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# Embed rpaths so you don't need LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set_target_properties(dtensor_alloc PROPERTIES
  BUILD_RPATH "${NCCL_LIB_DIR};${CUDAToolkit_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN"
)

# Install next to your package so: import dtensor.dtensor_cpp
install(TARGETS dtensor_alloc
  LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python/dtensor
)
