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
find_package(OpenMP REQUIRED)

# ---- NCCL ----
find_path(NCCL_INCLUDE_DIR nccl.h PATH_SUFFIXES include)
find_library(NCCL_LIBRARY NAMES nccl PATH_SUFFIXES lib lib64)
if (NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL not found. Set NCCL_INCLUDE_DIR and NCCL_LIBRARY.")
endif()
get_filename_component(NCCL_LIB_DIR "${NCCL_LIBRARY}" DIRECTORY)

# CUDA archs (keep your original 52 if that’s your GPU)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 52)
endif()
message(STATUS "CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

# ---- TBB (define target before linking it) ----
find_package(TBB CONFIG QUIET)
if (NOT TBB_FOUND)
  find_package(TBB QUIET)
endif()
if (NOT TBB_FOUND)
  find_path(TBB_INCLUDE_DIR tbb/tbb.h HINTS /usr/include /usr/local/include)
  find_library(TBB_LIB tbb HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib)
  if (NOT TBB_INCLUDE_DIR OR NOT TBB_LIB)
    message(FATAL_ERROR "TBB not found. Set TBB_INCLUDE_DIR and TBB_LIB.")
  endif()
  add_library(TBB::tbb UNKNOWN IMPORTED)
  set_target_properties(TBB::tbb PROPERTIES
    IMPORTED_LOCATION "${TBB_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}")
endif()

# ---- Python extension target ----
pybind11_add_module(dtensor_cpp
  bindings/dtensor_pybind_integrated.cpp
  bindings/dtensor_integrated.cpp
  bindings/cachingAllocator.cpp
)

# ---- Import prebuilt kathirs_tensor lib (contains OwnTensor::cuda_matmul) ----
set(KATHIRS_LIB    "${CMAKE_CURRENT_SOURCE_DIR}/kathirs_tensor/lib/libtensor.so")
set(KATHIRS_INC    "${CMAKE_CURRENT_SOURCE_DIR}/kathirs_tensor/include")
get_filename_component(KATHIRS_LIB_DIR "${KATHIRS_LIB}" DIRECTORY)

add_library(kathirs_tensor_lib SHARED IMPORTED)
set_target_properties(kathirs_tensor_lib PROPERTIES
  IMPORTED_LOCATION             "${KATHIRS_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${KATHIRS_INC}"
)

# Includes
target_include_directories(dtensor_cpp
  PRIVATE
    ${NCCL_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings"
    "${KATHIRS_INC}"
)

# Compile defs
target_compile_definitions(dtensor_cpp PRIVATE WITH_CUDA)

# Link (single, consolidated call)
target_link_libraries(dtensor_cpp
  PRIVATE
    kathirs_tensor_lib
    CUDA::cudart CUDA::cublas CUDA::curand
    ${NCCL_LIBRARY} MPI::MPI_CXX TBB::tbb
    OpenMP::OpenMP_CXX
)

# Resolve CUDA lib dir for RPATH
get_target_property(_cudart_loc CUDA::cudart LOCATION)
if(_cudart_loc)
  get_filename_component(CUDA_LIB_DIR "${_cudart_loc}" DIRECTORY)
endif()

# Put the built .so where your PYTHONPATH expects it, and embed RPATHs
set_target_properties(dtensor_cpp PROPERTIES
  PREFIX ""
  OUTPUT_NAME "dtensor_cpp"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dtensor_pywrap/python/dtensor"
  BUILD_RPATH  "${KATHIRS_LIB_DIR};${NCCL_LIB_DIR};${CUDA_LIB_DIR}"
  INSTALL_RPATH "${KATHIRS_LIB_DIR};${NCCL_LIB_DIR};${CUDA_LIB_DIR}"
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# Install alongside your Python package
install(TARGETS dtensor_cpp
  LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/dtensor_pywrap/python/dtensor"
)
