cmake_minimum_required(VERSION 3.20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # compile files

# Prefer a Blackwell-ready CUDA toolchain by default, but allow callers to
# override it explicitly with -DCMAKE_CUDA_COMPILER=...
if(UNIX AND (NOT DEFINED CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER STREQUAL ""))
  if(EXISTS "/usr/local/cuda-12.8/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
  elseif(EXISTS "/usr/local/cuda-11.8/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
  endif()
endif()

project(MOAI-GPU LANGUAGES CXX CUDA)

# CMAKE config
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Debugging config
option(SHOW_WARNINGS "Enable warnings" OFF)
if (SHOW_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Wpedantic,-Wno-comment")
endif()

# Language config
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA config
# Default to Blackwell for RTX PRO 6000, but keep command-line overrides working.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --default-stream per-thread")

# Print CUDA info
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")

# PhantomFHE config
option(PHANTOM_USE_CUDA_PTX "Use CUDA PTX Assembly" ON)
message(STATUS "Use CUDA PTX Assembly: ${PHANTOM_USE_CUDA_PTX}")
if (PHANTOM_USE_CUDA_PTX)
    add_compile_definitions(PHANTOM_USE_CUDA_PTX)
endif()

# PhantomFHE
include_directories(thirdparty/phantom-fhe/include)


file(GLOB_RECURSE MOAI_CUDA_SOURCE_FILES CONFIGURE_DEPENDS
  "src/test.cu"
  "src/include/source/matrix_mul/Ct_pt_matrix_mul.cuh"
  "src/include/source/matrix_mul/Batch_encode_encrypt.cuh"
  "src/include/source/matrix_mul/Ct_ct_matrix_mul.cuh"
  "src/include/source/non_linear_func/gelu_other.cuh"
  "src/include/source/non_linear_func/layernorm.cuh"
  "src/include/source/non_linear_func/softmax.cuh"
  "src/include/test/test_phantom_ckks.cuh"
  "src/include/test/matrix_mul/test_batch_encode_encrypt.hpp"
  "src/include/test/matrix_mul/test_ct_pt_matrix_mul.cu"
  "src/include/test/matrix_mul/test_ct_ct_matrix_mul.cu"
  "src/include/test/non_linear_func/test_gelu.cuh"
  "src/include/test/non_linear_func/test_layernorm.cuh"
  "src/include/test/bootstrapping/bootstrapping.cuh"
  "src/include/test/non_linear_func/test_softmax.cuh"
  "src/include/test/non_linear_func/test_BPmax_BatchLN.cuh"
  "src/include/test/test_single_layer.cuh"
  "src/include/source/utils.cuh"
  "src/include/source/utils_moai.cuh"
  # "src/bootstrapping/*.cu"
  # "src/bootstrapping/*.cu"
  # "src/source/bootstrapping.cu"
  "src/include/source/bootstrapping/*.cu"
  "src/include/source/ckks_evaluator_parallel.cu"
  "thirdparty/phantom-fhe/src/*.cu"
  "thirdparty/phantom-fhe/src/*/*.cu"
)



add_executable(test ${MOAI_CUDA_SOURCE_FILES})

# Phantom-FHE sources live in src/*.cu and src/*/*.cu (ntt/, util/, …)
file(GLOB PHANTOM_FHE_TOP CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/*.cu")
file(GLOB PHANTOM_FHE_SUB CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/src/*/*.cu")
set(PHANTOM_FHE_SRC ${PHANTOM_FHE_TOP} ${PHANTOM_FHE_SUB})
add_executable(moai_gen_keys src/scripts/gen_moai_keys.cu ${PHANTOM_FHE_SRC})

find_package(OpenMP REQUIRED)

target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)
# target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)


target_link_libraries(test PRIVATE ntl gmp m pthread)

target_link_libraries(moai_gen_keys PRIVATE OpenMP::OpenMP_CXX ntl gmp m pthread)
target_compile_options(moai_gen_keys PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)
if(DEFINED ENV{CONDA_PREFIX})
  target_link_directories(moai_gen_keys PRIVATE "$ENV{CONDA_PREFIX}/lib")
endif()

# NTL headers: system install (/usr/local) or conda (e.g. $CONDA_PREFIX/include/NTL/...)
if(DEFINED ENV{CONDA_PREFIX})
  target_include_directories(test PRIVATE "$ENV{CONDA_PREFIX}/include")
  target_include_directories(moai_gen_keys PRIVATE "$ENV{CONDA_PREFIX}/include")
endif()

target_include_directories(test PRIVATE
  /usr/local/include
  /usr/local/include/NTL
  # src/bootstrapping/common
  # src/bootstrapping
  # scratch
  # scratch/bootstrapping/common
  # scratch/bootstrapping
  ${CMAKE_SOURCE_DIR}/src/include/source/bootstrapping/common
  ${CMAKE_SOURCE_DIR}/src/include/source/bootstrapping
  ${CMAKE_SOURCE_DIR}/src/include
)

target_include_directories(moai_gen_keys PRIVATE
  /usr/local/include
  /usr/local/include/NTL
)


file(COPY "${CMAKE_SOURCE_DIR}/src/data/" DESTINATION "${CMAKE_BINARY_DIR}/")


