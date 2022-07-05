# CMakeList.txt : CMake project for svmfpgm, include source and define
# project specific logic here.

cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_C_COMPILER "icx")
SET(CMAKE_CXX_COMPILER "icpx")
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

project("svmfpgm")
SET(TARGETS svmfpgm)
SET(SOURCES "svmfpgm.cc" "fista.cc")
SET(HEADERS "svmfpgm.h" "fista.h" "common.h")
find_package(OpenMP REQUIRED)

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)

# Add source to this project's executable.
add_executable(${TARGETS} ${SOURCES} ${HEADERS})

if(BLAS_FOUND)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc")
  message("MKL environment detected")
  find_package(IntelDPCPP REQUIRED)
  find_package(MKL REQUIRED)
  include_directories(${MKL_INCLUDE_DIRS})
  link_directories(${MKL_LIBRARIES})

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Wextra -fast -fmessage-length=0 -fopenmp")
  add_definitions(-DUSE_MKL -DMKL_ILP64)

  target_link_libraries(${TARGETS} mkl_intel_lp64
    mkl_sequential
    mkl_core
  )

else()
  unset(BLA_VENDOR)
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS REQUIRED)

  if(BLAS_FOUND)
    if(MSVC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /qopenmp /nologo /O3 /EHsc /I. /D _WIN64 /D _CRT_SECURE_NO_DEPRECATE")
    endif(MSVC)

    if(MINGW)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Wextra -O3 -fmessage-length=0 -fopenmp")
    endif(MINGW)

    message("OpenBLAS found.")

    target_compile_options(${TARGETS} PRIVATE ${BLAS_LINKER_FLAGS})
    target_link_libraries(${TARGETS} ${BLAS_LIBRARIES})
  endif(BLAS_FOUND)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
