# CMakeList.txt : CMake project for svmnral, include source and define
# project specific logic here.

cmake_minimum_required(VERSION 3.20)

project("svmnral")
SET(TARGETS svmnral)
SET(SOURCES "svmnral.cc" "nral.cc")
SET(HEADERS "svmnral.h" "nral.h" "common.h")
find_package(OpenMP REQUIRED)

set(BLA_VENDOR Intel10_64lp)
find_package(LAPACK REQUIRED)

find_package(BLAS)

if(BLAS_FOUND)
  message("MKL environment detected")
  add_definitions(-DUSE_MKL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 -fast")
  set(CUSTOM_MKL_LINKER_FLAGS "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")
else()
  unset(BLA_VENDOR)
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS REQUIRED)

  if(BLAS_FOUND)
    if(MSVC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /qopenmp /nologo /O3 /EHsc /I. /D _WIN64 /D _CRT_SECURE_NO_DEPRECATE")
    endif(MSVC)

    if(MINGW)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=0 -fopenmp")
    endif(MINGW)

  target_compile_options(${TARGETS} PRIVATE ${BLAS_LINKER_FLAGS} ${CUSTOM_MKL_LINKER_FLAGS})
  target_link_libraries(${TARGETS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    add_definitions(-DUSE_CBLAS)
    message("OpenBLAS found.")
  endif(BLAS_FOUND)
endif()



# Add source to this project's executable.
add_executable(${TARGETS} ${SOURCES} ${HEADERS})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Wextra ")
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
