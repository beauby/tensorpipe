# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(tensorpipe-benchmarks LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../../cmake")

# set(protobuf_BUILD_TESTS OFF)
# add_subdirectory(protobuf/cmake)

include(ExternalProject)
ExternalProject_Add(cmake-3.17.1
  PREFIX cmake-3.17.1
  URL https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1-${CMAKE_SYSTEM_NAME}-x86_64.tar.gz
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND cp -R <SOURCE_DIR> <INSTALL_DIR>
  DOWNLOAD_NO_PROGRESS ON)
ExternalProject_Add(tensorpipe-build
  PREFIX "tensorpipe"
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/../../../tensorpipe2
  DEPENDS cmake-3.17.1
  CMAKE_COMMAND ${PROJECT_BINARY_DIR}/cmake-3.17.1/cmake-3.17.1$<$<PLATFORM_ID:Darwin>:/CMake.app/Contents>/bin/cmake
  CMAKE_ARGS -DTP_BUILD_PYTHON=OFF -DBUILD_TESTING=OFF -DTP_BUILD_LIBUV=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  EXCLUDE_FROM_ALL 1)

ExternalProject_Get_Property(tensorpipe-build INSTALL_DIR)
#set(tensorpipe_DIR ${INSTALL_DIR}/lib/cmake/tensorpipe)
#find_package(tensorpipe REQUIRED CONFIG)
#file(MAKE_DIRECTORY ${INSTALL_DIR}/lib)
file(MAKE_DIRECTORY ${INSTALL_DIR}/include)

find_package(Protobuf 3 REQUIRED)
find_package(uv REQUIRED)

add_library(tensorpipe INTERFACE IMPORTED GLOBAL)
add_dependencies(tensorpipe tensorpipe-build)
target_link_libraries(tensorpipe INTERFACE "${INSTALL_DIR}/lib/libtensorpipe.a" protobuf::libprotobuf uv::uv)
target_include_directories(tensorpipe INTERFACE "${INSTALL_DIR}/include")

add_executable(benchmark_transport benchmark_transport.cc options.cc)
target_link_libraries(benchmark_transport
  PRIVATE
  tensorpipe)
  # protobuf::libprotobuf-lite)

add_executable(benchmark_pipe benchmark_pipe.cc options.cc)
target_link_libraries(benchmark_pipe
  PRIVATE
  tensorpipe)
# protobuf::libprotobuf-lite)
