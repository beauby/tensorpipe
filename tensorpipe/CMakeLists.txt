# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(tensorpipe)

# Support `#include <tensorpipe/foo.h>`.
target_include_directories(tensorpipe
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set(TENSORPIPE_PUBLIC_HEADERS
  transport/connection.h
  transport/context.h
  transport/error.h
  transport/defs.h
  transport/listener.h
  transport/uv/context.h
  core/error.h
  core/listener.h
  core/context.h
  core/message.h
  core/pipe.h
  channel/error.h
  channel/channel.h
  channel/context.h
  channel/helpers.h
  channel/basic/context.h
  common/error_macros.h
  common/error.h
  common/defs.h
  common/address.h
  common/optional.h
  common/system.h
  common/callback.h
  common/queue.h
  tensorpipe.h)

target_sources(tensorpipe PRIVATE
  channel/context.cc
  channel/error.cc
  common/address.cc
  common/error.cc
  common/system.cc
  core/context.cc
  core/error.cc
  core/listener.cc
  core/pipe.cc
  proto/core.proto
  transport/connection.cc
  transport/error.cc
  )


## Channels

### basic

target_sources(tensorpipe PRIVATE
  channel/basic/channel.cc
  channel/basic/context.cc
  proto/channel/basic.proto)

### cma

if(TP_ENABLE_CMA)
  target_sources(tensorpipe PRIVATE
    channel/cma/channel.cc
    channel/cma/context.cc
    proto/channel/cma.proto)
  list(APPEND TENSORPIPE_PUBLIC_HEADERS channel/cma/context.h)
  target_compile_definitions(tensorpipe INTERFACE TP_ENABLE_CMA)
endif()


## Transports

### uv

target_sources(tensorpipe PRIVATE
  transport/uv/connection.cc
  transport/uv/context.cc
  transport/uv/error.cc
  transport/uv/listener.cc
  transport/uv/loop.cc
  transport/uv/sockaddr.cc
  transport/uv/uv.cc
  )

find_package(uv REQUIRED)
target_link_libraries(tensorpipe PRIVATE uv::uv)

### shm

if(TP_ENABLE_SHM)
  target_sources(tensorpipe PRIVATE
    transport/shm/context.cc
    transport/shm/connection.cc
    transport/shm/fd.cc
    transport/shm/listener.cc
    transport/shm/loop.cc
    transport/shm/reactor.cc
    transport/shm/socket.cc
    util/ringbuffer/shm.cc
    util/shm/segment.cc
    )
  list(APPEND TENSORPIPE_PUBLIC_HEADERS transport/shm/context.h)
  target_compile_definitions(tensorpipe INTERFACE TP_ENABLE_SHM)
endif()


## Protobufs

find_package(Protobuf 3 REQUIRED
  HINTS ${TP_PROTOBUF_HINTS})
target_link_libraries(tensorpipe PRIVATE protobuf::libprotobuf)

# Support `#include <tensorpipe/proto/foo.pb.h>`.
target_include_directories(tensorpipe PRIVATE ${PROJECT_BINARY_DIR})

# Support `#include "proto/foo.pb.h"`, as generate by protoc.
target_include_directories(tensorpipe PRIVATE ${PROJECT_BINARY_DIR}/tensorpipe)

protobuf_generate(TARGET tensorpipe)


## Python bindings

if(TP_BUILD_PYTHON)
  add_subdirectory(python)
endif()


## Tests

if(BUILD_TESTING)
  add_subdirectory(test)
endif()


## Installing

include(GNUInstallDirs)
foreach (file ${TENSORPIPE_PUBLIC_HEADERS})
  get_filename_component(dir ${file} DIRECTORY)
  install(FILES ${file} DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tensorpipe/${dir})
endforeach()
install(FILES ${PROJECT_SOURCE_DIR}/cmake/TensorpipeConfig.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME})
install(TARGETS tensorpipe
  EXPORT tensorpipe-targets
  LIBRARY       DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
  ARCHIVE       DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
  )
install(EXPORT tensorpipe-targets
  FILE        TensorpipeTargets.cmake
  NAMESPACE   tensorpipe::
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
  )
