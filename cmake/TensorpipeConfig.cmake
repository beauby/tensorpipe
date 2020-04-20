# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

get_filename_component(Tensorpipe_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

if(NOT TARGET tensorpipe::libtensorpipe)
    include("${Tensorpipe_CMAKE_DIR}/TensorpipeTargets.cmake")
endif()