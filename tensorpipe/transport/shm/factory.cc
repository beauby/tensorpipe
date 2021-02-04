/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/transport/shm/factory.h>

#include <tensorpipe/transport/context_boilerplate.h>
#include <tensorpipe/transport/shm/connection_impl.h>
#include <tensorpipe/transport/shm/context_impl.h>
#include <tensorpipe/transport/shm/listener_impl.h>

namespace tensorpipe {
namespace transport {
namespace shm {

std::shared_ptr<Context> create() {
  return std::make_shared<
      ContextBoilerplate<ContextImpl, ListenerImpl, ConnectionImpl>>(
      ContextImpl::create());
}

} // namespace shm
} // namespace transport
} // namespace tensorpipe