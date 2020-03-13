/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <tensorpipe/transport/context.h>
#include <tensorpipe/transport/uv/loop.h>

namespace tensorpipe {
namespace transport {
namespace uv {

class Context final : public transport::Context {
 public:
  explicit Context();

  ~Context();

  void join() override;

  std::shared_ptr<transport::Connection> connect(address_t addr) override;

  std::shared_ptr<transport::Listener> listen(address_t addr) override;

  const std::string& domainDescriptor() const override;

 private:
  std::shared_ptr<Loop> loop_;
  std::string domainDescriptor_;
};

}  // namespace uv
}  // namespace transport
}  // namespace tensorpipe
