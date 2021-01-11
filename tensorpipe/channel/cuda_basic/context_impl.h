/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <tensorpipe/channel/context_impl_boilerplate.h>
#include <tensorpipe/channel/cpu_context.h>
#include <tensorpipe/channel/cuda_context.h>
#include <tensorpipe/common/cuda.h>
#include <tensorpipe/common/cuda_buffer.h>
#include <tensorpipe/common/cuda_loop.h>
#include <tensorpipe/common/deferred_executor.h>

namespace tensorpipe {
namespace channel {
namespace cuda_basic {

class ChannelImpl;

class ContextImpl final
    : public ContextImplBoilerplate<CudaBuffer, ContextImpl, ChannelImpl> {
 public:
  explicit ContextImpl(std::shared_ptr<CpuContext> cpuContext);

  std::shared_ptr<CudaChannel> createChannel(
      std::shared_ptr<transport::Connection> connection,
      Endpoint endpoint);

  // Implement the DeferredExecutor interface.
  bool inLoop() override;
  void deferToLoop(std::function<void()> fn) override;

 protected:
  // Implement the entry points called by ContextImplBoilerplate.
  void closeImpl() override;
  void joinImpl() override;
  void setIdImpl() override;

 private:
  OnDemandDeferredExecutor loop_;

  const std::shared_ptr<CpuContext> cpuContext_;
  // TODO: Lazy initialization of cuda loop.
  CudaLoop cudaLoop_;
  CudaPinnedBufferAllocator cudaPinnedBufferAllocator_;
};

} // namespace cuda_basic
} // namespace channel
} // namespace tensorpipe
