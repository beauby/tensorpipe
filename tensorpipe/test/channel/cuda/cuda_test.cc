/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <numeric>

#include <cuda_runtime.h>

#include <tensorpipe/channel/cuda/context.h>
#include <tensorpipe/test/channel/channel_test.h>

namespace {

class CudaChannelTest : public ProcessChannelTest {};

class CudaChannelTestHelper : public ChannelTestHelper {
 public:
  std::shared_ptr<tensorpipe::channel::Context> makeContext(
      std::string id) override {
    auto context = std::make_shared<tensorpipe::channel::cuda::Context>();
    context->setId(std::move(id));
    return context;
  }
};

CudaChannelTestHelper helper;

} // namespace

using namespace tensorpipe;
using namespace tensorpipe::channel;

TEST_P(CudaChannelTest, ClientToServer) {
  constexpr auto dataSize = 256;

  // Initialize with sequential values.
  std::array<uint8_t, dataSize> data;
  std::iota(data.begin(), data.end(), 0);

  testConnectionPair(
      [&](std::shared_ptr<transport::Connection> conn,
          TSendStr sendStr,
          TRecvStr recvStr) {},
      [&](std::shared_ptr<transport::Connection> conn,
          TSendStr sendStr,
          TRecvStr recvStr) {});
}

INSTANTIATE_TEST_CASE_P(Cuda, CudaChannelTest, ::testing::Values(&helper));

CHANNEL_TEST(ChannelTest, FooBar) {}

CHANNEL_TEST_server(ChannelTest, Foobar) {}

CHANNEL_TEST

class FooTest : public ChannelTest {
 private:
  constexpr auto dataSize = 256;
  std::array<uint8_t, dataSize> data_;

 public:
  void setup() {
    std::iota(data_.begin(), data_.end());
  }

  void server() {
    EXPECT_EQ(cudaSuccess, cudaSetDevice(0));

    std::shared_ptr<Context> serverCtx = makeContext("server");
    auto channel =
        serverCtx->createChannel(std::move(conn), Channel::Endpoint::kListen);

    // Copy data to device.
    void* gpuData;
    EXPECT_EQ(cudaSuccess, cudaMalloc(&gpuData, sizeof(data)));
    EXPECT_EQ(
        cudaSuccess,
        cudaMemcpy(gpuData, data.data(), sizeof(data), cudaMemcpyHostToDevice));

    // Perform send and wait for completion.
    std::future<std::tuple<Error, Channel::TDescriptor>> descriptorFuture;
    std::future<Error> sendFuture;
    std::tie(descriptorFuture, sendFuture) =
        sendWithFuture(channel, gpuData, data.size());
    Error descriptorError;
    Channel::TDescriptor descriptor;
    std::tie(descriptorError, descriptor) = descriptorFuture.get();
    EXPECT_FALSE(descriptorError) << descriptorError.what();

    // Send descriptor over to connecting side.
    sendStr(descriptor);

    Error sendError = sendFuture.get();
    EXPECT_FALSE(sendError) << sendError.what();

    EXPECT_EQ(cudaSuccess, cudaFree(gpuData));

    serverCtx->join();
  }

  void client() {
    EXPECT_EQ(cudaSuccess, cudaSetDevice(0));

    std::shared_ptr<Context> clientCtx = makeContext("client");
    auto channel =
        clientCtx->createChannel(std::move(conn), Channel::Endpoint::kConnect);

    // Initialize with zeroes.
    void* gpuData;
    EXPECT_EQ(cudaSuccess, cudaMalloc(&gpuData, dataSize));

    // Receive descriptor from listening side.
    Channel::TDescriptor descriptor = recvStr();

    // Perform recv and wait for completion.
    std::future<Error> recvFuture =
        recvWithFuture(channel, descriptor, gpuData, dataSize);
    Error recvError = recvFuture.get();
    EXPECT_FALSE(recvError) << recvError.what();

    // Copy received data to host.
    std::array<uint8_t, dataSize> recvData;
    std::fill(recvData.begin(), recvData.end(), 0);
    EXPECT_EQ(
        cudaSuccess,
        cudaMemcpy(recvData.data(), gpuData, dataSize, cudaMemcpyDeviceToHost));

    // Validate contents of vector.
    for (auto i = 0; i < dataSize; i++) {
      EXPECT_EQ(recvData[i], i);
    }

    EXPECT_EQ(cudaSuccess, cudaFree(gpuData));

    clientCtx->join();
  }
};

CHANNEL_TEST(ChannelTest, FooTest);

TEST_P(ChannelTest, FooTest) {
  FooTest t(std::get<0>(GetParam()), std::get<1>(GetParam()));
  t.run();
}
