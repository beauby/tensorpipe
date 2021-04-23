/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <future>
#include <memory>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <tensorpipe/common/cpu_buffer.h>
#include <tensorpipe/common/optional.h>
#include <tensorpipe/tensorpipe.h>
#include <tensorpipe/test/peer_group.h>

#if TENSORPIPE_SUPPORTS_CUDA
#include <tensorpipe/common/cuda.h>
#include <tensorpipe/common/cuda_buffer.h>
#endif // TENSORPIPE_SUPPORTS_CUDA

class Storage {
 public:
  std::vector<std::shared_ptr<void>> payloads;
  std::vector<std::pair<std::shared_ptr<void>, tensorpipe::Buffer>> tensors;
};

class InlineMessage {
 public:
  class Payload {
   public:
    std::string data;
    std::string metadata;
  };

  class Tensor {
   public:
    std::string data;
    std::string metadata;
    tensorpipe::optional<tensorpipe::Device> device;
    tensorpipe::optional<tensorpipe::Device> targetDevice;
  };

  std::vector<Payload> payloads;
  std::vector<Tensor> tensors;
  std::string metadata;
};

inline std::pair<tensorpipe::Message, Storage> makeMessage(
    InlineMessage imessage) {
  tensorpipe::Message message;
  Storage storage;

  for (auto& payload : imessage.payloads) {
    size_t length = payload.data.length();
    auto data = std::unique_ptr<uint8_t, std::default_delete<uint8_t[]>>(
        new uint8_t[length]);
    std::memcpy(data.get(), &payload.data[0], length);
    message.payloads.push_back({
        .data = data.get(),
        .length = length,
        .metadata = payload.metadata,
    });
    storage.payloads.push_back(std::move(data));
  }

  for (auto& tensor : imessage.tensors) {
    size_t length = tensor.data.length();
    if (!tensor.device || tensor.device->type == tensorpipe::kCpuDeviceType) {
      auto data = std::unique_ptr<uint8_t, std::default_delete<uint8_t[]>>(
          new uint8_t[length]);
      std::memcpy(data.get(), &tensor.data[0], length);
      tensorpipe::Buffer buffer = tensorpipe::CpuBuffer{.ptr = data.get()};
      message.tensors.push_back({
          .buffer = buffer,
          .length = length,
          .targetDevice = tensor.targetDevice,
          .metadata = tensor.metadata,
      });
      storage.tensors.push_back({std::move(data), std::move(buffer)});
#if TENSORPIPE_SUPPORTS_CUDA
    } else if (tensor.device->type == tensorpipe::kCudaDeviceType) {
      void* cudaPtr;
      TP_CUDA_CHECK(cudaSetDevice(tensor.device->index));
      TP_CUDA_CHECK(cudaMalloc(&cudaPtr, length));
      auto data = std::unique_ptr<void, std::function<void(void*)>>(
          cudaPtr, [](void* ptr) { TP_CUDA_CHECK(cudaFree(ptr)); });
      // TODO: Properly dispose of stream when done.
      cudaStream_t stream;
      TP_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      tensorpipe::Buffer buffer = tensorpipe::CudaBuffer{
          .ptr = data.get(),
          .stream = stream,
      };
      TP_CUDA_CHECK(cudaMemcpyAsync(
          cudaPtr, &tensor.data[0], length, cudaMemcpyDefault, stream));
      message.tensors.push_back({
          .buffer = buffer,
          .length = length,
          .targetDevice = tensor.targetDevice,
          .metadata = tensor.metadata,
      });
      storage.tensors.push_back({std::move(data), std::move(buffer)});
#endif // TENSORPIPE_SUPPORTS_CUDA
    } else {
      ADD_FAILURE() << "Unexpected source device: "
                    << tensor.device->toString();
    }
  }

  message.metadata = imessage.metadata;

  return {std::move(message), std::move(storage)};
}

inline std::pair<tensorpipe::Allocation, Storage> makeAllocation(
    const tensorpipe::Descriptor& descriptor,
    const std::vector<tensorpipe::Device>& devices) {
  tensorpipe::Allocation allocation;
  Storage storage;
  for (const auto& payload : descriptor.payloads) {
    auto data = std::unique_ptr<uint8_t, std::default_delete<uint8_t[]>>(
        new uint8_t[payload.length]);
    allocation.payloads.push_back({.data = data.get()});
    storage.payloads.push_back(std::move(data));
  }

  TP_DCHECK(devices.empty() || devices.size() == descriptor.tensors.size());
  for (size_t tensorIdx = 0; tensorIdx < descriptor.tensors.size();
       ++tensorIdx) {
    const auto& tensor = descriptor.tensors[tensorIdx];
    tensorpipe::Device targetDevice;
    if (!devices.empty()) {
      TP_DCHECK(
          !tensor.targetDevice.has_value() ||
          devices[tensorIdx] == *tensor.targetDevice);
      targetDevice = devices[tensorIdx];
    } else if (tensor.targetDevice) {
      targetDevice = *tensor.targetDevice;
    } else {
      // Default to CPU for unspecified target device.
      targetDevice = tensorpipe::Device{tensorpipe::kCpuDeviceType, 0};
    }

    // Default to receiving on CPU when target device not specified.
    if (targetDevice.type == tensorpipe::kCpuDeviceType) {
      auto data = std::unique_ptr<uint8_t, std::default_delete<uint8_t[]>>(
          new uint8_t[tensor.length]);
      tensorpipe::Buffer buffer = tensorpipe::CpuBuffer{.ptr = data.get()};
      allocation.tensors.push_back({.buffer = buffer});
      storage.tensors.push_back({std::move(data), std::move(buffer)});
#if TENSORPIPE_SUPPORTS_CUDA
    } else if (targetDevice.type == tensorpipe::kCudaDeviceType) {
      void* cudaPtr;
      TP_CUDA_CHECK(cudaSetDevice(targetDevice.index));
      TP_CUDA_CHECK(cudaMalloc(&cudaPtr, tensor.length));
      auto data = std::unique_ptr<void, std::function<void(void*)>>(
          cudaPtr, [](void* ptr) { TP_CUDA_CHECK(cudaFree(ptr)); });
      // TODO: Properly dispose of stream when done.
      cudaStream_t stream;
      TP_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      tensorpipe::Buffer buffer = tensorpipe::CudaBuffer{
          .ptr = data.get(),
          .stream = stream,
      };
      allocation.tensors.push_back({.buffer = buffer});
      storage.tensors.push_back({std::move(data), std::move(buffer)});
#endif // TENSORPIPE_SUPPORTS_CUDA
    } else {
      ADD_FAILURE() << "Unexpected target device: " << targetDevice.toString();
    }
  }

  return {std::move(allocation), std::move(storage)};
}

inline std::future<void> pipeWriteWithFuture(
    tensorpipe::Pipe& pipe,
    tensorpipe::Message message) {
  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();

  pipe.write(
      std::move(message),
      [promise{std::move(promise)}](const tensorpipe::Error& error) {
        if (error) {
          promise->set_exception(
              std::make_exception_ptr(std::runtime_error(error.what())));
          return;
        }

        promise->set_value();
      });

  return future;
}

inline std::future<std::tuple<tensorpipe::Descriptor, Storage>>
pipeReadWithFuture(
    tensorpipe::Pipe& pipe,
    std::vector<tensorpipe::Device> devices = {}) {
  auto promise = std::make_shared<
      std::promise<std::tuple<tensorpipe::Descriptor, Storage>>>();
  auto future = promise->get_future();
  pipe.readDescriptor([&pipe,
                       promise{std::move(promise)},
                       devices{std::move(devices)}](
                          const tensorpipe::Error& error,
                          tensorpipe::Descriptor descriptor) mutable {
    if (error) {
      promise->set_exception(
          std::make_exception_ptr(std::runtime_error(error.what())));
      return;
    }

    tensorpipe::Allocation allocation;
    Storage storage;
    std::tie(allocation, storage) = makeAllocation(descriptor, devices);
    pipe.read(
        std::move(allocation),
        [promise{std::move(promise)},
         descriptor{std::move(descriptor)},
         storage{std::move(storage)}](const tensorpipe::Error& error) mutable {
          if (error) {
            promise->set_exception(
                std::make_exception_ptr(std::runtime_error(error.what())));
            return;
          }

          promise->set_value(std::make_tuple<tensorpipe::Descriptor, Storage>(
              std::move(descriptor), std::move(storage)));
        });
  });

  return future;
}

inline void expectDescriptorAndStorageMatchMessage(
    tensorpipe::Descriptor descriptor,
    Storage storage,
    InlineMessage imessage) {
  EXPECT_EQ(imessage.metadata, descriptor.metadata);

  EXPECT_EQ(descriptor.payloads.size(), storage.payloads.size());
  EXPECT_EQ(imessage.payloads.size(), storage.payloads.size());
  for (size_t idx = 0; idx < imessage.payloads.size(); ++idx) {
    EXPECT_EQ(
        imessage.payloads[idx].metadata, descriptor.payloads[idx].metadata);
    EXPECT_EQ(
        imessage.payloads[idx].data,
        std::string(
            static_cast<char*>(storage.payloads[idx].get()),
            descriptor.payloads[idx].length));
  }

  EXPECT_EQ(descriptor.tensors.size(), storage.tensors.size());
  EXPECT_EQ(imessage.tensors.size(), storage.tensors.size());
  for (size_t idx = 0; idx < imessage.tensors.size(); ++idx) {
    EXPECT_TRUE(
        !imessage.tensors[idx].device ||
        *imessage.tensors[idx].device == descriptor.tensors[idx].sourceDevice);
    EXPECT_EQ(imessage.tensors[idx].metadata, descriptor.tensors[idx].metadata);
    EXPECT_EQ(
        imessage.tensors[idx].targetDevice,
        descriptor.tensors[idx].targetDevice);
    const tensorpipe::Device& device = storage.tensors[idx].second.device();
    EXPECT_TRUE(
        !imessage.tensors[idx].targetDevice ||
        imessage.tensors[idx].targetDevice == device);
    if (device.type == tensorpipe::kCpuDeviceType) {
      const tensorpipe::CpuBuffer& buffer =
          storage.tensors[idx].second.unwrap<tensorpipe::CpuBuffer>();
      EXPECT_EQ(
          imessage.tensors[idx].data,
          std::string(
              static_cast<char*>(buffer.ptr), descriptor.tensors[idx].length));
#if TENSORPIPE_SUPPORTS_CUDA
    } else if (device.type == tensorpipe::kCudaDeviceType) {
      const tensorpipe::CudaBuffer& buffer =
          storage.tensors[idx].second.unwrap<tensorpipe::CudaBuffer>();
      size_t length = descriptor.tensors[idx].length;
      std::string data(length, 0x0);
      TP_CUDA_CHECK(cudaStreamSynchronize(buffer.stream));
      TP_CUDA_CHECK(
          cudaMemcpy(&data[0], buffer.ptr, length, cudaMemcpyDefault));
      EXPECT_EQ(imessage.tensors[idx].data, data.data());
#endif // TENSORPIPE_SUPPORTS_CUDA
    } else {
      ADD_FAILURE() << "Unexpected target device: " << device.toString();
    }
  }
}

inline std::string createUniqueShmAddr() {
  const ::testing::TestInfo* const testInfo =
      ::testing::UnitTest::GetInstance()->current_test_info();
  std::ostringstream ss;
  // Once we upgrade googletest, also use test_info->test_suite_name() here.
  ss << "shm://tensorpipe_test_" << testInfo->name() << "_" << getpid();
  return ss.str();
}

inline std::vector<std::string> genUrls() {
  std::vector<std::string> res;

#if TENSORPIPE_HAS_SHM_TRANSPORT
  res.push_back(createUniqueShmAddr());
#endif // TENSORPIPE_HAS_SHM_TRANSPORT
  res.push_back("uv://127.0.0.1");

  return res;
}

inline std::shared_ptr<tensorpipe::Context> makeContext() {
  auto context = std::make_shared<tensorpipe::Context>();

  context->registerTransport(0, "uv", tensorpipe::transport::uv::create());
#if TENSORPIPE_HAS_SHM_TRANSPORT
  context->registerTransport(1, "shm", tensorpipe::transport::shm::create());
#endif // TENSORPIPE_HAS_SHM_TRANSPORT
  context->registerChannel(0, "basic", tensorpipe::channel::basic::create());
#if TENSORPIPE_HAS_CMA_CHANNEL
  context->registerChannel(1, "cma", tensorpipe::channel::cma::create());
#endif // TENSORPIPE_HAS_CMA_CHANNEL
#if TENSORPIPE_SUPPORTS_CUDA
  context->registerChannel(
      10,
      "cuda_basic",
      tensorpipe::channel::cuda_basic::create(
          tensorpipe::channel::basic::create()));
#if TENSORPIPE_HAS_CUDA_IPC_CHANNEL
  context->registerChannel(
      11, "cuda_ipc", tensorpipe::channel::cuda_ipc::create());
#endif // TENSORPIPE_HAS_CUDA_IPC_CHANNEL
  context->registerChannel(
      12, "cuda_xth", tensorpipe::channel::cuda_xth::create());
#endif // TENSORPIPE_SUPPORTS_CUDA

  return context;
}

class ClientServerPipeTestCase {
  ForkedThreadPeerGroup pg_;

 public:
  void run() {
    pg_.spawn(
        [&]() {
          auto context = makeContext();

          auto listener = context->listen(genUrls());
          pg_.send(PeerGroup::kClient, listener->url("uv"));

          std::promise<std::shared_ptr<tensorpipe::Pipe>> promise;
          listener->accept([&](const tensorpipe::Error& error,
                               std::shared_ptr<tensorpipe::Pipe> pipe) {
            if (error) {
              promise.set_exception(
                  std::make_exception_ptr(std::runtime_error(error.what())));
            } else {
              promise.set_value(std::move(pipe));
            }
          });

          std::shared_ptr<tensorpipe::Pipe> pipe = promise.get_future().get();
          server(*pipe);

          pg_.done(PeerGroup::kServer);
          pg_.join(PeerGroup::kServer);

          context->join();
        },
        [&]() {
          auto context = makeContext();

          auto url = pg_.recv(PeerGroup::kClient);
          auto pipe = context->connect(url);

          client(*pipe);

          pg_.done(PeerGroup::kClient);
          pg_.join(PeerGroup::kClient);

          context->join();
        });
  }

  virtual void client(tensorpipe::Pipe& pipe) = 0;
  virtual void server(tensorpipe::Pipe& pipe) = 0;

  virtual ~ClientServerPipeTestCase() = default;
};
