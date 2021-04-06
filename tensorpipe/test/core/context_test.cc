/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/tensorpipe.h>

#include <cstring>
#include <exception>
#include <future>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#if TENSORPIPE_SUPPORTS_CUDA
#include <tensorpipe/common/cuda.h>
#endif // TENSORPIPE_SUPPORTS_CUDA

#include <tensorpipe/test/peer_group.h>

using namespace tensorpipe;

namespace {

::testing::AssertionResult buffersAreEqual(
    const void* ptr1,
    const size_t len1,
    const void* ptr2,
    const size_t len2) {
  if (ptr1 == nullptr && ptr2 == nullptr) {
    if (len1 == 0 && len2 == 0) {
      return ::testing::AssertionSuccess();
    }
    if (len1 != 0) {
      return ::testing::AssertionFailure()
          << "first pointer is null but length isn't 0";
    }
    if (len1 != 0) {
      return ::testing::AssertionFailure()
          << "second pointer is null but length isn't 0";
    }
  }
  if (ptr1 == nullptr) {
    return ::testing::AssertionFailure()
        << "first pointer is null but second one isn't";
  }
  if (ptr2 == nullptr) {
    return ::testing::AssertionFailure()
        << "second pointer is null but first one isn't";
  }
  if (len1 != len2) {
    return ::testing::AssertionFailure()
        << "first length is " << len1 << " but second one is " << len2;
  }
  if (std::memcmp(ptr1, ptr2, len1) != 0) {
    return ::testing::AssertionFailure() << "buffer contents aren't equal";
  }
  return ::testing::AssertionSuccess();
}

#if TENSORPIPE_SUPPORTS_CUDA
std::vector<uint8_t> unwrapCudaBuffer(CudaBuffer b, size_t length) {
  std::vector<uint8_t> result(length);
  TP_CUDA_CHECK(cudaStreamSynchronize(b.stream));
  TP_CUDA_CHECK(cudaMemcpy(result.data(), b.ptr, length, cudaMemcpyDefault));

  return result;
}
#endif // TENSORPIPE_SUPPORTS_CUDA

::testing::AssertionResult messagesAreEqual(
    const Message& m1,
    const Message& m2) {
  if (m1.payloads.size() != m2.payloads.size()) {
    return ::testing::AssertionFailure()
        << "first message has " << m1.payloads.size()
        << " payloads but second has " << m2.payloads.size();
  }
  for (size_t idx = 0; idx < m1.payloads.size(); idx++) {
    EXPECT_TRUE(buffersAreEqual(
        m1.payloads[idx].data,
        m1.payloads[idx].length,
        m2.payloads[idx].data,
        m2.payloads[idx].length));
  }
  if (m1.tensors.size() != m2.tensors.size()) {
    return ::testing::AssertionFailure()
        << "first message has " << m1.tensors.size()
        << " tensors but second has " << m2.tensors.size();
  }
  for (size_t idx = 0; idx < m1.tensors.size(); idx++) {
    EXPECT_EQ(m1.tensors[idx].buffer.device(), m2.tensors[idx].buffer.device());
    const std::string& deviceType = m1.tensors[idx].buffer.device().type;

    if (deviceType == kCpuDeviceType) {
      EXPECT_TRUE(buffersAreEqual(
          m1.tensors[idx].buffer.unwrap<CpuBuffer>().ptr,
          m1.tensors[idx].length,
          m2.tensors[idx].buffer.unwrap<CpuBuffer>().ptr,
          m2.tensors[idx].length));
#if TENSORPIPE_SUPPORTS_CUDA
    } else if (deviceType == kCudaDeviceType) {
      std::vector<uint8_t> buffer1 = unwrapCudaBuffer(
          m1.tensors[idx].buffer.unwrap<CudaBuffer>(), m1.tensors[idx].length);
      std::vector<uint8_t> buffer2 = unwrapCudaBuffer(
          m2.tensors[idx].buffer.unwrap<CudaBuffer>(), m2.tensors[idx].length);
      EXPECT_TRUE(buffersAreEqual(
          buffer1.data(), buffer1.size(), buffer2.data(), buffer2.size()));
#endif // TENSORPIPE_SUPPORTS_CUDA
    } else {
      ADD_FAILURE() << "Unexpected device type: " << deviceType;
    }
  }
  return ::testing::AssertionSuccess();
}

#if TENSORPIPE_SUPPORTS_CUDA
struct CudaPointerDeleter {
  void operator()(void* ptr) {
    TP_CUDA_CHECK(cudaFree(ptr));
  }
};

std::unique_ptr<void, CudaPointerDeleter> makeCudaPointer(size_t length) {
  void* cudaPtr;
  TP_CUDA_CHECK(cudaMalloc(&cudaPtr, length));
  return std::unique_ptr<void, CudaPointerDeleter>(cudaPtr);
}
#endif // TENSORPIPE_SUPPORTS_CUDA

// Having 4 payloads per message is arbitrary.
constexpr int kNumPayloads = 4;
// Having 4 tensors per message ensures there are 2 CPU tensors and 2 CUDA
// tensors.
constexpr int kNumTensors = 4;
std::string kPayloadData = "I'm a payload";
std::string kTensorData = "And I'm a tensor";
#if TENSORPIPE_SUPPORTS_CUDA
const int kCudaTensorLength = 32;
const uint8_t kCudaTensorFillValue = 0x42;
#endif // TENSORPIPE_SUPPORTS_CUDA

Message::Tensor makeTensor(int index) {
#if TENSORPIPE_SUPPORTS_CUDA
  static std::unique_ptr<void, CudaPointerDeleter> kCudaTensorData = []() {
    auto cudaPtr = makeCudaPointer(kCudaTensorLength);
    TP_CUDA_CHECK(
        cudaMemset(cudaPtr.get(), kCudaTensorFillValue, kCudaTensorLength));
    return cudaPtr;
  }();

  if (index % 2 == 1) {
    return {
        .buffer =
            CudaBuffer{
                .ptr = kCudaTensorData.get(),
                .stream = cudaStreamDefault,
            },
        // FIXME: Use non-blocking stream.
        .length = kCudaTensorLength,
    };
  }
#endif // TENSORPIPE_SUPPORTS_CUDA

  return {
      .buffer =
          CpuBuffer{
              .ptr = reinterpret_cast<void*>(
                  const_cast<char*>(kTensorData.data())),
          },
      .length = kTensorData.length(),
  };
}

Message makeMessage(int numPayloads, int numTensors) {
  Message message;
  for (int i = 0; i < numPayloads; i++) {
    Message::Payload payload;
    payload.data =
        reinterpret_cast<void*>(const_cast<char*>(kPayloadData.data()));
    payload.length = kPayloadData.length();
    message.payloads.push_back(std::move(payload));
  }
  for (int i = 0; i < numTensors; i++) {
    message.tensors.push_back(makeTensor(i));
  }
  return message;
}

void allocateMessage(
    Message& message,
    std::vector<std::shared_ptr<void>>& buffers) {
  for (auto& payload : message.payloads) {
    // FIXME: Changing this to a make_shared causes havoc.
    auto payloadData = std::unique_ptr<uint8_t, std::default_delete<uint8_t[]>>(
        new uint8_t[payload.length]);
    payload.data = payloadData.get();
    buffers.push_back(std::move(payloadData));
  }
  for (auto& tensor : message.tensors) {
    // FIXME: Until the Pipe provides the `sourceDevice` directly to
    // `readDescriptor()`, we need to rely on `Buffer::deviceType()` rather than
    // `Buffer::device().type` since at this stage the buffer is not allocated,
    // hence `Buffer::device()` would call `cuPointerGetAttribute()` on
    // `nullptr`.
    if (tensor.buffer.deviceType() == DeviceType::kCpu) {
      auto tensorData =
          std::unique_ptr<uint8_t, std::default_delete<uint8_t[]>>(
              new uint8_t[tensor.length]);
      tensor.buffer.unwrap<CpuBuffer>().ptr = tensorData.get();
      buffers.push_back(std::move(tensorData));
#if TENSORPIPE_SUPPORTS_CUDA
    } else if (tensor.buffer.deviceType() == DeviceType::kCuda) {
      auto tensorData = makeCudaPointer(tensor.length);
      tensor.buffer.unwrap<CudaBuffer>().ptr = tensorData.get();
      // FIXME: Use non-blocking streams.
      tensor.buffer.unwrap<CudaBuffer>().stream = cudaStreamDefault;
      buffers.push_back(std::move(tensorData));
#endif // TENSORPIPE_SUPPORTS_CUDA
    } else {
      ADD_FAILURE() << "Unrecognized device type: "
                    << tensor.buffer.device().type;
    }
  }
}

std::string createUniqueShmAddr() {
  const ::testing::TestInfo* const testInfo =
      ::testing::UnitTest::GetInstance()->current_test_info();
  std::ostringstream ss;
  // Once we upgrade googletest, also use test_info->test_suite_name() here.
  ss << "shm://tensorpipe_test_" << testInfo->name() << "_" << getpid();
  return ss.str();
}

std::vector<std::string> genUrls() {
  std::vector<std::string> res;

#if TENSORPIPE_HAS_SHM_TRANSPORT
  res.push_back(createUniqueShmAddr());
#endif // TENSORPIPE_HAS_SHM_TRANSPORT
  res.push_back("uv://127.0.0.1");

  return res;
}

std::shared_ptr<Context> makeContext() {
  auto context = std::make_shared<Context>();

  context->registerTransport(0, "uv", transport::uv::create());
#if TENSORPIPE_HAS_SHM_TRANSPORT
  context->registerTransport(1, "shm", transport::shm::create());
#endif // TENSORPIPE_HAS_SHM_TRANSPORT
  context->registerChannel(0, "basic", channel::basic::create());
#if TENSORPIPE_HAS_CMA_CHANNEL
  context->registerChannel(1, "cma", channel::cma::create());
#endif // TENSORPIPE_HAS_CMA_CHANNEL
#if TENSORPIPE_SUPPORTS_CUDA
  context->registerChannel(
      10, "cuda_basic", channel::cuda_basic::create(channel::basic::create()));
#if TENSORPIPE_HAS_CUDA_IPC_CHANNEL
  context->registerChannel(11, "cuda_ipc", channel::cuda_ipc::create());
#endif // TENSORPIPE_HAS_CUDA_IPC_CHANNEL
  context->registerChannel(12, "cuda_xth", channel::cuda_xth::create());
#endif // TENSORPIPE_SUPPORTS_CUDA

  return context;
}

} // namespace

TEST(Context, ClientPingSerial) {
  ForkedThreadPeerGroup pg;
  pg.spawn(
      [&]() {
        std::vector<std::shared_ptr<void>> buffers;
        std::promise<std::shared_ptr<Pipe>> serverPipePromise;
        std::promise<Message> readDescriptorPromise;
        std::promise<Message> readMessagePromise;

        auto context = makeContext();

        auto listener = context->listen(genUrls());
        pg.send(PeerGroup::kClient, listener->url("uv"));

        listener->accept([&](const Error& error, std::shared_ptr<Pipe> pipe) {
          if (error) {
            serverPipePromise.set_exception(
                std::make_exception_ptr(std::runtime_error(error.what())));
          } else {
            serverPipePromise.set_value(std::move(pipe));
          }
        });
        std::shared_ptr<Pipe> serverPipe = serverPipePromise.get_future().get();

        serverPipe->readDescriptor(
            [&readDescriptorPromise](const Error& error, Message message) {
              if (error) {
                readDescriptorPromise.set_exception(
                    std::make_exception_ptr(std::runtime_error(error.what())));
              } else {
                readDescriptorPromise.set_value(std::move(message));
              }
            });

        Message message(readDescriptorPromise.get_future().get());
        allocateMessage(message, buffers);
        serverPipe->read(
            std::move(message),
            [&readMessagePromise](const Error& error, Message message) {
              if (error) {
                readMessagePromise.set_exception(
                    std::make_exception_ptr(std::runtime_error(error.what())));
              } else {
                readMessagePromise.set_value(std::move(message));
              }
            });
        EXPECT_TRUE(messagesAreEqual(
            readMessagePromise.get_future().get(),
            makeMessage(kNumPayloads, kNumTensors)));

        pg.done(PeerGroup::kServer);
        pg.join(PeerGroup::kServer);

        context->join();
      },
      [&]() {
        std::promise<Message> writtenMessagePromise;

        auto context = makeContext();

        auto url = pg.recv(PeerGroup::kClient);
        auto clientPipe = context->connect(url);

        clientPipe->write(
            makeMessage(kNumPayloads, kNumTensors),
            [&writtenMessagePromise](const Error& error, Message message) {
              if (error) {
                writtenMessagePromise.set_exception(
                    std::make_exception_ptr(std::runtime_error(error.what())));
              } else {
                writtenMessagePromise.set_value(std::move(message));
              }
            });
        EXPECT_TRUE(messagesAreEqual(
            writtenMessagePromise.get_future().get(),
            makeMessage(kNumPayloads, kNumTensors)));

        pg.done(PeerGroup::kClient);
        pg.join(PeerGroup::kClient);

        context->join();
      });
}

TEST(Context, ClientPingInline) {
  ForkedThreadPeerGroup pg;
  pg.spawn(
      [&]() {
        std::vector<std::shared_ptr<void>> buffers;
        std::promise<std::shared_ptr<Pipe>> serverPipePromise;
        std::promise<void> readCompletedProm;

        auto context = makeContext();

        auto listener = context->listen(genUrls());
        pg.send(PeerGroup::kClient, listener->url("uv"));

        listener->accept([&](const Error& error, std::shared_ptr<Pipe> pipe) {
          if (error) {
            serverPipePromise.set_exception(
                std::make_exception_ptr(std::runtime_error(error.what())));
          } else {
            serverPipePromise.set_value(std::move(pipe));
          }
        });
        std::shared_ptr<Pipe> serverPipe = serverPipePromise.get_future().get();

        serverPipe->readDescriptor([&serverPipe, &readCompletedProm, &buffers](
                                       const Error& error, Message message) {
          if (error) {
            ADD_FAILURE() << error.what();
            readCompletedProm.set_value();
            return;
          }
          allocateMessage(message, buffers);
          serverPipe->read(
              std::move(message),
              [&readCompletedProm](
                  const Error& error, Message message) mutable {
                if (error) {
                  ADD_FAILURE() << error.what();
                  readCompletedProm.set_value();
                  return;
                }
                EXPECT_TRUE(messagesAreEqual(
                    message, makeMessage(kNumPayloads, kNumTensors)));
                readCompletedProm.set_value();
              });
        });
        readCompletedProm.get_future().get();

        pg.done(PeerGroup::kServer);
        pg.join(PeerGroup::kServer);

        context->join();
      },
      [&]() {
        std::promise<void> writeCompletedProm;

        auto context = makeContext();

        auto url = pg.recv(PeerGroup::kClient);
        auto clientPipe = context->connect(url);

        clientPipe->write(
            makeMessage(kNumPayloads, kNumTensors),
            [&writeCompletedProm](const Error& error, Message /* unused */) {
              if (error) {
                ADD_FAILURE() << error.what();
                writeCompletedProm.set_value();
                return;
              }
              writeCompletedProm.set_value();
            });
        writeCompletedProm.get_future().get();

        pg.done(PeerGroup::kClient);
        pg.join(PeerGroup::kClient);

        context->join();
      });
}

TEST(Context, ServerPingPongTwice) {
  ForkedThreadPeerGroup pg;
  pg.spawn(
      [&]() {
        std::vector<std::shared_ptr<void>> buffers;
        std::promise<std::shared_ptr<Pipe>> serverPipePromise;
        std::promise<void> pingCompletedProm;

        auto context = makeContext();

        auto listener = context->listen(genUrls());
        pg.send(PeerGroup::kClient, listener->url("uv"));

        listener->accept([&](const Error& error, std::shared_ptr<Pipe> pipe) {
          if (error) {
            serverPipePromise.set_exception(
                std::make_exception_ptr(std::runtime_error(error.what())));
          } else {
            serverPipePromise.set_value(std::move(pipe));
          }
        });
        std::shared_ptr<Pipe> serverPipe = serverPipePromise.get_future().get();

        int numPingsGoneThrough = 0;
        for (int i = 0; i < 2; i++) {
          serverPipe->write(
              makeMessage(kNumPayloads, kNumTensors),
              [&serverPipe,
               &pingCompletedProm,
               &buffers,
               &numPingsGoneThrough,
               i](const Error& error, Message /* unused */) {
                if (error) {
                  ADD_FAILURE() << error.what();
                  pingCompletedProm.set_value();
                  return;
                }
                serverPipe->readDescriptor([&serverPipe,
                                            &pingCompletedProm,
                                            &buffers,
                                            &numPingsGoneThrough,
                                            i](const Error& error,
                                               Message message) {
                  if (error) {
                    ADD_FAILURE() << error.what();
                    pingCompletedProm.set_value();
                    return;
                  }
                  allocateMessage(message, buffers);
                  serverPipe->read(
                      std::move(message),
                      [&pingCompletedProm, &numPingsGoneThrough, i](
                          const Error& error, Message message) {
                        if (error) {
                          ADD_FAILURE() << error.what();
                          pingCompletedProm.set_value();
                          return;
                        }
                        EXPECT_TRUE(messagesAreEqual(
                            message, makeMessage(kNumPayloads, kNumTensors)));
                        EXPECT_EQ(numPingsGoneThrough, i);
                        numPingsGoneThrough++;
                        if (numPingsGoneThrough == 2) {
                          pingCompletedProm.set_value();
                        }
                      });
                });
              });
        }
        pingCompletedProm.get_future().get();

        pg.done(PeerGroup::kServer);
        pg.join(PeerGroup::kServer);

        context->join();
      },
      [&]() {
        std::vector<std::shared_ptr<void>> buffers;
        std::promise<void> pongCompletedProm;

        auto context = makeContext();

        auto url = pg.recv(PeerGroup::kClient);
        auto clientPipe = context->connect(url);

        int numPongsGoneThrough = 0;
        for (int i = 0; i < 2; i++) {
          clientPipe->readDescriptor([&clientPipe,
                                      &pongCompletedProm,
                                      &buffers,
                                      &numPongsGoneThrough,
                                      i](const Error& error, Message message) {
            if (error) {
              ADD_FAILURE() << error.what();
              pongCompletedProm.set_value();
              return;
            }
            allocateMessage(message, buffers);
            clientPipe->read(
                std::move(message),
                [&clientPipe, &pongCompletedProm, &numPongsGoneThrough, i](
                    const Error& error, Message message) mutable {
                  if (error) {
                    ADD_FAILURE() << error.what();
                    pongCompletedProm.set_value();
                    return;
                  }
                  clientPipe->write(
                      std::move(message),
                      [&pongCompletedProm, &numPongsGoneThrough, i](
                          const Error& error, Message /* unused */) {
                        if (error) {
                          ADD_FAILURE() << error.what();
                          pongCompletedProm.set_value();
                          return;
                        }
                        EXPECT_EQ(numPongsGoneThrough, i);
                        numPongsGoneThrough++;
                        if (numPongsGoneThrough == 2) {
                          pongCompletedProm.set_value();
                        }
                      });
                });
          });
        }
        pongCompletedProm.get_future().get();

        pg.done(PeerGroup::kClient);
        pg.join(PeerGroup::kClient);

        context->join();
      });
}

static void pipeRead(
    std::shared_ptr<Pipe>& pipe,
    std::vector<std::shared_ptr<void>>& buffers,
    std::function<void(const Error&, Message)> fn) {
  pipe->readDescriptor([&pipe, &buffers, fn{std::move(fn)}](
                           const Error& error, Message message) mutable {
    ASSERT_FALSE(error);
    allocateMessage(message, buffers);
    pipe->read(
        std::move(message),
        [fn{std::move(fn)}](const Error& error, Message message) mutable {
          fn(error, std::move(message));
        });
  });
}

TEST(Context, MixedTensorMessage) {
  constexpr int kNumMessages = 2;

  ForkedThreadPeerGroup pg;
  pg.spawn(
      [&]() {
        std::vector<std::shared_ptr<void>> buffers;
        std::promise<std::shared_ptr<Pipe>> serverPipePromise;
        std::promise<void> readCompletedProm;

        auto context = makeContext();

        auto listener = context->listen(genUrls());
        pg.send(PeerGroup::kClient, listener->url("uv"));

        listener->accept([&](const Error& error, std::shared_ptr<Pipe> pipe) {
          if (error) {
            serverPipePromise.set_exception(
                std::make_exception_ptr(std::runtime_error(error.what())));
          } else {
            serverPipePromise.set_value(std::move(pipe));
          }
        });
        std::shared_ptr<Pipe> serverPipe = serverPipePromise.get_future().get();

        std::atomic<int> readNum(kNumMessages);
        pipeRead(
            serverPipe,
            buffers,
            [&readNum, &readCompletedProm](
                const Error& error, Message message) {
              ASSERT_FALSE(error);
              EXPECT_TRUE(messagesAreEqual(
                  message, makeMessage(kNumPayloads, kNumTensors)));
              if (--readNum == 0) {
                readCompletedProm.set_value();
              }
            });
        pipeRead(
            serverPipe,
            buffers,
            [&readNum, &readCompletedProm](
                const Error& error, Message message) {
              ASSERT_FALSE(error);
              EXPECT_TRUE(messagesAreEqual(message, makeMessage(0, 0)));
              if (--readNum == 0) {
                readCompletedProm.set_value();
              }
            });
        readCompletedProm.get_future().get();

        pg.done(PeerGroup::kServer);
        pg.join(PeerGroup::kServer);

        context->join();
      },
      [&]() {
        std::promise<void> writeCompletedProm;

        auto context = makeContext();

        auto url = pg.recv(PeerGroup::kClient);
        auto clientPipe = context->connect(url);

        std::atomic<int> writeNum(kNumMessages);
        clientPipe->write(
            makeMessage(kNumPayloads, kNumTensors),
            [&writeNum, &writeCompletedProm](
                const Error& error, Message /* unused */) {
              ASSERT_FALSE(error) << error.what();
              if (--writeNum == 0) {
                writeCompletedProm.set_value();
              }
            });
        clientPipe->write(
            makeMessage(0, 0),
            [&writeNum, &writeCompletedProm](
                const Error& error, Message /* unused */) {
              ASSERT_FALSE(error) << error.what();
              if (--writeNum == 0) {
                writeCompletedProm.set_value();
              }
            });
        writeCompletedProm.get_future().get();

        pg.done(PeerGroup::kClient);
        pg.join(PeerGroup::kClient);

        context->join();
      });
}
