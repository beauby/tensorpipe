/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/test/channel/channel_test.h>

#include <numeric>

#include <tensorpipe/common/queue.h>

[[nodiscard]] std::pair<
    std::future<std::tuple<
        tensorpipe::Error,
        tensorpipe::channel::Channel::TDescriptor>>,
    std::future<tensorpipe::Error>>
sendWithFuture(
    std::shared_ptr<tensorpipe::channel::Channel> channel,
    const void* ptr,
    size_t length) {
  auto descriptorPromise = std::make_shared<
      std::promise<std::tuple<tensorpipe::Error, std::string>>>();
  auto promise = std::make_shared<std::promise<tensorpipe::Error>>();
  auto descriptorFuture = descriptorPromise->get_future();
  auto future = promise->get_future();
  channel->send(
      ptr,
      length,
      [descriptorPromise{std::move(descriptorPromise)}](
          const tensorpipe::Error& error, std::string descriptor) {
        descriptorPromise->set_value(
            std::make_tuple(error, std::move(descriptor)));
      },
      [promise{std::move(promise)}](const tensorpipe::Error& error) {
        promise->set_value(error);
      });
  return {std::move(descriptorFuture), std::move(future)};
}

[[nodiscard]] std::future<tensorpipe::Error> recvWithFuture(
    std::shared_ptr<tensorpipe::channel::Channel> channel,
    tensorpipe::channel::Channel::TDescriptor descriptor,
    void* ptr,
    size_t length) {
  auto promise = std::make_shared<std::promise<tensorpipe::Error>>();
  auto future = promise->get_future();
  channel->recv(
      std::move(descriptor),
      ptr,
      length,
      [promise{std::move(promise)}](const tensorpipe::Error& error) {
        promise->set_value(error);
      });
  return future;
}

using namespace tensorpipe;
using namespace tensorpipe::channel;

TEST_P(ChannelTest, DomainDescriptor) {
  std::shared_ptr<Context> context1 = GetParam()->makeContext("ctx1");
  std::shared_ptr<Context> context2 = GetParam()->makeContext("ctx2");
  EXPECT_FALSE(context1->domainDescriptor().empty());
  EXPECT_FALSE(context2->domainDescriptor().empty());
  EXPECT_EQ(context1->domainDescriptor(), context2->domainDescriptor());
}

TEST_P(ChannelTest, ClientToServer) {
  std::shared_ptr<Context> serverCtx = GetParam()->makeContext("server");
  std::shared_ptr<Context> clientCtx = GetParam()->makeContext("client");
  constexpr auto dataSize = 256;
  Queue<Channel::TDescriptor> descriptorQueue;
  std::promise<void> sendCompletedProm;
  std::promise<void> recvCompletedProm;

  testConnectionPair(
      [&](std::shared_ptr<transport::Connection> conn) {
        auto channel = serverCtx->createChannel(
            std::move(conn), Channel::Endpoint::kListen);

        // Initialize with sequential values.
        std::vector<uint8_t> data(dataSize);
        std::iota(data.begin(), data.end(), 0);

        // Perform send and wait for completion.
        std::future<std::tuple<Error, Channel::TDescriptor>> descriptorFuture;
        std::future<Error> sendFuture;
        std::tie(descriptorFuture, sendFuture) =
            sendWithFuture(channel, data.data(), data.size());
        Error descriptorError;
        Channel::TDescriptor descriptor;
        std::tie(descriptorError, descriptor) = descriptorFuture.get();
        EXPECT_FALSE(descriptorError) << descriptorError.what();
        descriptorQueue.push(std::move(descriptor));
        Error sendError = sendFuture.get();
        EXPECT_FALSE(sendError) << sendError.what();

        sendCompletedProm.set_value();
        recvCompletedProm.get_future().get();
      },
      [&](std::shared_ptr<transport::Connection> conn) {
        auto channel = clientCtx->createChannel(
            std::move(conn), Channel::Endpoint::kConnect);

        // Initialize with zeroes.
        std::vector<uint8_t> data(dataSize);
        std::fill(data.begin(), data.end(), 0);

        // Perform recv and wait for completion.
        std::future<Error> recvFuture = recvWithFuture(
            channel, descriptorQueue.pop(), data.data(), data.size());
        Error recvError = recvFuture.get();
        EXPECT_FALSE(recvError) << recvError.what();

        // Validate contents of vector.
        for (auto i = 0; i < data.size(); i++) {
          EXPECT_EQ(data[i], i);
        }

        recvCompletedProm.set_value();
        sendCompletedProm.get_future().get();
      });

  serverCtx->join();
  clientCtx->join();
}

TEST_P(ChannelTest, ServerToClient) {
  std::shared_ptr<Context> serverCtx = GetParam()->makeContext("server");
  std::shared_ptr<Context> clientCtx = GetParam()->makeContext("client");
  constexpr auto dataSize = 256;
  Queue<Channel::TDescriptor> descriptorQueue;
  std::promise<void> sendCompletedProm;
  std::promise<void> recvCompletedProm;

  testConnectionPair(
      [&](std::shared_ptr<transport::Connection> conn) {
        auto channel = serverCtx->createChannel(
            std::move(conn), Channel::Endpoint::kListen);

        // Initialize with zeroes.
        std::vector<uint8_t> data(dataSize);
        std::fill(data.begin(), data.end(), 0);

        // Perform recv and wait for completion.
        std::future<Error> recvFuture = recvWithFuture(
            channel, descriptorQueue.pop(), data.data(), data.size());
        Error recvError = recvFuture.get();
        EXPECT_FALSE(recvError) << recvError.what();

        // Validate contents of vector.
        for (auto i = 0; i < data.size(); i++) {
          EXPECT_EQ(data[i], i);
        }

        recvCompletedProm.set_value();
        sendCompletedProm.get_future().get();
      },
      [&](std::shared_ptr<transport::Connection> conn) {
        auto channel = clientCtx->createChannel(
            std::move(conn), Channel::Endpoint::kConnect);

        // Initialize with sequential values.
        std::vector<uint8_t> data(dataSize);
        std::iota(data.begin(), data.end(), 0);

        // Perform send and wait for completion.
        std::future<std::tuple<Error, Channel::TDescriptor>> descriptorFuture;
        std::future<Error> sendFuture;
        std::tie(descriptorFuture, sendFuture) =
            sendWithFuture(channel, data.data(), data.size());
        Error descriptorError;
        Channel::TDescriptor descriptor;
        std::tie(descriptorError, descriptor) = descriptorFuture.get();
        EXPECT_FALSE(descriptorError) << descriptorError.what();
        descriptorQueue.push(std::move(descriptor));
        Error sendError = sendFuture.get();
        EXPECT_FALSE(sendError) << sendError.what();

        sendCompletedProm.set_value();
        recvCompletedProm.get_future().get();
      });

  serverCtx->join();
  clientCtx->join();
}

TEST_P(ChannelTest, SendMultipleTensors) {
  std::shared_ptr<Context> serverCtx = GetParam()->makeContext("server");
  std::shared_ptr<Context> clientCtx = GetParam()->makeContext("client");
  constexpr auto dataSize = 256 * 1024; // 256KB
  Queue<Channel::TDescriptor> descriptorQueue;
  std::promise<void> sendCompletedProm;
  std::promise<void> recvCompletedProm;
  constexpr int numTensors = 100;

  testConnectionPair(
      [&](std::shared_ptr<transport::Connection> conn) {
        auto channel = serverCtx->createChannel(
            std::move(conn), Channel::Endpoint::kListen);

        // Initialize with sequential values.
        std::vector<uint8_t> data(dataSize);
        std::iota(data.begin(), data.end(), 0);

        // Error futures
        std::vector<std::future<Error>> sendFutures;

        // Perform send and wait for completion.
        for (int i = 0; i < numTensors; i++) {
          std::future<std::tuple<Error, Channel::TDescriptor>> descriptorFuture;
          std::future<Error> sendFuture;
          std::tie(descriptorFuture, sendFuture) =
              sendWithFuture(channel, data.data(), data.size());
          Error descriptorError;
          Channel::TDescriptor descriptor;
          std::tie(descriptorError, descriptor) = descriptorFuture.get();
          EXPECT_FALSE(descriptorError) << descriptorError.what();
          descriptorQueue.push(std::move(descriptor));
          sendFutures.push_back(std::move(sendFuture));
        }
        for (auto& sendFuture : sendFutures) {
          Error sendError = sendFuture.get();
          EXPECT_FALSE(sendError) << sendError.what();
        }

        sendCompletedProm.set_value();
        recvCompletedProm.get_future().get();
      },
      [&](std::shared_ptr<transport::Connection> conn) {
        auto channel = clientCtx->createChannel(
            std::move(conn), Channel::Endpoint::kConnect);

        // Initialize with zeroes.
        std::vector<std::vector<uint8_t>> dataVec(
            numTensors, std::vector<uint8_t>(dataSize, 0));

        // Error futures
        std::vector<std::future<Error>> recvFutures;

        // Perform recv and wait for completion.
        for (int i = 0; i < numTensors; i++) {
          std::future<Error> recvFuture = recvWithFuture(
              channel, descriptorQueue.pop(), dataVec[i].data(), dataSize);
          recvFutures.push_back(std::move(recvFuture));
        }
        for (auto& recvFuture : recvFutures) {
          Error recvError = recvFuture.get();
          EXPECT_FALSE(recvError) << recvError.what();
        }

        // Validate contents of vector.
        for (auto& data : dataVec) {
          for (int i = 0; i < data.size(); i++) {
            uint8_t value = i;
            EXPECT_EQ(data[i], value);
          }
        }

        recvCompletedProm.set_value();
        sendCompletedProm.get_future().get();
      });

  serverCtx->join();
  clientCtx->join();
}

TEST_P(ChannelTest, contextIsNotJoined) {
  std::shared_ptr<Context> context = GetParam()->makeContext("ctx");
  std::promise<void> serverReadyProm;

  testConnectionPair(
      [&](std::shared_ptr<transport::Connection> conn) {
        serverReadyProm.set_value();
        context->createChannel(std::move(conn), Channel::Endpoint::kListen);
      },
      [&](std::shared_ptr<transport::Connection> conn) {
        serverReadyProm.get_future().wait();
        context->createChannel(std::move(conn), Channel::Endpoint::kConnect);
      });
}

TEST_P(ChannelTest, CallbacksAreDeferred) {
  // This test wants to make sure that the "heavy lifting" of copying data isn't
  // performed inline inside the recv method as that would make the user-facing
  // read method of the pipe blocking. However, since we can't really check that
  // behavior, we'll check a highly correlated one: that the recv callback isn't
  // called inline from within the recv method. We do so by having that behavior
  // cause a deadlock.
  std::shared_ptr<Context> serverCtx = GetParam()->makeContext("server");
  std::shared_ptr<Context> clientCtx = GetParam()->makeContext("client");
  constexpr auto dataSize = 256;
  Queue<Channel::TDescriptor> descriptorQueue;
  std::promise<void> sendCompletedProm;
  std::promise<void> recvCompletedProm;

  testConnectionPair(
      [&](std::shared_ptr<transport::Connection> conn) {
        auto channel = serverCtx->createChannel(
            std::move(conn), Channel::Endpoint::kListen);

        // Initialize with sequential values.
        std::vector<uint8_t> data(dataSize);
        std::iota(data.begin(), data.end(), 0);

        // Perform send and wait for completion.
        std::promise<std::tuple<Error, Channel::TDescriptor>> descriptorPromise;
        std::promise<Error> sendPromise;
        std::mutex mutex;
        std::unique_lock<std::mutex> callerLock(mutex);
        channel->send(
            data.data(),
            data.size(),
            [&descriptorPromise](
                const Error& error, Channel::TDescriptor descriptor) {
              descriptorPromise.set_value(
                  std::make_tuple(error, std::move(descriptor)));
            },
            [&sendPromise, &mutex](const Error& error) {
              std::unique_lock<std::mutex> calleeLock(mutex);
              sendPromise.set_value(error);
            });
        callerLock.unlock();
        Error descriptorError;
        Channel::TDescriptor descriptor;
        std::tie(descriptorError, descriptor) =
            descriptorPromise.get_future().get();
        EXPECT_FALSE(descriptorError) << descriptorError.what();
        descriptorQueue.push(std::move(descriptor));
        Error sendError = sendPromise.get_future().get();
        EXPECT_FALSE(sendError) << sendError.what();

        sendCompletedProm.set_value();
        recvCompletedProm.get_future().get();
      },
      [&](std::shared_ptr<transport::Connection> conn) {
        auto channel = clientCtx->createChannel(
            std::move(conn), Channel::Endpoint::kConnect);

        // Initialize with zeroes.
        std::vector<uint8_t> data(dataSize);
        std::fill(data.begin(), data.end(), 0);

        // Perform recv and wait for completion.
        std::promise<Error> recvPromise;
        std::mutex mutex;
        std::unique_lock<std::mutex> callerLock(mutex);
        channel->recv(
            descriptorQueue.pop(),
            data.data(),
            data.size(),
            [&recvPromise, &mutex](const Error& error) {
              std::unique_lock<std::mutex> calleeLock(mutex);
              recvPromise.set_value(error);
            });
        callerLock.unlock();
        Error recvError = recvPromise.get_future().get();
        EXPECT_FALSE(recvError) << recvError.what();

        // Validate contents of vector.
        for (auto i = 0; i < data.size(); i++) {
          EXPECT_EQ(data[i], i);
        }

        recvCompletedProm.set_value();
        sendCompletedProm.get_future().get();
      });

  serverCtx->join();
  clientCtx->join();
}

// Each channel test has a general setup and two peers. Those two peers are
// either processes or threads (for now). The general setup is shared by all
// peers (possibly copied when it makes sense). Each peer is a method on the
// test class with a few helpers for message passing (pipes for all for now),
// and joining (sending the other peer it is done and waiting for the other peer
// to be done).

CHANNEL_TEST(ProcessTest, FooBar) {
  // common setup
}

CHANNEL_TEST_SERVER(FooBar) {
  sendClient("descriptor");
  join();
}

CHANNEL_TEST_CLIENT(FooBar) {
  join();
}
