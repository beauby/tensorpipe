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
#include <thread>

#include <tensorpipe/channel/context.h>
#include <tensorpipe/common/queue.h>
#include <tensorpipe/transport/uv/context.h>

#include <gtest/gtest.h>

[[nodiscard]] std::pair<
    std::future<std::tuple<
        tensorpipe::Error,
        tensorpipe::channel::Channel::TDescriptor>>,
    std::future<tensorpipe::Error>>
sendWithFuture(
    std::shared_ptr<tensorpipe::channel::Channel> channel,
    const void* ptr,
    size_t length);

[[nodiscard]] std::future<tensorpipe::Error> recvWithFuture(
    std::shared_ptr<tensorpipe::channel::Channel> channel,
    tensorpipe::channel::Channel::TDescriptor descriptor,
    void* ptr,
    size_t length);

class ChannelTestHelper {
 public:
  virtual std::shared_ptr<tensorpipe::channel::Context> makeContext(
      std::string id) = 0;

  virtual ~ChannelTestHelper() = default;
};

class ChannelTest : public ::testing::TestWithParam<ChannelTestHelper*> {
 public:
  void testConnectionPair(
      std::function<void(std::shared_ptr<tensorpipe::transport::Connection>)>
          f1,
      std::function<void(std::shared_ptr<tensorpipe::transport::Connection>)>
          f2) {
    auto context = std::make_shared<tensorpipe::transport::uv::Context>();
    context->setId("harness");
    auto addr = "127.0.0.1";

    {
      tensorpipe::Queue<std::shared_ptr<tensorpipe::transport::Connection>> q1,
          q2;

      // Listening side.
      auto listener = context->listen(addr);
      listener->accept(
          [&](const tensorpipe::Error& error,
              std::shared_ptr<tensorpipe::transport::Connection> connection) {
            ASSERT_FALSE(error) << error.what();
            q1.push(std::move(connection));
          });

      // Connecting side.
      q2.push(context->connect(listener->addr()));

      // Run user specified functions.
      std::thread t1([&] { f1(q1.pop()); });
      std::thread t2([&] { f2(q2.pop()); });
      t1.join();
      t2.join();
    }

    context->join();
  }
};

class ProcessChannelTest : public ::testing::TestWithParam<ChannelTestHelper*> {
 public:
  using TSendStr = std::function<void(std::string)>;
  using TRecvStr = std::function<std::string()>;

  ProcessChannelTest() {
    pipe(pipefd1);
    pipe(pipefd2);
  }

  ~ProcessChannelTest() {
    for (int i = 0; i < 2; ++i) {
      close(pipefd1[i]);
      close(pipefd2[i]);
    }
  }

  void testConnectionPair(
      std::function<void(
          std::shared_ptr<tensorpipe::transport::Connection>,
          TSendStr,
          TRecvStr)> f1,
      std::function<void(
          std::shared_ptr<tensorpipe::transport::Connection>,
          TSendStr,
          TRecvStr)> f2) {
    auto addr = "127.0.0.1";

    // Listening side.
    pid_t pid1 = fork();
    ASSERT_GE(pid1, 0);
    if (pid1 == 0) {
      auto context = std::make_shared<tensorpipe::transport::uv::Context>();
      context->setId("harness");

      auto listener = context->listen(addr);

      std::string laddr = listener->addr();
      sendStr(laddr, pipefd1);
      tensorpipe::Queue<std::shared_ptr<tensorpipe::transport::Connection>> q;
      listener->accept(
          [&](const tensorpipe::Error& error,
              std::shared_ptr<tensorpipe::transport::Connection> connection) {
            ASSERT_FALSE(error) << error.what();
            q.push(std::move(connection));
          });

      f1(
          q.pop(),
          [this](std::string str) { this->sendStr(str, this->pipefd1); },
          [this]() { return this->recvStr(this->pipefd2); });

      context->join();
      return;
    }

    // Connecting side.
    pid_t pid2 = fork();
    ASSERT_GE(pid2, 0);

    if (pid2 == 0) {
      auto context = std::make_shared<tensorpipe::transport::uv::Context>();
      context->setId("harness");

      std::string laddr = this->recvStr(pipefd1);

      f2(
          context->connect(laddr),
          [this](std::string str) { this->sendStr(str, this->pipefd2); },
          [this]() { return this->recvStr(this->pipefd1); });

      context->join();
      return;
    }

    wait(0);
    wait(0);
  }

 private:
  int pipefd1[2];
  int pipefd2[2];

  void sendStr(std::string str, int pipefd[2]) {
    write(pipefd[1], str.c_str(), str.length());
    write(pipefd[1], "\n", 1);
  }

  std::string recvStr(int pipefd[2]) {
    std::string str;
    char c;
    while (read(pipefd[0], &c, 1) > 0) {
      if (c == '\n') {
        break;
      }
      str.push_back(c);
    }

    return str;
  }
};
