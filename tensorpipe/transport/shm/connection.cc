/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/transport/shm/connection.h>

#include <string.h>

#include <vector>

#include <tensorpipe/common/callback.h>
#include <tensorpipe/common/defs.h>
#include <tensorpipe/common/error_macros.h>
#include <tensorpipe/transport/error.h>
#include <tensorpipe/util/ringbuffer/protobuf_streams.h>

namespace tensorpipe {
namespace transport {
namespace shm {

namespace {
// Reads happen only if the user supplied a callback (and optionally
// a destination buffer). The callback is run from the event loop
// thread upon receiving a notification from our peer.
//
// The memory pointer argument to the callback is valid only for the
// duration of the callback. If the memory contents must be
// preserved for longer, it must be copied elsewhere.
//
class ReadOperation {
  enum Mode {
    READ_LENGTH,
    READ_PAYLOAD,
  };

 public:
  using read_fn = std::function<ssize_t(util::ringbuffer::Consumer&)>;
  explicit ReadOperation(void* ptr, size_t len, read_callback_fn fn);
  explicit ReadOperation(read_fn reader, read_callback_fn fn);
  explicit ReadOperation(read_callback_fn fn);

  // Processes a pending read.
  bool handleRead(util::ringbuffer::Consumer& consumer);

  bool completed() const {
    return (mode_ == READ_PAYLOAD && bytesRead_ == len_);
  }

  void handleError(const Error& error);

 private:
  Mode mode_{READ_LENGTH};
  void* ptr_{nullptr};
  read_fn reader_;
  std::unique_ptr<uint8_t[]> buf_;
  size_t len_{0};
  size_t bytesRead_{0};
  read_callback_fn fn_;
};

  ReadOperation::ReadOperation(void* ptr, size_t len, read_callback_fn fn)
    : ptr_(ptr), len_(len), fn_(std::move(fn)) {}

ReadOperation::ReadOperation(read_fn reader, read_callback_fn fn)
    : reader_(std::move(reader)), fn_(std::move(fn)) {}

ReadOperation::ReadOperation(read_callback_fn fn)
    : fn_(std::move(fn)) {}

bool ReadOperation::handleRead(util::ringbuffer::Consumer& inbox) {
  // Start read transaction.
  // Retry because this must succeed.
  for (;;) {
    const auto ret = inbox.startTx();
    TP_DCHECK(ret >= 0 || ret == -EAGAIN);
    if (ret < 0) {
      continue;
    }
    break;
  }

  bool lengthRead = false;
  if (reader_) {
    auto ret = reader_(inbox);
    if (ret == -ENODATA) {
      ret = inbox.cancelTx();
      TP_THROW_SYSTEM_IF(ret < 0, -ret);
      return false;
    }
    TP_THROW_SYSTEM_IF(ret < 0, -ret);

    mode_ = READ_PAYLOAD;
    bytesRead_ = len_ = ret;
  } else {
    if (mode_ == READ_LENGTH) {
      uint32_t length;
      {
        ssize_t ret;
        ret = inbox.copyInTx(sizeof(length), &length);
        if (ret == -ENODATA) {
          ret = inbox.cancelTx();
          TP_THROW_SYSTEM_IF(ret < 0, -ret);
          return false;
        }
        TP_THROW_SYSTEM_IF(ret < 0, -ret);
      }

      if (ptr_ != nullptr) {
        TP_DCHECK_EQ(length, len_);
      } else {
        len_ = length;
        buf_ = std::make_unique<uint8_t[]>(len_);
        ptr_ = buf_.get();
      }
      mode_ = READ_PAYLOAD;
      lengthRead = true;
    }

    {
      const auto ret = inbox.copyAtMostInTx(
          len_ - bytesRead_, reinterpret_cast<uint8_t*>(ptr_) + bytesRead_);
      if (ret == -ENODATA) {
        if (lengthRead) {
          const auto ret = inbox.commitTx();
          TP_THROW_SYSTEM_IF(ret < 0, -ret);
          return true;
        } else {
          const auto ret = inbox.cancelTx();
          TP_THROW_SYSTEM_IF(ret < 0, -ret);
          return false;
        }
      }
      TP_THROW_SYSTEM_IF(ret < 0, -ret);
      bytesRead_ += ret;
    }
  }

  {
    const auto ret = inbox.commitTx();
    TP_THROW_SYSTEM_IF(ret < 0, -ret);
  }

  if (completed()) {
    fn_(Error::kSuccess, ptr_, len_);
  }

  return true;
}

void ReadOperation::handleError(const Error& error) {
  fn_(error, nullptr, 0);
}

// Writes happen only if the user supplied a memory pointer, the
// number of bytes to write, and a callback to execute upon
// completion of the write.
//
// The memory pointed to by the pointer may only be reused or freed
// after the callback has been called.
//
class WriteOperation {
  enum Mode {
    WRITE_LENGTH,
    WRITE_PAYLOAD,
  };

 public:
  using write_fn = std::function<ssize_t(util::ringbuffer::Producer&)>;
  WriteOperation(const void* ptr, size_t len, write_callback_fn fn);
  WriteOperation(write_fn writer, write_callback_fn fn);

  bool handleWrite(util::ringbuffer::Producer& producer);

  bool completed() const {
    return (mode_ == WRITE_PAYLOAD && bytesWritten_ == len_);
  }

  void handleError(const Error& error);

 private:
  Mode mode_{WRITE_LENGTH};
  const void* ptr_{nullptr};
  write_fn writer_;
  size_t len_{0};
  size_t bytesWritten_{0};
  write_callback_fn fn_;
};

WriteOperation::WriteOperation(
    const void* ptr,
    size_t len,
    write_callback_fn fn)
    : ptr_(ptr), len_(len), fn_(std::move(fn)) {}

WriteOperation::WriteOperation(
    write_fn writer,
    write_callback_fn fn)
    : writer_(std::move(writer)), fn_(std::move(fn)) {}

bool WriteOperation::handleWrite(
    util::ringbuffer::Producer& outbox) {
  // Start write transaction.
  // Retry because this must succeed.
  // TODO: fallback if it doesn't.
  for (;;) {
    const auto ret = outbox.startTx();
    TP_DCHECK(ret >= 0 || ret == -EAGAIN);
    if (ret < 0) {
      continue;
    }
    break;
  }

  ssize_t ret;
  if (writer_) {
    ret = writer_(outbox);
    if (ret > 0) {
      mode_ = WRITE_PAYLOAD;
      bytesWritten_ = len_ = ret;
    }
  } else {
    if (mode_ == WRITE_LENGTH) {
      ret = outbox.writeInTx<uint32_t>(len_);
      if (ret > 0) {
        mode_ = WRITE_PAYLOAD;
      }
    }
    if (mode_ == WRITE_PAYLOAD) {
      ret = outbox.writeAtMostInTx(
          len_ - bytesWritten_,
          static_cast<const uint8_t*>(ptr_) + bytesWritten_);
      if (ret > 0) {
        bytesWritten_ += ret;
      }
    }
  }

  if (ret == -ENOSPC) {
    const auto ret = outbox.cancelTx();
    TP_THROW_SYSTEM_IF(ret < 0, -ret);
    return false;
  }
  TP_THROW_SYSTEM_IF(ret < 0, -ret);

  {
    const auto ret = outbox.commitTx();
    TP_THROW_SYSTEM_IF(ret < 0, -ret);
  }

  if (completed()) {
    fn_(Error::kSuccess);
  }

  return true;
}

void WriteOperation::handleError(const Error& error) {
  fn_(error);
}

} // namespace

class Impl : public std::enable_shared_from_this<Connection::Impl>,
             public EventHandler {
 public:
  Impl(std::shared_ptr<Loop> loop, std::shared_ptr<Socket> socket);

  void initFromLoop();

  void closeFromLoop();

  // Implementation of EventHandler.
  void handleEventsFromReactor(int events) override;

  // Handle events of type EPOLLIN.
  void handleEventInFromReactor(std::unique_lock<std::mutex> lock);

  // Handle events of type EPOLLOUT.
  void handleEventOutFromReactor(std::unique_lock<std::mutex> lock);

  // Handle events of type EPOLLERR.
  void handleEventErrFromReactor(std::unique_lock<std::mutex> lock);

  // Handle events of type EPOLLHUP.
  void handleEventHupFromReactor(std::unique_lock<std::mutex> lock);

  // Handle inbox being readable.
  //
  // This is triggered from the reactor loop when this connection's
  // peer has written an entry into our inbox. It is called once per
  // message. Because it's called from another thread, we must always
  // take care to acquire the connection's lock here.
  void handleInboxReadableFromReactor();

  // Handle outbox being writable.
  //
  // This is triggered from the reactor loop when this connection's
  // peer has read an entry from our outbox. It is called once per
  // message. Because it's called from another thread, we must always
  // take care to acquire the connection's lock here.
  void handleOutboxWritableFromReactor();

 private:
  // Defer execution of processReadOperations to loop thread.
  void triggerProcessReadOperations();

  // Process pending read operations if in an operational or error state.
  void processReadOperationsFromReactor(std::unique_lock<std::mutex>& lock);

  // Defer execution of processWriteOperations to loop thread.
  void triggerProcessWriteOperations();

  // Process pending write operations if in an operational state.
  void processWriteOperationsFromReactor(std::unique_lock<std::mutex>& lock);

  // Set error object while holding mutex.
  void setErrorHoldingMutexFromReactor(Error&&);

  // Fail with error while holding mutex.
  void failHoldingMutexFromReactor(Error&&, std::unique_lock<std::mutex>& lock);

  // Close connection while holding mutex.
  void closeHoldingMutex();

  static constexpr auto kDefaultSize = 2 * 1024 * 1024;

  enum State {
    INITIALIZING = 1,
    SEND_FDS,
    RECV_FDS,
    ESTABLISHED,
    DESTROYING,
  };

  std::mutex mutex_;
  State state_{INITIALIZING};
  Error error_;
  std::shared_ptr<Reactor> reactor_;
  std::shared_ptr<Socket> socket_;

  // Inbox.
  int inboxHeaderFd_;
  int inboxDataFd_;
  optional<util::ringbuffer::Consumer> inbox_;
  optional<Reactor::TToken> inboxReactorToken_;

  // Outbox.
  optional<util::ringbuffer::Producer> outbox_;
  optional<Reactor::TToken> outboxReactorToken_;

  // Peer trigger/tokens.
  optional<Reactor::Trigger> peerReactorTrigger_;
  optional<Reactor::TToken> peerInboxReactorToken_;
  optional<Reactor::TToken> peerOutboxReactorToken_;

  // Pending read operations.
  std::deque<ReadOperation> readOperations_;

  // Pending write operations.
  std::deque<WriteOperation> writeOperations_;

  // By having the instance store a shared_ptr to itself we create a reference
  // cycle which will "leak" the instance. This allows us to detach its
  // lifetime from the connection and sync it with the TCPHandle's life cycle.
  std::shared_ptr<Impl> leak_;
};

Connection::Impl::Impl(
    std::shared_ptr<Loop> loop,
    std::shared_ptr<Socket> socket)
    : loop_(std::move(loop)),
      socket_(std::move(socket)),
      reactor_(loop_->reactor()) {}

void Connection::Impl::initFromLoop() {
  leak_ = shared_from_this();

  // Ensure underlying control socket is non-blocking such that it
  // works well with event driven I/O.
  socket_->block(false);

  // Create ringbuffer for inbox.
  std::shared_ptr<util::ringbuffer::RingBuffer> inboxRingBuffer;
  std::tie(inboxHeaderFd_, inboxDataFd_, inboxRingBuffer) =
      util::ringbuffer::shm::create(kDefaultSize);
  inbox_.emplace(std::move(inboxRingBuffer));

  // Register method to be called when our peer writes to our inbox.
  inboxReactorToken_ = reactor_->add(
      runIfAlive(*this, std::function<void(Connection&)>([](Connection& conn) {
        conn.handleInboxReadableFromReactor();
      })));

  // Register method to be called when our peer reads from our outbox.
  outboxReactorToken_ = reactor_->add(
      runIfAlive(*this, std::function<void(Connection&)>([](Connection& conn) {
        conn.handleOutboxWritableFromReactor();
      })));

  // We're sending file descriptors first, so wait for writability.
  state_ = SEND_FDS;
  loop_->registerDescriptor(socket_->fd(), EPOLLOUT, shared_from_this());
}

std::shared_ptr<Connection> Connection::create_(
    std::shared_ptr<Loop> loop,
    std::shared_ptr<Socket> socket) {
  auto conn = std::make_shared<Connection>(
      ConstructorToken(), std::move(loop), std::move(socket));
  conn->init_();
  return conn;
}

Connection::Connection(
    ConstructorToken /* unused */,
    std::shared_ptr<Loop> loop,
    std::shared_ptr<Socket> socket)
    : loop_(std::move(loop)),
      socket_(std::move(socket)) {
}

Connection::~Connection() {
  loop_->deferToLoop([impl{impl_}]() { impl->closeFromLoop(); });
}

void Connection::init_() {
  loop_->deferToLoop([impl{impl_}]() { impl->initFromLoop(); });
}

// Implementation of transport::Connection.
void Connection::read(read_callback_fn fn) {
  loop_->deferToLoop([impl{impl_}, fn{std::move(fn)}]() mutable {
    impl->readFromLoop(std::move(fn));
  });
}

void Connection::Impl::read(read_callback_fn fn) {
  std::unique_lock<std::mutex> guard(mutex_);
  readOperations_.emplace_back(std::move(fn));

  // If there are pending read operations, make sure the event loop
  // processes them, now that we have an additional callback.
  triggerProcessReadOperations();
}

// Implementation of transport::Connection.
void Connection::read(
    google::protobuf::MessageLite& message,
    read_proto_callback_fn fn) {
  loop_->deferToLoop([impl{impl_}, &message, fn{std::move(fn)}]() mutable {
    impl->readFromLoop(message, std::move(fn));
  });
}

void Connection::Impl::read(
    google::protobuf::MessageLite& message,
    read_proto_callback_fn fn) {
  std::unique_lock<std::mutex> guard(mutex_);
  readOperations_.emplace_back(
      [&message](util::ringbuffer::Consumer& inbox) -> ssize_t {
        uint32_t len;
        {
          const auto ret = inbox.copyInTx(sizeof(len), &len);
          if (ret == -ENODATA) {
            return -ENODATA;
          }
          TP_THROW_SYSTEM_IF(ret < 0, -ret);
        }

        if (len + sizeof(uint32_t) > kDefaultSize) {
          return -EPERM;
        }

        util::ringbuffer::ZeroCopyInputStream is(&inbox, len);
        if (!message.ParseFromZeroCopyStream(&is)) {
          return -ENODATA;
        }

        TP_DCHECK_EQ(len, is.ByteCount());
        return is.ByteCount();
      },
      [fn{std::move(fn)}](
          const Error& error, const void* /* unused */, size_t /* unused */) {
        fn(error);
      });

  // If there are pending read operations, make sure the event loop
  // processes them, now that we have an additional callback.
  triggerProcessReadOperations();
}

// Implementation of transport::Connection.
void Connection::read(void* ptr, size_t length, read_callback_fn fn) {
  loop_->deferToLoop([impl{impl_}, ptr, length, fn{std::move(fn)}]() mutable {
    impl->readFromLoop(ptr, length, std::move(fn));
  });
}

void Connection::Impl::read(void* ptr, size_t length, read_callback_fn fn) {
  std::unique_lock<std::mutex> guard(mutex_);
  readOperations_.emplace_back(ptr, length, std::move(fn));

  // If there are pending read operations, make sure the event loop
  // processes them, now that we have an additional callback.
  triggerProcessReadOperations();
}

// Implementation of transport::Connection
void Connection::write(const void* ptr, size_t length, write_callback_fn fn) {
  loop_->deferToLoop([impl{impl_}, ptr, length, fn{std::move(fn)}]() mutable {
    impl->writeFromLoop(ptr, length, std::move(fn));
  });
}

void Connection::Impl::write(
    const void* ptr,
    size_t length,
    write_callback_fn fn) {
  std::unique_lock<std::mutex> guard(mutex_);
  writeOperations_.emplace_back(ptr, length, std::move(fn));
  triggerProcessWriteOperations();
}

// Implementation of transport::Connection
void Connection::write(
    const google::protobuf::MessageLite& message,
    write_callback_fn fn) {
  loop_->deferToLoop([impl{impl_}, &message, fn{std::move(fn)}]() mutable {
    impl->writeFromLoop(message, std::move(fn));
  });
}

void Connection::Impl::write(
    const google::protobuf::MessageLite& message,
    write_callback_fn fn) {
  std::unique_lock<std::mutex> guard(mutex_);
  writeOperations_.emplace_back(
      [&message](util::ringbuffer::Producer& outbox) -> ssize_t {
        size_t len = message.ByteSize();
        if (len + sizeof(uint32_t) > kDefaultSize) {
          return -EPERM;
        }

        const auto ret = outbox.writeInTx<uint32_t>(len);
        if (ret < 0) {
          return ret;
        }

        util::ringbuffer::ZeroCopyOutputStream os(&outbox, len);
        if (!message.SerializeToZeroCopyStream(&os)) {
          return -ENOSPC;
        }

        TP_DCHECK_EQ(len, os.ByteCount());

        return os.ByteCount();
      },
      std::move(fn));
  triggerProcessWriteOperations();
}

  void Connection::Impl::handleEventsFromReactor(int events) {
  TP_DCHECK(loop_->inReactorThread());
  std::unique_lock<std::mutex> lock(mutex_);

  // Handle only one of the events in the mask. Events on the control
  // file descriptor are rare enough for the cost of having epoll call
  // into this function multiple times to not matter. The benefit is
  // that we never have to acquire the lock more than once and that
  // every handler can close and unregister the control file
  // descriptor from the event loop, without worrying about the next
  // handler trying to do so as well.
  if (events & EPOLLIN) {
    handleEventInFromReactor(std::move(lock));
    return;
  }
  if (events & EPOLLOUT) {
    handleEventOutFromReactor(std::move(lock));
    return;
  }
  if (events & EPOLLERR) {
    handleEventErrFromReactor(std::move(lock));
    return;
  }
  if (events & EPOLLHUP) {
    handleEventHupFromReactor(std::move(lock));
    return;
  }
}

void Connection::Impl::handleEventInFromReactor(
    std::unique_lock<std::mutex> lock) {
  TP_DCHECK(loop_->inReactorThread());
  if (state_ == RECV_FDS) {
    Fd reactorHeaderFd;
    Fd reactorDataFd;
    Fd outboxHeaderFd;
    Fd outboxDataFd;
    Reactor::TToken peerInboxReactorToken;
    Reactor::TToken peerOutboxReactorToken;

    // Receive the reactor token, reactor fds, and inbox fds.
    auto err = socket_->recvPayloadAndFds(
        peerInboxReactorToken,
        peerOutboxReactorToken,
        reactorHeaderFd,
        reactorDataFd,
        outboxHeaderFd,
        outboxDataFd);
    if (err) {
      failHoldingMutexFromReactor(std::move(err), lock);
      return;
    }

    // Load ringbuffer for outbox.
    outbox_.emplace(util::ringbuffer::shm::load(
        outboxHeaderFd.release(), outboxDataFd.release()));

    // Initialize remote reactor trigger.
    peerReactorTrigger_.emplace(
        std::move(reactorHeaderFd), std::move(reactorDataFd));

    peerInboxReactorToken_ = peerInboxReactorToken;
    peerOutboxReactorToken_ = peerOutboxReactorToken;

    // The connection is usable now.
    state_ = ESTABLISHED;
    processWriteOperationsFromReactor(lock);
    // Trigger read operations in case a pair of local read() and remote
    // write() happened before connection is established. Otherwise read()
    // callback would lose if it's the only read() request.
    processReadOperationsFromReactor(lock);
    return;
  }

  if (state_ == ESTABLISHED) {
    // We don't expect to read anything on this socket once the
    // connection has been established. If we do, assume it's a
    // zero-byte read indicating EOF.
    setErrorHoldingMutexFromReactor(TP_CREATE_ERROR(EOFError));
    closeHoldingMutex();
    processReadOperationsFromReactor(lock);
    return;
  }

  TP_LOG_WARNING() << "handleEventIn not handled";
}

void Connection::Impl::handleEventOutFromReactor(
    std::unique_lock<std::mutex> lock) {
  TP_DCHECK(loop_->inReactorThread());
  if (state_ == SEND_FDS) {
    int reactorHeaderFd;
    int reactorDataFd;
    std::tie(reactorHeaderFd, reactorDataFd) = reactor_->fds();

    // Send our reactor token, reactor fds, and inbox fds.
    auto err = socket_->sendPayloadAndFds(
        inboxReactorToken_.value(),
        outboxReactorToken_.value(),
        reactorHeaderFd,
        reactorDataFd,
        inboxHeaderFd_,
        inboxDataFd_);
    if (err) {
      failHoldingMutexFromReactor(std::move(err), lock);
      return;
    }

    // Sent our fds. Wait for fds from peer.
    state_ = RECV_FDS;
    loop_->registerDescriptor(socket_->fd(), EPOLLIN, shared_from_this());
    return;
  }

  TP_LOG_WARNING() << "handleEventOut not handled";
}

void Connection::Impl::handleEventErrFromReactor(
    std::unique_lock<std::mutex> lock) {
  TP_DCHECK(loop_->inReactorThread());
  setErrorHoldingMutexFromReactor(TP_CREATE_ERROR(EOFError));
  closeHoldingMutex();
  processReadOperationsFromReactor(lock);
}

void Connection::Impl::handleEventHupFromReactor(
    std::unique_lock<std::mutex> lock) {
  TP_DCHECK(loop_->inReactorThread());
  setErrorHoldingMutexFromReactor(TP_CREATE_ERROR(EOFError));
  closeHoldingMutex();
  processReadOperationsFromReactor(lock);
}

void Connection::Impl::handleInboxReadableFromReactor() {
  TP_DCHECK(loop_->inReactorThread());
  std::unique_lock<std::mutex> lock(mutex_);
  processReadOperationsFromReactor(lock);
}

void Connection::Impl::handleOutboxWritableFromReactor() {
  TP_DCHECK(loop_->inReactorThread());
  std::unique_lock<std::mutex> lock(mutex_);
  processWriteOperationsFromReactor(lock);
}

void Connection::Impl::triggerProcessReadOperations() {
  loop_->deferToReactor([ptr{shared_from_this()}, this] {
    std::unique_lock<std::mutex> lock(mutex_);
    processReadOperationsFromReactor(lock);
  });
}

void Connection::Impl::processReadOperationsFromReactor(
    std::unique_lock<std::mutex>& lock) {
  TP_DCHECK(loop_->inReactorThread());
  TP_DCHECK(lock.owns_lock());

  if (error_) {
    std::deque<ReadOperation> operationsToError;
    std::swap(operationsToError, readOperations_);
    lock.unlock();
    for (auto& readOperation : operationsToError) {
      readOperation.handleError(error_);
    }
    lock.lock();
    return;
  }

  // Process all read read operations that we can immediately serve, only
  // when connection is established.
  if (state_ != ESTABLISHED) {
    return;
  }
  // Serve read operations
  while (!readOperations_.empty()) {
    auto readOperation = std::move(readOperations_.front());
    readOperations_.pop_front();
    lock.unlock();
    if (readOperation.handleRead(*inbox_)) {
      peerReactorTrigger_->run(peerOutboxReactorToken_.value());
    }
    lock.lock();
    if (!readOperation.completed()) {
      readOperations_.push_front(std::move(readOperation));
      break;
    }
  }

  TP_DCHECK(lock.owns_lock());
}

void Connection::Impl::triggerProcessWriteOperations() {
  loop_->deferToReactor([ptr{shared_from_this()}, this] {
    std::unique_lock<std::mutex> lock(mutex_);
    processWriteOperationsFromReactor(lock);
  });
}

void Connection::Impl::processWriteOperationsFromReactor(
    std::unique_lock<std::mutex>& lock) {
  TP_DCHECK(loop_->inReactorThread());
  TP_DCHECK(lock.owns_lock());

  if (state_ < ESTABLISHED) {
    return;
  }

  if (error_) {
    std::deque<WriteOperation> operationsToError;
    std::swap(operationsToError, writeOperations_);
    lock.unlock();
    for (auto& writeOperation : operationsToError) {
      writeOperation.handleError(error_);
    }
    lock.lock();
    return;
  }

  while (!writeOperations_.empty()) {
    auto writeOperation = std::move(writeOperations_.front());
    writeOperations_.pop_front();
    lock.unlock();
    if (writeOperation.handleWrite(*outbox_)) {
      peerReactorTrigger_->run(peerInboxReactorToken_.value());
    }
    lock.lock();
    if (!writeOperation.completed()) {
      writeOperations_.push_front(writeOperation);
      break;
    }
  }

  TP_DCHECK(lock.owns_lock());
}

void Connection::Impl::setErrorHoldingMutexFromReactor(Error&& error) {
  TP_DCHECK(loop_->inReactorThread());
  error_ = error;
}

void Connection::Impl::failHoldingMutexFromReactor(
    Error&& error,
    std::unique_lock<std::mutex>& lock) {
  TP_DCHECK(loop_->inReactorThread());
  setErrorHoldingMutexFromReactor(std::move(error));
  while (!readOperations_.empty()) {
    auto& readOperation = readOperations_.front();
    lock.unlock();
    readOperation.handleError(error_);
    lock.lock();
    readOperations_.pop_front();
  }
  while (!writeOperations_.empty()) {
    auto& writeOperation = writeOperations_.front();
    lock.unlock();
    writeOperation.handleError(error_);
    lock.lock();
    writeOperations_.pop_front();
  }
}

void Connection::Impl::closeFromLoop() {
  // To avoid races, the close operation should also be queued and deferred to
  // the reactor. However, since close can be called from the destructor, we
  // can't extend its lifetime by capturing a shared_ptr and increasing its
  // refcount.
  std::unique_lock<std::mutex> guard(mutex_);
  closeHoldingMutex();
}

void Connection::Impl::closeHoldingMutex() {
  if (inboxReactorToken_.has_value()) {
    reactor_->remove(inboxReactorToken_.value());
    inboxReactorToken_.reset();
  }
  if (outboxReactorToken_.has_value()) {
    reactor_->remove(outboxReactorToken_.value());
    outboxReactorToken_.reset();
  }
  if (socket_) {
    loop_->unregisterDescriptor(socket_->fd());
    socket_.reset();
  }

  leak_.reset();
}

} // namespace shm
} // namespace transport
} // namespace tensorpipe
