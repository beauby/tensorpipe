/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <tensorpipe/config.h>

#include <tensorpipe/common/cpu_buffer.h>
#include <tensorpipe/common/defs.h>
#if TENSORPIPE_SUPPORTS_CUDA
#include <tensorpipe/common/cuda_buffer.h>
#endif // TENSORPIPE_SUPPORTS_CUDA

namespace tensorpipe {

enum class DeviceType {
  kCpu,
#if TENSORPIPE_SUPPORTS_CUDA
  kCuda,
#endif // TENSORPIPE_SUPPORTS_CUDA
};

inline std::string deviceTypeName(DeviceType type) {
  switch (type) {
  case DeviceType::kCpu:
    return "cpu";
    break;
#if TENSORPIPE_SUPPORTS_CUDA
  case DeviceType::kCuda:
    return "cuda";
    break;
#endif // TENSORPIPE_SUPPORTS_CUDA
  default:
    TP_THROW_ASSERT() << "Unrecognized device type";
    return "";
  }
}

struct Device {
  DeviceType type;
  int id;

  std::string name() const {
    return deviceTypeName(type) + std::to_string(id);
  }
};

struct Buffer {
  Buffer() {}

  /* implicit */ Buffer(CpuBuffer buffer)
      : type(DeviceType::kCpu), cpu(buffer) {}

  Buffer& operator=(CpuBuffer& buffer) {
    type = DeviceType::kCpu;
    cpu = buffer;

    return *this;
  }

#if TENSORPIPE_SUPPORTS_CUDA
  /* implicit */ Buffer(CudaBuffer buffer)
      : type(DeviceType::kCuda), cuda(buffer) {}

  Buffer& operator=(CudaBuffer& buffer) {
    type = DeviceType::kCuda;
    cuda = buffer;

    return *this;
  }

#endif // TENSORPIPE_SUPPORTS_CUDA

  DeviceType type{DeviceType::kCpu};
  union {
    CpuBuffer cpu;
#if TENSORPIPE_SUPPORTS_CUDA
    CudaBuffer cuda;
#endif // TENSORPIPE_SUPPORTS_CUDA
  };
};

template <typename TBuffer>
TBuffer unwrap(Buffer);

template <>
inline CpuBuffer unwrap(Buffer b) {
  TP_DCHECK(DeviceType::kCpu == b.type);
  return b.cpu;
}

#if TENSORPIPE_SUPPORTS_CUDA
template <>
inline CudaBuffer unwrap(Buffer b) {
  TP_DCHECK(DeviceType::kCuda == b.type);
  return b.cuda;
}
#endif // TENSORPIPE_SUPPORTS_CUDA

} // namespace tensorpipe
