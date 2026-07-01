/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * Compile-time shim only — no graph logic.
 * xllm-musa style: ATen/cuda headers + musamapping custom_defines (#define cuda musa).
 * Do not stub-out MUSAGraphsC10Utils; CUDAGraph.h maps to MUSAGraph and needs
 * MempoolId_t / CaptureId_t from that header.
 */
#pragma once

#if defined(XLLM_TORCH_MUSA)
#include <musa_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
namespace xllm_device_graph = at::cuda;
#else
#include <ATen/cuda/CUDAGraph.h>
#include <cuda_runtime.h>
namespace xllm_device_graph = at::cuda;
#endif
