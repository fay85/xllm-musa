#pragma once
#if defined(USE_MUSA)
#define EIGEN_NO_CUDA
#define EIGEN_NO_GPU
// Include only from MUSA graph TUs that also pull ATen/cuda/* compatibility
// headers.
#define TORCH_MUSA_CSRC_CORE_MUSACACHINGALLOCATOR_H_
#define TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_
#endif