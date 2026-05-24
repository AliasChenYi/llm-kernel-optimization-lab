#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void rmsnorm_kernel(const scalar_t* __restrict__ x,
                               const scalar_t* __restrict__ weight,
                               scalar_t* __restrict__ y,
                               int rows,
                               int hidden,
                               float eps) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    float value = static_cast<float>(x[row * hidden + col]);
    sum += value * value;
  }

  shared[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(shared[0] / hidden + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    float value = static_cast<float>(x[row * hidden + col]);
    float gamma = static_cast<float>(weight[col]);
    y[row * hidden + col] = static_cast<scalar_t>(value * inv_rms * gamma);
  }
}

torch::Tensor rmsnorm_cuda_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  const auto hidden = static_cast<int>(x.size(-1));
  const auto rows = static_cast<int>(x.numel() / hidden);
  auto y = torch::empty_like(x);

  const int threads = 256;
  const int shared_bytes = threads * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rmsnorm_cuda_forward", [&] {
    rmsnorm_kernel<scalar_t><<<rows, threads, shared_bytes, stream>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        rows,
        hidden,
        static_cast<float>(eps));
  });

  return y;
}

