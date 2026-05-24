#include <torch/extension.h>

torch::Tensor rmsnorm_cuda_forward(torch::Tensor x, torch::Tensor weight, double eps);

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(x.size(-1) == weight.numel(), "weight length must match x.size(-1)");
  return rmsnorm_cuda_forward(x, weight, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
}

