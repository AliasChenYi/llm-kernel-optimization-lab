import pytest
import torch

from llm_kernel_lab.ops import rms_norm_ref, rms_norm_triton


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 1e-5, 1e-5), (torch.float16, 2e-3, 2e-3)])
@pytest.mark.parametrize("shape", [(16, 1024), (8, 4096), (2, 8192)])
def test_rms_norm_triton_matches_reference(dtype, atol, rtol, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    pytest.importorskip("triton")

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)

    actual = rms_norm_triton(x, weight)
    expected = rms_norm_ref(x, weight)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

