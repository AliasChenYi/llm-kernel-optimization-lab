import pytest
import torch

from llm_kernel_lab.ops import softmax_ref, softmax_triton


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 1e-5, 1e-5), (torch.float16, 2e-3, 2e-3)])
@pytest.mark.parametrize("shape", [(32, 512), (16, 2048), (8, 4096)])
@pytest.mark.parametrize("variant", ["default", "exp2"])
def test_softmax_triton_matches_reference(dtype, atol, rtol, shape, variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    pytest.importorskip("triton")

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)

    actual = softmax_triton(x, variant=variant)
    expected = softmax_ref(x)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
