from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="llm_kernel_lab_cuda",
    ext_modules=[
        CUDAExtension(
            name="llm_kernel_lab_cuda",
            sources=["rmsnorm.cpp", "rmsnorm_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-gencode=arch=compute_70,code=sm_70"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

