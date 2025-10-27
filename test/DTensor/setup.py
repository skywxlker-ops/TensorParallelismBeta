
# DTensor/setup.py
from setuptools import setup, Extension
import os
import pybind11

# Get CUDA path
def find_cuda():
    cuda_path = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(cuda_path):
        cuda_path = '/usr/local/cuda'
    return cuda_path

cuda_path = find_cuda()

# Get NCCL path  
def find_nccl():
    nccl_path = os.getenv('NCCL_HOME', '/usr/local/nccl')
    if not os.path.exists(nccl_path):
        nccl_path = '/usr'
    return nccl_path

nccl_path = find_nccl()

ext_modules = [
    Extension(
        "dtensor",
        sources=[
            "dtensor_wrapper.cpp",
            "dtensor_core.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            os.path.join(cuda_path, 'include'),
            os.path.join(nccl_path, 'include'),
        ],
        library_dirs=[
            os.path.join(cuda_path, 'lib64'),
            os.path.join(nccl_path, 'lib'),
        ],
        libraries=['cudart', 'nccl'],
        language='c++',
        extra_compile_args=[
            '-std=c++14',
            '-O3',
            '-Wall',
            '-Wextra'
        ],
        extra_link_args=[
            f'-L{os.path.join(cuda_path, "lib64")}',
            f'-L{os.path.join(nccl_path, "lib")}',
        ],
    ),
]

setup(
    name="dtensor",
    version="0.1.0",
    author="Your Name",
    description="DTensor Python Bindings",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7",
)