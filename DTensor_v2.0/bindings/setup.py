from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11, os, subprocess

CUDA_HOME = "/usr/local/cuda"
NVCC = os.path.join(CUDA_HOME, "bin", "nvcc")
CUDA_INCLUDE = os.path.join(CUDA_HOME, "include")
CUDA_LIB64 = os.path.join(CUDA_HOME, "lib64")


class BuildExtNVCC(build_ext):
    def build_extensions(self):
        super().build_extensions()
        build_temp = self.build_temp
        os.makedirs(os.path.join(build_temp, "../tensor"), exist_ok=True)

        # --- Step 1: compile dtensor.cpp manually ---
        dtensor_cpp = "../tensor/dtensor.cpp"
        dtensor_obj = os.path.join(build_temp, "../tensor/dtensor.o")
        compile_cmd = [
            "g++",
            "-c", dtensor_cpp,
            "-O3", "-std=c++17", "-fPIC",
            "-I..", "-I../tensor", "-I../bridge", "-I../ckpt", "-I../memory",
            "-I../process_group", "-I../Tensor-Implementations/include",
            "-I../Tensor-Implementations/src",
            f"-I{pybind11.get_include()}",
            f"-I{CUDA_INCLUDE}",
            "-I/usr/lib/x86_64-linux-gnu/openmpi/include",
            "-I/usr/include/python3.10",
            "-o", dtensor_obj
        ]
        print(f"[C++] {' '.join(compile_cmd)}")
        subprocess.check_call(compile_cmd)

        # --- Step 2: paths for other important objects ---
        allocator_obj = os.path.join(build_temp, "../memory/cachingAllocator.o")
        process_group_obj = os.path.join(build_temp, "../process_group/process_group_nccl.o")
        bridge_obj = os.path.join(build_temp, "../bridge/tensor_ops_bridge.o")

        # --- Step 3: link everything with NVCC ---
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            objs = []
            for root, _, files in os.walk(build_temp):
                for f in files:
                    if f.endswith(".o") and f not in [
                        "dtensor.o", "process_group_nccl.o",
                        "cachingAllocator.o", "tensor_ops_bridge.o"
                    ]:
                        objs.append(os.path.join(root, f))

            cmd = [
                NVCC,
                "-shared",
                "-Xcompiler", "-fPIC",
                "-o", ext_path,
                "-Xlinker", "--whole-archive",
                dtensor_obj,
                bridge_obj,          #  TensorOpsBridge linkage
                process_group_obj,   #  ProcessGroup linkage
                allocator_obj,       #  CachingAllocator linkage
                "-Xlinker", "--no-whole-archive",
                *objs,
                "../Tensor-Implementations/lib/libtensor.a",
                "-L/usr/lib/x86_64-linux-gnu/openmpi/lib",
                "-L/usr/local/cuda/lib64",
                "-lcudart", "-lcublas", "-lnccl", "-lmpi", "-lmpi_cxx",
                "-Xlinker", "--no-as-needed",
            ]
            print(f"[NVCC-LINK] {' '.join(cmd)}")
            subprocess.check_call(cmd)


# ---------------- Setup configuration ----------------
include_dirs = [
    "..",
    pybind11.get_include(),
    "../tensor",
    "../bridge",
    "../ckpt",
    "../memory",
    "../process_group",
    "../Tensor-Implementations/include",
    "../Tensor-Implementations/src",
    CUDA_INCLUDE,
    "/usr/lib/x86_64-linux-gnu/openmpi/include",
]

library_dirs = [
    CUDA_LIB64,
    "/usr/lib/x86_64-linux-gnu/openmpi/lib",
]

libraries = ["cudart", "cublas", "mpi", "mpi_cxx", "nccl"]

# ✅ Added `force_link_dtensor.cpp` for safe symbol linkage
sources = [
    "bindings.cpp",
    "force_link_dtensor.cpp",
    "../bridge/tensor_ops_bridge.cpp",
    "../ckpt/ckpt.cpp",
    "../memory/cachingAllocator.cpp",
    "../process_group/process_group_nccl.cpp",
]

extra_compile_args = ["-O3", "-std=c++17", "-fPIC"]

ext_modules = [
    Extension(
        "dtensor",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        language="c++",
    )
]

setup(
    name="dtensor",
    version="2.0.0",
    description="BluBridge DTensor PyBind11 bindings with CUDA, cuBLAS, and MPI",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtNVCC},
)
