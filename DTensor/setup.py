from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "simple_dtensor",
        sources=["simple_wrapper.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3'],
    ),
]

setup(
    name="simple_dtensor",
    ext_modules=ext_modules,
    zip_safe=False,
)