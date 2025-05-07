import os

from setuptools import find_packages, setup
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CUDAExtension(
        name="flexkv.c_ext",
        sources=[
            "csrc/bindings.cpp",
            "csrc/transfer.cu",
            "csrc/hash.cpp",
            "csrc/index.cpp",
        ],
        library_dirs=[os.path.abspath("build/lib")],
        include_dirs=[os.path.abspath("build/include")],
        extra_compile_args={"nvcc": ["-O3"], "cxx": ["-std=c++17"]},
        extra_link_args=["-lcuda", "-lssl", "-lcrypto", "-lxxhash"],
    ),
]

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="flexkv",
    description="A global KV-Cache manager for LLM inference",
    version="0.1.0",
    packages=find_packages(exclude=("benchmarks", "csrc", "examples", "tests")),
    install_requires=install_requires,
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    python_requires=">=3.8",
)
