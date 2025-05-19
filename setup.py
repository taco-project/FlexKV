import os
from setuptools import find_packages, setup
from Cython.Build import cythonize
from torch.utils import cpp_extension

# Set build directory for all generated files
build_dir = os.path.abspath("build")
os.makedirs(build_dir, exist_ok=True)

ext_modules = [
    cpp_extension.CUDAExtension(
        name="flexkv.c_ext",
        sources=[
            "csrc/bindings.cpp",
            "csrc/transfer.cu",
            "csrc/hash.cpp",
            "csrc/index.cpp",
        ],
        library_dirs=[os.path.join(build_dir, "lib")],
        include_dirs=[os.path.join(build_dir, "include")],
        extra_compile_args={"nvcc": ["-O3"], "cxx": ["-std=c++17"]},
        extra_link_args=["-lcuda", "-lssl", "-lcrypto", "-lxxhash"],
    ),
]

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

# 使用 cythonize 编译 Python 模块
# 排除 __init__.py 文件和测试文件
python_files = ["flexkv/**/*.py"]
excluded_files = ["flexkv/**/__init__.py", "flexkv/**/test_*.py", "flexkv/**/benchmark_*.py", "flexkv/benchmark/**/*.py"]

setup(
    name="flexkv",
    description="A global KV-Cache manager for LLM inference",
    version="0.1.0",
    packages=find_packages(exclude=("benchmarks", "csrc", "examples", "tests")),
    install_requires=install_requires,
    ext_modules=ext_modules + cythonize(
        python_files,
        exclude=excluded_files,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
        build_dir=build_dir,  # Direct Cython to use the build directory
    ),
    cmdclass={
        "build_ext": cpp_extension.BuildExtension.with_options(
            no_python_abi_suffix=True,
            build_temp=os.path.join(build_dir, "temp"),  # Temporary build files
            build_lib=os.path.join(build_dir, "lib"),    # Output library files
        )
    },
    python_requires=">=3.8",
)
