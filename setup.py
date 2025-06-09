import os
import sys
from setuptools import find_packages, setup
from torch.utils import cpp_extension
from Cython.Build import cythonize

# Set build directory for all generated files
build_dir = os.path.abspath("build")
os.makedirs(build_dir, exist_ok=True)

# Check if we're in debug mode using environment variable
debug = os.environ.get("FLEXKV_DEBUG") == "1"
if debug:
    print("Running in debug mode - Cython compilation disabled")

enable_cfs = os.environ.get("FLEXKV_ENABLE_CFS", "0") == "1"

# Define C++ extensions
cpp_sources = [
    "csrc/bindings.cpp",
    "csrc/transfer.cu",
    "csrc/hash.cpp",
    "csrc/tp_transfer_thread_group.cpp",
    "csrc/transfer_ssd.cpp",
]

extra_link_args = ["-lcuda", "-lxxhash", "-lpthread", "-lrt"]
extra_compile_args = ["-std=c++17"]
include_dirs = [os.path.join(build_dir, "include")]

if enable_cfs:
    print("ENABLE_CFS = true: compiling and link cfs related content")
    cpp_sources.append("csrc/pcfs/pcfs.cpp")
    extra_link_args.append("-lhifs_client_sdk")
    extra_compile_args.append("-DFLEXKV_ENABLE_CFS")

cpp_extensions = [
    cpp_extension.CUDAExtension(
        name="flexkv.c_ext",
        sources=cpp_sources,
        library_dirs=[os.path.join(build_dir, "lib")],
        include_dirs=include_dirs,
        extra_compile_args={"nvcc": ["-O3"], "cxx": extra_compile_args},
        extra_link_args=extra_link_args,
    ),
]

# Initialize ext_modules with C++ extensions
ext_modules = cpp_extensions

# Only use Cython in release mode
if not debug:
    # Compile Python modules with cythonize
    # Exclude __init__.py files and test files
    python_files = ["flexkv/**/*.py"]
    excluded_files = ["flexkv/**/__init__.py",
                      "flexkv/**/test_*.py",
                      "flexkv/**/benchmark_*.py",
                      "flexkv/benchmark/**/*.py",
                      "flexkv/benchmark/test_kvmanager.py"]
    cythonized_modules = cythonize(
        python_files,
        exclude=excluded_files,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
        build_dir=build_dir,  # Direct Cython to use the build directory
    )
    # Add Cython modules to ext_modules
    ext_modules.extend(cythonized_modules)
    print("Release mode: Including Cython compilation")
else:
    print("Debug mode: Skipping Cython compilation")

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="flexkv",
    description="A global KV-Cache manager for LLM inference",
    version="0.1.0",
    packages=find_packages(exclude=("benchmarks", "csrc", "examples", "tests")),
    install_requires=install_requires,
    ext_modules=ext_modules,  # Now contains both C++ and Cython modules as needed
    cmdclass={
        "build_ext": cpp_extension.BuildExtension.with_options(
            no_python_abi_suffix=True,
            build_temp=os.path.join(build_dir, "temp"),  # Temporary build files
            build_lib=os.path.join(build_dir, "lib"),    # Output library files
        )
    },
    python_requires=">=3.8",
)
