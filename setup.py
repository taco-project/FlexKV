import os
import shutil
import sys


from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension

def get_version():
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        return f.read().strip()

build_dir = "build"
os.makedirs(build_dir, exist_ok=True)

# Check if we're in debug mode using environment variable
debug = os.environ.get("FLEXKV_DEBUG") == "1"
if debug:
    print("Running in debug mode - Cython compilation disabled")

enable_cfs = os.environ.get("FLEXKV_ENABLE_CFS", "0") == "1"
enable_gds = os.environ.get("FLEXKV_ENABLE_GDS", "0") == "1"

# Define C++ extensions
cpp_sources = [
    "csrc/bindings.cpp",
    "csrc/transfer.cu",  # Skip CUDA file for now
    "csrc/hash.cpp",
    "csrc/tp_transfer_thread_group.cpp",
    "csrc/transfer_ssd.cpp",
    "csrc/radix_tree.cpp",
    "csrc/layerwise.cpp"
    "csrc/dist/distributed_radix_tree.cpp",
    "csrc/dist/local_radix_tree.cpp",
    "csrc/dist/redis_meta_channel.cpp",
    "csrc/dist/lease_meta_mempool.cpp",
]

hpp_sources = [
    "csrc/cache_utils.h",
    "csrc/tp_transfer_thread_group.h",
    "csrc/transfer_ssd.h",
    "csrc/radix_tree.h",
    "csrc/layerwise.h",
]

#extra_link_args = ["-lcuda", "-lxxhash", "-lpthread", "-lrt", "-luring"]
extra_link_args = ["-lcuda", "-lxxhash", "-lpthread", "-lrt", "-luring", "-lhiredis"]
extra_compile_args = ["-std=c++17"]
include_dirs = [os.path.abspath(os.path.join(build_dir, "include"))]

# Add rpath to find libraries at runtime
lib_dir = os.path.join(build_dir, "lib")
if os.path.exists(lib_dir):
    extra_link_args.extend([f"-Wl,-rpath,{lib_dir}", "-Wl,-rpath,$ORIGIN"])
    # Also add the current package directory to rpath for installed libraries
    extra_link_args.append("-Wl,-rpath,$ORIGIN/../lib")

if enable_cfs:
    print("ENABLE_CFS = true: compiling and link cfs related content")
    cpp_sources.append("csrc/pcfs/pcfs.cpp")
    hpp_sources.append("csrc/pcfs/pcfs.h")
    extra_link_args.append("-lhifs_client_sdk")
    extra_compile_args.append("-DFLEXKV_ENABLE_CFS")
extra_compile_args.append("-DCUDA_AVAILABLE")

nvcc_compile_args = ["-O3"]
if enable_gds:
    print("ENABLE_GDS = true: Compiling and linking gds related content")
    cpp_sources.extend([
        "csrc/gds/gds_manager.cpp",
        "csrc/gds/tp_gds_transfer_thread_group.cpp",
        "csrc/gds/layout_transform.cu",
    ])
    hpp_sources.extend([
        "csrc/gds/gds_manager.h",
        "csrc/gds/tp_gds_transfer_thread_group.h",
        "csrc/gds/layout_transform.cuh",
    ])
    extra_link_args.append("-lcufile")
    extra_compile_args.append("-DFLEXKV_ENABLE_GDS")
    nvcc_compile_args.append("-DFLEXKV_ENABLE_GDS")

cpp_extensions = [
    cpp_extension.CUDAExtension(
        name="flexkv.c_ext",
        sources=cpp_sources,
        library_dirs=[os.path.join(build_dir, "lib")],
        include_dirs=include_dirs,
        depends=hpp_sources,
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
    # Import cython when debug is turned off.
    from Cython.Build import cythonize
    cythonized_modules = cythonize(
        python_files,
        exclude=excluded_files,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "profile": True,
        },
        build_dir=build_dir,  # Direct Cython to use the build directory
    )
    # Add Cython modules to ext_modules
    ext_modules.extend(cythonized_modules)
    print("Release mode: Including Cython compilation")
else:
    print("Debug mode: Skipping Cython compilation")

class CustomBuildExt(cpp_extension.BuildExtension):
    def run(self):
        super().run()
        # Copy required shared libraries to the package directory after building
        self.copy_shared_libraries()

    def copy_shared_libraries(self):
        """Copy shared libraries to the package lib directory"""
        source_lib_dir = os.path.join(build_dir, "lib")
        if not os.path.exists(source_lib_dir):
            print(f"Warning: Source library directory {source_lib_dir} does not exist")
            return

        # Create lib directory in the package
        package_lib_dir = os.path.join("flexkv", "lib")
        os.makedirs(package_lib_dir, exist_ok=True)

        # Copy all .so files
        for file in os.listdir(source_lib_dir):
            if file.endswith(".so") or file.endswith(".so.*"):
                source_file = os.path.join(source_lib_dir, file)
                dest_file = os.path.join(package_lib_dir, file)
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, dest_file)
                    print(f"Copied {source_file} to {dest_file}")

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="flexkv",
    description="A global KV-Cache manager for LLM inference",
    version=get_version(),
    packages=find_packages(exclude=("benchmarks", "csrc", "examples", "tests")),
    package_data={
        "flexkv": ["*.so", "lib/*.so", "lib/*.so.*"],
    },
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=ext_modules,  # Now contains both C++ and Cython modules as needed
    cmdclass={
        "build_ext": CustomBuildExt.with_options(
            include_dirs=os.path.join(build_dir, "include"),  # Include directory for xxhash
            no_python_abi_suffix=True,
            build_temp=os.path.join(build_dir, "temp"),  # Temporary build files
        )
    },
    #python_requires=">=3.8",
    python_requires=">=3.6",
)
