import os
import shutil
import sys


from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext as _setuptools_build_ext
from setuptools.extension import Extension

enable_musa = os.environ.get("FLEXKV_USE_MUSA", "0").strip() == "1"

_has_cuda = (
    bool(os.environ.get("CUDA_HOME"))
    or bool(os.environ.get("CUDA_PATH"))
    or shutil.which("nvcc") is not None
)
build_cuda_ext = _has_cuda or not enable_musa

if build_cuda_ext:
    from torch.utils import cpp_extension as _cpp_ext
else:
    _cpp_ext = None


def get_version():
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        return f.read().strip()

build_dir = "build"
os.makedirs(build_dir, exist_ok=True)

debug = os.environ.get("FLEXKV_DEBUG") == "1"
if debug:
    print("Running in debug mode - Cython compilation disabled")

enable_cfs = os.environ.get("FLEXKV_ENABLE_CFS", "0") == "1"
enable_gds = os.environ.get("FLEXKV_ENABLE_GDS", "0") == "1"
enable_p2p = os.environ.get("FLEXKV_ENABLE_P2P", "0") == "1"
enable_cputest = os.environ.get("FLEXKV_ENABLE_CPUTEST", "0") == "1"
enable_metrics = os.environ.get("FLEXKV_ENABLE_METRICS", "1") != "0"

lib_dir = os.path.join(build_dir, "lib")
common_compile_args = ["-std=c++17"]

cpp_extensions = []

if build_cuda_ext:
    cpp_sources = [
        "csrc/bindings.cpp",
        "csrc/transfer.cu",
        "csrc/hash.cpp",
        "csrc/tp_transfer_thread_group.cpp",
        "csrc/transfer_ssd.cpp",
        "csrc/radix_tree.cpp",
        "csrc/monitoring/metrics_manager.cpp",
    ]
    hpp_sources = [
        "csrc/cache_utils.h",
        "csrc/tp_transfer_thread_group.h",
        "csrc/transfer_ssd.h",
        "csrc/radix_tree.h",
        "csrc/monitoring/metrics_manager.h",
    ]
    extra_link_args = ["-lcuda", "-lxxhash", "-lpthread", "-lrt", "-luring"]
    if enable_p2p:
        extra_link_args.append("-lhiredis")
    if enable_cputest:
        extra_link_args.remove("-lcuda")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;9.0"
    if enable_metrics:
        extra_link_args.extend(["-lprometheus-cpp-pull", "-lprometheus-cpp-core"])
    else:
        print("FLEXKV_ENABLE_METRICS=0: building without Prometheus monitoring")
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;9.0"

    extra_compile_args = list(common_compile_args)
    if enable_metrics:
        extra_compile_args.append("-DFLEXKV_ENABLE_MONITORING")
    include_dirs = [os.path.abspath(os.path.join(build_dir, "include")), os.path.abspath("third_party/xxHash")]

    if os.path.exists(lib_dir):
        extra_link_args.extend([f"-Wl,-rpath,{lib_dir}", "-Wl,-rpath,$ORIGIN"])
        extra_link_args.append("-Wl,-rpath,$ORIGIN/../lib")

    if enable_cfs:
        print("ENABLE_CFS = true: compiling and link cfs related content")
        cpp_sources.append("csrc/pcfs/pcfs.cpp")
        hpp_sources.append("csrc/pcfs/pcfs.h")
        extra_link_args.append("-lhifs_client_sdk")
        extra_compile_args.append("-DFLEXKV_ENABLE_CFS")
    extra_compile_args.append("-DCUDA_AVAILABLE")

    nvcc_compile_args = ["-O3"]
    if enable_metrics:
        nvcc_compile_args.append("-DFLEXKV_ENABLE_MONITORING")
    if enable_gds:
        print("ENABLE_GDS = true: Compiling and linking GDS content")
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
    if enable_p2p:
        print("ENABLE_P2P = true: Compiling and linking distributed (P2P/Redis) content")
        cpp_sources.extend([
            "csrc/dist/distributed_radix_tree.cpp",
            "csrc/dist/local_radix_tree.cpp",
            "csrc/dist/redis_meta_channel.cpp",
            "csrc/dist/lease_meta_mempool.cpp",
        ])
        extra_compile_args.append("-DFLEXKV_ENABLE_P2P")
    if not enable_gds:
        print("ENABLE_GDS = false: Skipping GDS code")
    if not enable_p2p:
        print("ENABLE_P2P = false: Skipping distributed (P2P/Redis) code; no libhiredis or Redis deps required")

    cpp_extensions.append(
        _cpp_ext.CUDAExtension(
            name="flexkv.c_ext",
            sources=cpp_sources,
            library_dirs=[os.path.join(build_dir, "lib")],
            include_dirs=include_dirs,
            depends=hpp_sources,
            extra_compile_args={"nvcc": nvcc_compile_args, "cxx": extra_compile_args},
            extra_link_args=extra_link_args,
        ),
    )
else:
    print("MUSA-only build: skipping flexkv.c_ext (no CUDA toolkit found)")

if enable_musa:
    print("FLEXKV_USE_MUSA=1: Adding flexkv.c_ext_musa extension")

    musa_home = os.environ.get("MUSA_HOME", "")
    has_musa_sdk = bool(musa_home) and os.path.isdir(musa_home)
    has_mcc = has_musa_sdk and os.path.isfile(os.path.join(musa_home, "bin", "mcc"))

    musa_sources = [
        "csrc/musa/bindings_musa.cpp",
        "csrc/transfer_ssd.cpp",
        "csrc/hash.cpp",
        "csrc/radix_tree.cpp",
        "csrc/monitoring/metrics_manager.cpp",
    ]
    import torch
    from torch.utils.cpp_extension import include_paths
    pytorch_includes = include_paths()
    musa_include_dirs = [
        os.path.abspath("csrc"),
        os.path.abspath("csrc/musa"),
        os.path.abspath(os.path.join(build_dir, "include")),
        os.path.abspath("third_party/xxHash"),
    ]
    musa_include_dirs.extend(pytorch_includes)
    musa_compile_args = list(common_compile_args)
    musa_link_args = ["-lxxhash", "-lpthread", "-lrt", "-luring"]
    if os.path.exists(lib_dir):
        musa_link_args.extend([f"-Wl,-rpath,{lib_dir}", "-Wl,-rpath,$ORIGIN"])
        musa_link_args.append("-Wl,-rpath,$ORIGIN/../lib")

    if enable_cfs:
        musa_sources.append("csrc/pcfs/pcfs.cpp")
        musa_compile_args.append("-DFLEXKV_ENABLE_CFS")
        musa_link_args.append("-lhifs_client_sdk")

    if has_musa_sdk and has_mcc:
        print(f"  MUSA SDK detected at {musa_home} — building with mcc")
        musa_compile_args.append("-DFLEXKV_HAS_MUSA_SDK")
        musa_include_dirs.append(os.path.join(musa_home, "include"))
        musa_link_args.extend([
            f"-L{os.path.join(musa_home, 'lib')}",
            f"-Wl,-rpath,{os.path.join(musa_home, 'lib')}",
            "-lmusa",
        ])
        musa_sources.extend([
            "csrc/musa/transfer_musa.mu",
            "csrc/musa/tp_transfer_thread_group_musa.cpp",
        ])
        if enable_gds:
            musa_sources.extend([
                "csrc/musa/gds/gds_manager_musa.cpp",
                "csrc/musa/gds/tp_gds_transfer_thread_group_musa.cpp",
                "csrc/musa/gds/layout_transform_musa.mu",
            ])
            musa_compile_args.append("-DFLEXKV_ENABLE_GDS")
            musa_link_args.append("-lmufile")
    else:
        print("  MUSA SDK not found — building C++ stub only (no mcc, no MUSA runtime)")


    from torch.utils import cpp_extension

    cpp_extensions.append(
        cpp_extension.CppExtension(
            name="flexkv.c_ext_musa",
            sources=musa_sources,
            include_dirs=musa_include_dirs,
            extra_compile_args=musa_compile_args,
            extra_link_args=musa_link_args,
        )
    )
# Initialize ext_modules with C++ extensions
ext_modules = cpp_extensions

# Only use Cython in release mode
if not debug:
    python_files = ["flexkv/**/*.py"]
    excluded_files = ["flexkv/**/__init__.py",
                      "flexkv/**/test_*.py",
                      "flexkv/**/benchmark_*.py",
                      "flexkv/benchmark/**/*.py",
                      "flexkv/benchmark/test_kvmanager.py"]
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
        build_dir=build_dir,
    )
    ext_modules.extend(cythonized_modules)
    print("Release mode: Including Cython compilation")
else:
    print("Debug mode: Skipping Cython compilation")

_BuildExtBase = _cpp_ext.BuildExtension if _cpp_ext is not None else _setuptools_build_ext


class CustomBuildExt(_BuildExtBase):
    def build_extensions(self):
        for ext in self.extensions:
            mu_sources = [s for s in ext.sources if s.endswith(".mu")]
            if mu_sources:
                ext.sources = [s for s in ext.sources if not s.endswith(".mu")]
                musa_home = os.environ.get("MUSA_HOME", "")
                mcc = os.path.join(musa_home, "bin", "mcc") if musa_home else "mcc"
                for mu_src in mu_sources:
                    obj = os.path.splitext(mu_src)[0] + ".o"
                    obj_path = os.path.join(self.build_temp, obj)
                    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
                    inc_flags = []
                    for d in ext.include_dirs:
                        inc_flags.extend(["-I", d])
                    cmd = [mcc, "-c", "-O3","-fPIC", "--std=c++17",
                           "-DFLEXKV_HAS_MUSA_SDK",
                           *inc_flags, mu_src, "-o", obj_path]
                    self.spawn(cmd)
                    ext.extra_objects.append(obj_path)
        super().build_extensions()

    def run(self):
        super().run()
        self.copy_shared_libraries()

    def copy_shared_libraries(self):
        """Copy shared libraries to the package lib directory"""
        source_lib_dir = os.path.join(build_dir, "lib")
        if not os.path.exists(source_lib_dir):
            print(f"Warning: Source library directory {source_lib_dir} does not exist")
            return

        package_lib_dir = os.path.join("flexkv", "lib")
        os.makedirs(package_lib_dir, exist_ok=True)

        for file in os.listdir(source_lib_dir):
            if file.endswith(".so") or file.endswith(".so.*"):
                source_file = os.path.join(source_lib_dir, file)
                dest_file = os.path.join(package_lib_dir, file)
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, dest_file)
                    print(f"Copied {source_file} to {dest_file}")

with open("requirements.txt") as f:
    install_requires = [
        line for line in f.read().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
if enable_musa and not _has_cuda:
    install_requires = [
        r for r in install_requires
        if not r.strip().lower().startswith("nvtx")
    ]

if _cpp_ext is not None:
    _cmdclass = {
        "build_ext": CustomBuildExt.with_options(
            include_dirs=os.path.join(build_dir, "include"),
            no_python_abi_suffix=True,
            build_temp=os.path.join(build_dir, "temp"),
        )
    }
else:
    _cmdclass = {"build_ext": CustomBuildExt}

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
    ext_modules=ext_modules,
    cmdclass=_cmdclass,
    python_requires=">=3.6",
)

