import importlib.metadata
import importlib.util
import os
import shutil
from pathlib import Path
from typing import NamedTuple


from setuptools import find_packages, setup

try:
    from torch.utils import cpp_extension
except ImportError as exc:  # pragma: no cover - build-time diagnostic
    raise RuntimeError(
        "flexkv builds a C++/CUDA extension against your installed PyTorch, "
        "so torch must be importable at build time.\n"
        "  1. Install the torch build you intend to run with.\n"
        "  2. Build with isolation disabled so that same torch is used:\n"
        "       pip install --no-build-isolation flexkv\n"
        "torch is deliberately not a build requirement: an isolated build "
        "environment would link flexkv against a different libtorch than the "
        "one loaded at runtime."
    ) from exc


class NvcompInfo(NamedTuple):
    include_dirs: list
    lib_dir: str
    link_name: str
    source: str


NVCOMP_SOURCES = [
    "csrc/compression/common/packed_ssd.cpp",
    "csrc/compression/common/transfer_ssd_packed.cpp",
    "csrc/compression/common/common_bindings.cpp",
    "csrc/compression/ans/nvcomp_ans.cu",
    "csrc/compression/ans/nvcomp_ans_tp.cpp",
    "csrc/compression/ans/ans_bindings.cpp",
]

NVCOMP_HEADERS = [
    "csrc/compression/common/staging_transfer.cuh",
    "csrc/compression/common/packed_ssd.h",
    "csrc/compression/common/transfer_ssd_packed.h",
    "csrc/compression/ans/nvcomp_ans.cuh",
    "csrc/compression/ans/nvcomp_ans_tp.h",
]


# Mainstream datacenter + workstation architectures we want the shipped
# c_ext.so to run on out of the box (Ampere -> Hopper -> Blackwell). The final
# list is intersected with what the local nvcc actually supports, so this stays
# buildable on older CUDA toolkits that lack sm_100/sm_120.
MAINSTREAM_ARCHS = ["8.0", "8.6", "8.9", "9.0", "10.0", "12.0"]


def _nvcc_supported_archs():
    """Return the set of 'major.minor' arches the local nvcc can target.

    Parses ``nvcc --list-gpu-arch`` (lines like ``compute_90``). Returns an
    empty set if nvcc is unavailable, in which case callers should not filter."""
    import re
    import shutil
    import subprocess
    nvcc = shutil.which("nvcc") or os.path.join(
        os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
    try:
        out = subprocess.run([nvcc, "--list-gpu-arch"],
                             capture_output=True, text=True, check=True).stdout
    except Exception as e:
        print(f"Could not query nvcc for supported arches: {e}")
        return set()
    archs = set()
    for m in re.finditer(r"compute_(\d+)", out):
        code = m.group(1)  # e.g. "90" -> 9.0, "100" -> 10.0, "120" -> 12.0
        archs.add(f"{int(code[:-1])}.{code[-1]}")
    return archs


def detect_cuda_arch():
    """Return a semicolon-separated TORCH_CUDA_ARCH_LIST.

    By default we build a *multi-arch* binary covering mainstream datacenter and
    workstation GPUs so a single c_ext.so is portable across machines (this
    avoids the "no kernel image is available for execution on the device" error
    that a single-arch build hits when moved to a different GPU). The mainstream
    set is filtered to what the local nvcc supports, the locally-detected arch is
    always added, and +PTX is appended to the newest arch for forward-compat JIT
    onto future GPUs."""
    supported = _nvcc_supported_archs()

    # Start from the mainstream set, filtered by what nvcc can actually build.
    archs = {a for a in MAINSTREAM_ARCHS if not supported or a in supported}

    # Always cover the GPU(s) present on the build host, even if not mainstream.
    local = set()
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                local.add(f"{major}.{minor}")
    except Exception as e:
        print(f"GPU architecture auto-detection failed: {e}")
    archs |= {a for a in local if not supported or a in supported}

    if not archs:
        # nvcc query failed AND no torch/GPU: fall back to a broad static list.
        fallback = "8.0;8.6;9.0"
        print(f"No arch info available, using fallback architectures: {fallback}")
        return fallback

    ordered = sorted(archs, key=lambda a: tuple(int(x) for x in a.split(".")))
    # Emit PTX for the newest arch so unknown future GPUs can JIT from PTX.
    arch_list = ";".join(ordered[:-1] + [f"{ordered[-1]}+PTX"])
    print(f"Building for architectures: {arch_list} "
          f"(mainstream default + local {sorted(local) or 'none'})")
    return arch_list


def _probe_nvcomp_root(root, source):
    root = Path(root)
    include_dirs = [
        str(path)
        for path in (root / "include", root / "build" / "include")
        if path.is_dir()
    ]
    if not any((Path(path) / "nvcomp" / "ans.h").exists()
               for path in include_dirs):
        return None

    for subdir in ("build/lib", "lib/x86_64-linux-gnu", "lib64", "lib", ""):
        lib_dir = root / subdir if subdir else root
        if not lib_dir.is_dir():
            continue
        if (lib_dir / "libnvcomp.so").exists():
            return NvcompInfo(include_dirs, str(lib_dir), "nvcomp", source)
        versioned = sorted(lib_dir.glob("libnvcomp.so.*"))
        if versioned:
            return NvcompInfo(
                include_dirs,
                str(lib_dir),
                ":" + versioned[-1].name,
                source,
            )
    return None


def _cuda_major():
    """Return the CUDA major version used by PyTorch, when available."""
    try:
        import torch
        if torch.version.cuda:
            return int(torch.version.cuda.split(".", 1)[0])
    except (ImportError, TypeError, ValueError):
        pass
    return None


def _pip_nvcomp_roots(cuda_major):
    """Yield nvCOMP roots installed by either supported NVIDIA wheel."""
    supported_majors = (cuda_major,) if cuda_major in (12, 13) else (13, 12)
    seen = set()

    # The standalone C++ wheels can be namespace packages, for which
    # find_spec("nvidia.nvcomp").origin is None. CUDA 13 wheels use
    # nvidia/libnvcomp while older wheels may use nvidia/nvcomp.
    for major in supported_majors:
        for dist_name in (
                f"nvidia-libnvcomp-cu{major}",
                f"nvidia-nvcomp-cu{major}"):
            try:
                dist = importlib.metadata.distribution(dist_name)
            except importlib.metadata.PackageNotFoundError:
                continue
            for package_dir in ("nvidia/libnvcomp", "nvidia/nvcomp"):
                root = Path(dist.locate_file(package_dir))
                if root not in seen:
                    seen.add(root)
                    yield root, f"pip {dist_name} ({root})"

    if cuda_major not in (12, 13):
        # Retain compatibility with older wheels that expose an importable
        # module but may not have one of the current distribution names.
        spec = importlib.util.find_spec("nvidia.nvcomp")
        if spec:
            roots = list(spec.submodule_search_locations or ())
            if spec.origin:
                roots.append(os.path.dirname(spec.origin))
            for root in map(Path, roots):
                if root not in seen:
                    seen.add(root)
                    yield root, f"pip nvidia.nvcomp ({root})"


def _find_nvcomp(nvcomp_root):
    """Locate public nvcomp headers and library.

    Probing priority:
      1. NVCOMP_ROOT (error if set but not usable; no silent fallback).
      2. CUDA-matched nvidia-libnvcomp-cu{12,13} / nvidia-nvcomp-cu{12,13}.
      3. System /usr.
    """
    if nvcomp_root:
        if not os.path.exists(nvcomp_root):
            raise ValueError(f"NVCOMP_ROOT={nvcomp_root} does not exist")
        result = _probe_nvcomp_root(nvcomp_root, f"NVCOMP_ROOT={nvcomp_root}")
        if not result:
            raise ValueError(
                f"NVCOMP_ROOT={nvcomp_root} does not contain a usable nvcomp "
                "install (need include/nvcomp/ans.h and libnvcomp.so*)"
            )
        return result

    cuda_major = _cuda_major()
    for pip_root, source in _pip_nvcomp_roots(cuda_major):
        result = _probe_nvcomp_root(pip_root, source)
        if result:
            return result

    result = _probe_nvcomp_root("/usr", "system (/usr)")
    if result:
        return result

    cuda_suffix = str(cuda_major) if cuda_major in (12, 13) else "{12|13}"
    raise ValueError(
        "nvcomp not found. Install the package matching this CUDA toolkit:\n"
        f"  pip install nvidia-libnvcomp-cu{cuda_suffix}   (C++ library)\n"
        f"  pip install nvidia-nvcomp-cu{cuda_suffix}      (Python API + C++ library)\n"
        "Or install a system/distro package, or set "
        "NVCOMP_ROOT=/path/to/nvcomp manually."
    )


def _enable_nvcomp_build(cpp_sources, hpp_sources, include_dirs, library_dirs,
                         extra_link_args, extra_compile_args,
                         nvcc_compile_args):
    nvcomp = _find_nvcomp(os.environ.get("NVCOMP_ROOT"))
    print(f"ENABLE_NVCOMP = true: Compiling with nvcomp ANS support "
          f"(source={nvcomp.source}, lib={nvcomp.lib_dir})")

    cpp_sources.extend(NVCOMP_SOURCES)
    hpp_sources.extend(NVCOMP_HEADERS)
    include_dirs.extend(nvcomp.include_dirs)
    library_dirs.append(nvcomp.lib_dir)
    extra_link_args.extend([
        f"-l{nvcomp.link_name}",
        f"-Wl,-rpath,{nvcomp.lib_dir}",
    ])
    extra_compile_args.append("-DFLEXKV_ENABLE_NVCOMP")
    nvcc_compile_args.append("-DFLEXKV_ENABLE_NVCOMP")

VERSION_FILE = Path(__file__).resolve().parent / "flexkv" / "_version.py"


def _version_from_git():
    import subprocess
    try:
        # e.g. "v1.0.0-0-gabc1234" or "v1.0.0-3-gabc1234"
        raw = subprocess.check_output(
            ["git", "describe", "--tags", "--long", "--match", "v*"],
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
        # parse: v1.0.0-<distance>-g<hash>
        parts = raw.rsplit("-", 2)
        if len(parts) != 3:
            raise ValueError(f"Unexpected git describe output format: {raw!r}")
        tag, distance, git_hash = parts
        tag = tag.lstrip("v")
        if distance == "0":
            return tag  # clean release
        else:
            return f"{tag}+git{git_hash[1:]}"  # dev build
    except Exception:
        return None


def _version_from_file():
    """Read the version baked into the tree by a previous sdist build."""
    if not VERSION_FILE.is_file():
        return None
    namespace = {}
    exec(VERSION_FILE.read_text(), namespace)  # noqa: S102
    return namespace.get("__version__") or None


def get_version():
    """Resolve the version, in descending order of authority.

    ``FLEXKV_VERSION`` wins so a release job can stamp an exact PEP 440
    version. Otherwise fall back to git, and then to flexkv/_version.py --
    the last one matters because an sdist unpacked by pip has no .git, and
    without it every source install would report 0.0.0+unknown.

    Note that a non-tagged git build yields a PEP 440 *local* version
    (``1.2.1+gabc1234``). Local versions are valid for local installs but
    PyPI rejects them, so publishing must happen from a clean tag or with
    FLEXKV_VERSION set explicitly.
    """
    return (os.environ.get("FLEXKV_VERSION")
            or _version_from_git()
            or _version_from_file()
            or "0.0.0+unknown")


def write_version_file(version):
    """Persist the resolved version so it survives into the sdist."""
    VERSION_FILE.write_text(
        '"""Generated at build time by setup.py -- do not edit."""\n'
        f'__version__ = "{version}"\n'
    )


def get_git_commit():
    commit = os.environ.get("FLEXKV_GIT_COMMIT") or os.environ.get("GITHUB_SHA")
    if not commit:
        import subprocess
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            ).decode().strip()
        except Exception:
            return "unknown"

    commit = commit.strip().lower()
    if not commit or any(char not in "0123456789abcdef" for char in commit):
        return "unknown"
    return commit[:12]


build_git_commit = get_git_commit()

build_dir = "build"
os.makedirs(build_dir, exist_ok=True)


def ensure_cmake_prerequisites():
    """Build the CMake half of the tree if it has not been built already.

    ``flexkv.c_ext`` links ``-lxxhash`` against ``build/lib`` and includes
    ``build/include``, both of which are CMake outputs. build.sh runs CMake
    before setup.py, so a developer build already has them. A user running
    ``pip install flexkv`` invokes setup.py directly and would otherwise fail
    at link time with ``cannot find -lxxhash``, so run CMake here when the
    outputs are absent. An existing build/ is left alone: re-running CMake on
    a developer tree would discard the configuration build.sh chose.
    """
    import subprocess

    lib_dir = Path(build_dir) / "lib"
    if lib_dir.is_dir() and any(lib_dir.glob("libxxhash.so*")):
        return

    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError(
            "cmake is required to build flexkv from source but was not found "
            "on PATH. Install it (e.g. `pip install cmake` or "
            "`apt-get install cmake`) and retry."
        )

    cmake_args = [cmake, os.path.abspath(".")]
    if not enable_metrics:
        # Mirror build.sh: without metrics, prometheus-cpp is not needed, and
        # the sdist deliberately does not ship it.
        cmake_args.append("-DFLEXKV_ENABLE_MONITORING=OFF")
    print(f"=== Running CMake configuration: {' '.join(cmake_args)}")
    subprocess.check_call(cmake_args, cwd=build_dir)
    build_cmd = [cmake, "--build", "."]
    jobs = os.environ.get("MAX_JOBS")
    if jobs:
        build_cmd.extend(["-j", jobs])
    print(f"=== Building third-party libraries: {' '.join(build_cmd)}")
    subprocess.check_call(build_cmd, cwd=build_dir)


spdlog_include_dir = os.path.abspath("third_party/spdlog/include")
if not os.path.isdir(spdlog_include_dir):
    raise RuntimeError(
        "third_party/spdlog is missing; run "
        "git submodule update --init third_party/spdlog"
    )

# Check if we're in debug mode using environment variable
debug = os.environ.get("FLEXKV_DEBUG") == "1"
if debug:
    print("Running in debug mode - Cython compilation disabled")

enable_cfs = os.environ.get("FLEXKV_ENABLE_CFS", "0") == "1"
enable_gds = os.environ.get("FLEXKV_ENABLE_GDS", "0") == "1"
enable_p2p = os.environ.get("FLEXKV_ENABLE_P2P", "0") == "1"
enable_cputest = os.environ.get("FLEXKV_ENABLE_CPUTEST", "0") == "1"
enable_nvcomp = os.environ.get("FLEXKV_ENABLE_NVCOMP", "0") == "1"
# FLEXKV_ENABLE_METRICS=0: build without Prometheus (no prometheus-cpp dependency)
enable_metrics = os.environ.get("FLEXKV_ENABLE_METRICS", "0") == "1"

ensure_cmake_prerequisites()

# Define C++ extensions (base: no dist/Redis)
cpp_sources = [
    "csrc/bindings.cpp",
    "csrc/logging.cpp",
    "csrc/transfer.cu",  # Skip CUDA file for now
    "csrc/ce_transfer.cu",
    "csrc/hash.cpp",
    "csrc/tp_transfer_thread_group.cpp",
    "csrc/device_thread_pool.cpp",
    "csrc/transfer_backend.cpp",
    "csrc/region_batch.cpp",
    "csrc/layer_notify.cpp",
    "csrc/transfer_ssd.cpp",
    "csrc/radix_tree.cpp",
    "csrc/eviction_strategy.cpp",
    "csrc/monitoring/metrics_manager.cpp",  # Monitoring support
]

hpp_sources = [
    "csrc/logging.h",
    "csrc/cache_utils.h",
    "csrc/tp_transfer_thread_group.h",
    "csrc/device_thread_pool.h",
    "csrc/transfer_backend.h",
    "csrc/region_batch.h",
    "csrc/layer_notify.h",
    "csrc/transfer_ssd.h",
    "csrc/radix_tree.h",
    "csrc/eviction_strategy.h",
    "csrc/ce_transfer.h",
    "csrc/monitoring/metrics_manager.h",  # Monitoring support
]

# extra_link_args: dist/Redis (libhiredis) only when FLEXKV_ENABLE_P2P=1
lib_dir = os.path.join(build_dir, "lib")
library_dirs = [lib_dir]
extra_link_args = ["-lcuda", "-lxxhash", "-lpthread", "-lrt", "-luring"]
if enable_p2p:
    extra_link_args.append("-lhiredis")

if enable_cputest:
    extra_link_args.remove("-lcuda")
    # Set TORCH_CUDA_ARCH_LIST to avoid IndexError when no GPU is available
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;9.0"


# Prometheus libraries only when metrics enabled
if enable_metrics:
    extra_link_args.extend(["-lprometheus-cpp-pull", "-lprometheus-cpp-core"])
else:
    print("FLEXKV_ENABLE_METRICS=0: building without Prometheus monitoring")
# Auto-detect GPU architecture if TORCH_CUDA_ARCH_LIST is not explicitly set
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = detect_cuda_arch()
print(f"TORCH_CUDA_ARCH_LIST = {os.environ['TORCH_CUDA_ARCH_LIST']}")

extra_compile_args = [
    "-std=c++17",
    "-O3",
    f'-DFLEXKV_GIT_COMMIT="{build_git_commit}"',
]
if enable_metrics:
    extra_compile_args.append("-DFLEXKV_ENABLE_MONITORING")
include_dirs = [
    os.path.abspath(os.path.join(build_dir, "include")),
    os.path.abspath("csrc"),
    spdlog_include_dir,
]

# Add rpath to find libraries at runtime
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
if enable_nvcomp:
    _enable_nvcomp_build(cpp_sources, hpp_sources, include_dirs, library_dirs,
                         extra_link_args, extra_compile_args,
                         nvcc_compile_args)
else:
    print("ENABLE_NVCOMP = false: Skipping nvcomp ANS compression")
if not enable_gds:
    print("ENABLE_GDS = false: Skipping GDS code")
if not enable_p2p:
    print("ENABLE_P2P = false: Skipping distributed (P2P/Redis) code; no libhiredis or Redis deps required")

cpp_extensions = [
    cpp_extension.CUDAExtension(
        name="flexkv.c_ext",
        sources=cpp_sources,
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        depends=hpp_sources,
        extra_compile_args={"nvcc": nvcc_compile_args, "cxx": extra_compile_args},
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
                      "flexkv/benchmark/test_kvmanager.py",
                      # A one-line version literal; compiling it just produces
                      # a second copy of the same constant.
                      "flexkv/_version.py"]
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

# Runtime dependencies only. requirements.txt is the *development* environment
# and additionally lists build tools (setuptools, Cython) and test tools
# (pytest, pytest-benchmark); installing those into a user's environment as a
# side effect of `pip install flexkv` would be wrong.
#
# torch is deliberately absent even though flexkv imports it. flexkv.c_ext is
# linked against a specific libtorch ABI, so the torch that must be present is
# the one this was compiled against -- not whichever version a resolver would
# pick. Declaring it would also let pip silently replace the torch that the
# host inference engine (sglang/vLLM) depends on. torch must be installed
# first; setup.py imports it at build time and fails clearly if it is not.
INSTALL_REQUIRES = [
    "numpy>=1.20.0",
    "pyzmq>=22.0.0",
    "psutil>=5.8.0",
    "nvtx>=0.2.8",
    "pyyaml>=5.4.0",
    "expiring-dict==1.1.2",
    "redis>=4.0.0",
    "requests>=2.0.0",
]

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

version = get_version()
write_version_file(version)

setup(
    name="flexkv",
    description="A global KV-Cache manager for LLM inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version,
    license="Apache-2.0",
    url="https://github.com/taco-project/FlexKV",
    project_urls={
        "Source": "https://github.com/taco-project/FlexKV",
        "Issues": "https://github.com/taco-project/FlexKV/issues",
        "Changelog": "https://github.com/taco-project/FlexKV/blob/main/CHANGELOG.md",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=("benchmarks", "csrc", "examples", "tests")),
    package_data={
        "flexkv": ["*.so", "lib/*.so", "lib/*.so.*"],
    },
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-benchmark>=3.0.0",
        ],
    },
    ext_modules=ext_modules,  # Now contains both C++ and Cython modules as needed
    cmdclass={
        "build_ext": CustomBuildExt.with_options(
            include_dirs=os.path.join(build_dir, "include"),  # Include directory for xxhash
            no_python_abi_suffix=True,
            build_temp=os.path.join(build_dir, "temp"),  # Temporary build files
        )
    },
    # CI builds and tests on 3.10. The codebase uses syntax and typing features
    # that predate nothing older than 3.9, but 3.6/3.7 are long EOL and were
    # never tested; claiming them only converts a clean resolver error into a
    # SyntaxError after download.
    python_requires=">=3.9",
    entry_points={"console_scripts": ["flexkv=flexkv.cli.main:main"]},
)
