import importlib.metadata
import importlib.util
import os
import shutil
from pathlib import Path
from typing import NamedTuple


from setuptools import find_packages, setup
from torch.utils import cpp_extension


# ===========================================================================
# Platform / capability detection
# ===========================================================================
#
# FlexKV supports two classes of accelerator:
#
#   1. NVIDIA GPUs -- full functionality. nvcc compiles the custom PTX copy
#      kernels in csrc/transfer.cu, and NVTX annotations feed Nsight Systems.
#
#   2. CUDA-like accelerators without an NVIDIA GPU, notably Baidu Kunlun
#      P800. These expose a partial CUDA runtime (cudaMemcpyAsync, streams,
#      events, IPC) through a vendor toolchain, but:
#        * there is no nvcc and no PTX support, so device kernels cannot be
#          compiled at all;
#        * host<->device traffic can only use the Copy Engine, so a
#          kernel-driven H2D/D2H would be impossible even if it compiled;
#        * NVTX headers/libraries are absent.
#      For these platforms FlexKV builds "CE-only": every .cu file compiles as
#      ordinary host C++, and csrc/ce_transfer.cu (pure cudaMemcpyAsync, no
#      kernel launches) provides the sole transfer path.
#
# Both modes are produced from the same sources, gated by two macros:
#
#   FLEXKV_ENABLE_KERNEL_TRANSFER  custom PTX kernels in transfer.cu
#   FLEXKV_ENABLE_NVTX             real NVTX instead of the csrc/flexkv_nvtx.h
#                                  no-op shim
#
# Detection is automatic; both can be forced with same-named environment
# variables (set to 1/0).
# ===========================================================================


def _env_tristate(name):
    """Read an optional boolean env var.

    Returns True/False when the variable is set to a recognised value, and None
    when it is unset -- letting the caller fall back to auto-detection.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _find_nvcc():
    """Return the path to a usable nvcc, or None."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    candidate = os.path.join(
        os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
    return candidate if os.path.isfile(candidate) else None


# Python modules whose mere presence identifies a vendor accelerator that
# emulates CUDA rather than being one. These runtimes deliberately masquerade as
# CUDA -- they ship a CUDA toolkit for headers/libcudart, report a
# ``torch.version.cuda``, and even install shims named ``nvidia-smi`` -- so
# neither nvcc's presence nor torch's CUDA version can distinguish them.
#
#   torch_xmlir : Baidu Kunlun (P800 and friends), via XPU/XCCL/BKCL
#   torch_npu   : Huawei Ascend
#
# On such platforms device kernels cannot be compiled for the real accelerator,
# and host<->device traffic is Copy-Engine-only, so FlexKV must build CE-only.
_EMULATED_CUDA_MODULES = ("torch_xmlir", "torch_npu")


def _detect_emulated_cuda_backend():
    """Return the name of a detected non-NVIDIA CUDA-like backend, or None."""
    for name in _EMULATED_CUDA_MODULES:
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            # find_spec can raise for half-installed packages; treat as absent.
            continue
    return None


def _find_nvtx_header():
    """Return an include dir containing nvtx3/nvToolsExt.h, or None.

    torch always adds the CUDA include dir when building a CUDAExtension, so a
    plain existence check against the usual locations is enough; we only need to
    know whether the header is *available*, not to add a new -I flag.
    """
    roots = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        roots.append(os.path.join(cuda_home, "include"))
    roots.extend([
        "/usr/local/cuda/include",
        "/usr/include",
    ])
    try:
        import torch
        roots.append(os.path.join(os.path.dirname(torch.__file__), "include"))
    except ImportError:
        pass

    for root in roots:
        if os.path.isfile(os.path.join(root, "nvtx3", "nvToolsExt.h")):
            return root
    return None


def detect_kernel_transfer():
    """Decide whether to compile the custom PTX copy kernels.

    Three signals, checked in order of reliability:

    1. An emulated-CUDA backend (torch_xmlir / torch_npu) is installed. This is
       conclusive: the accelerator is not an NVIDIA GPU, so PTX kernels are
       useless even if nvcc happens to be available. Kunlun P800 containers do
       ship a full CUDA toolkit and a cu118 torch build, which is exactly why
       this check must come first.
    2. No nvcc -> nothing can compile device code.
    3. torch is not a CUDA build -> device code could compile but not link.
    """
    forced = _env_tristate("FLEXKV_ENABLE_KERNEL_TRANSFER")
    if forced is not None:
        print("FLEXKV_ENABLE_KERNEL_TRANSFER forced to "
              f"{int(forced)} by environment")
        return forced

    emulated = _detect_emulated_cuda_backend()
    if emulated:
        print(f"Detected {emulated} (CUDA-like non-NVIDIA accelerator) "
              "-> CE-only build (custom copy kernels disabled)")
        return False

    nvcc = _find_nvcc()
    if not nvcc:
        print("No nvcc found -> CE-only build (custom copy kernels disabled)")
        return False

    try:
        import torch
        torch_cuda = torch.version.cuda
    except ImportError:
        torch_cuda = None

    if not torch_cuda:
        print(f"nvcc found at {nvcc} but torch is not a CUDA build "
              "-> CE-only build (custom copy kernels disabled)")
        return False

    print(f"nvcc found at {nvcc} (torch CUDA {torch_cuda}) "
          "-> enabling custom copy kernels")
    return True


def detect_nvtx():
    """Decide whether to compile against real NVTX.

    NVTX is only meaningful on NVIDIA hardware: the events are consumed by
    Nsight Systems talking to the NVIDIA driver. Kunlun P800 images do ship the
    CUDA toolkit (and therefore the NVTX header), so the emulated-backend check
    has to take precedence over header discovery -- otherwise every P800 build
    would link NVTX that can never emit anything useful.
    """
    forced = _env_tristate("FLEXKV_ENABLE_NVTX")
    if forced is not None:
        print(f"FLEXKV_ENABLE_NVTX forced to {int(forced)} by environment")
        return forced

    emulated = _detect_emulated_cuda_backend()
    if emulated:
        print(f"Detected {emulated} (CUDA-like non-NVIDIA accelerator) "
              "-> using no-op NVTX shim")
        return False

    header_root = _find_nvtx_header()
    if header_root:
        print(f"NVTX header found under {header_root} -> enabling NVTX")
        return True
    print("NVTX header (nvtx3/nvToolsExt.h) not found -> using no-op NVTX shim")
    return False


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

def get_version():
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
        return "0.0.0+unknown"


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

# Platform capabilities (see the block at the top of this file).
enable_kernel_transfer = detect_kernel_transfer()
enable_nvtx = detect_nvtx()

if enable_gds and not enable_kernel_transfer:
    raise RuntimeError(
        "FLEXKV_ENABLE_GDS=1 requires the custom-kernel build: the GDS layout "
        "transform (csrc/gds/layout_transform.cu) is a CUDA kernel and needs "
        "nvcc. Disable GDS on Copy-Engine-only platforms."
    )
if enable_nvcomp and not enable_kernel_transfer:
    raise RuntimeError(
        "FLEXKV_ENABLE_NVCOMP=1 requires the custom-kernel build: nvCOMP ANS "
        "compression runs in CUDA kernels and needs nvcc plus libnvcomp. "
        "Disable nvCOMP on Copy-Engine-only platforms."
    )

# Define C++ extensions (base: no dist/Redis)
cpp_sources = [
    "csrc/bindings.cpp",
    "csrc/logging.cpp",
    # transfer.cu holds both the custom PTX kernels (compiled only when
    # FLEXKV_ENABLE_KERNEL_TRANSFER is defined) and the host dispatcher that
    # routes to the CE implementation, so it is always built.
    "csrc/transfer.cu",
    "csrc/ce_transfer.cu",
    "csrc/hash.cpp",
    "csrc/tp_transfer_thread_group.cpp",
    "csrc/transfer_ssd.cpp",
    "csrc/radix_tree.cpp",
    "csrc/eviction_strategy.cpp",
    "csrc/layerwise.cpp",
    "csrc/monitoring/metrics_manager.cpp",  # Monitoring support
]


def _host_compile_cuda_sources(sources):
    """Route .cu sources through the host compiler for CE-only builds.

    torch's ``BuildExtension`` dispatches purely on file extension: any ``.cu``
    entry is handed to nvcc, and nvcc invocations always go through
    ``_get_cuda_arch_flags()``. On a machine with no NVIDIA GPU that helper
    finds an empty arch list and dies with::

        File ".../torch/utils/cpp_extension.py", line 1984, in _get_cuda_arch_flags
            arch_list[-1] += '+PTX'
        IndexError: list index out of range

    In a CE-only build the two ``.cu`` files contain no device code at all --
    the kernels in transfer.cu are behind ``FLEXKV_ENABLE_KERNEL_TRANSFER`` and
    ce_transfer.cu is pure ``cudaMemcpyAsync`` host code -- so nvcc is not
    merely unavailable, it is unnecessary.

    Rather than pinning a fake ``TORCH_CUDA_ARCH_LIST`` (which would bake
    -gencode flags for GPUs that do not exist, and still route the files through
    a toolchain that cannot target the real accelerator), mirror each ``.cu``
    into ``build/host_src`` under a ``.cpp`` name and compile that instead.

    Symlinks are used so the mirrored files always track the originals; ``#line``
    accuracy is preserved because the compiler still reads the real content, and
    diagnostics point at the link path which resolves to the true source. Plain
    copies are used as a fallback on filesystems without symlink support.
    """
    mirror_dir = os.path.join(build_dir, "host_src")
    os.makedirs(mirror_dir, exist_ok=True)

    rerouted = []
    for src in sources:
        if not src.endswith(".cu"):
            rerouted.append(src)
            continue

        # Keep the original stem so object files and diagnostics stay readable,
        # e.g. csrc/transfer.cu -> build/host_src/transfer.cu.cpp
        mirrored = os.path.join(mirror_dir, os.path.basename(src) + ".cpp")
        abs_src = os.path.abspath(src)

        if os.path.islink(mirrored) or os.path.exists(mirrored):
            os.remove(mirrored)
        try:
            os.symlink(abs_src, mirrored)
        except (OSError, NotImplementedError):
            shutil.copy2(abs_src, mirrored)

        print(f"CE-only build: compiling {src} as host C++ via {mirrored}")
        rerouted.append(mirrored)

    return rerouted


if not enable_kernel_transfer:
    cpp_sources = _host_compile_cuda_sources(cpp_sources)

hpp_sources = [
    "csrc/logging.h",
    "csrc/cache_utils.h",
    "csrc/flexkv_nvtx.h",
    "csrc/gtensor_handler.cuh",
    "csrc/transfer.cuh",
    "csrc/tp_transfer_thread_group.h",
    "csrc/transfer_ssd.h",
    "csrc/radix_tree.h",
    "csrc/eviction_strategy.h",
    "csrc/layerwise.h",
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

# libcuda is the NVIDIA *driver* API (cuInit/cuMemCreate/cuIpc*/...). FlexKV
# never calls it: all device interaction goes through the CUDA *runtime*
# (cudaMemcpyAsync, cudaStream*, cudaEvent*, cudaMallocHost), which lives in
# libcudart and is linked by torch's CUDAExtension already.
#
# On CUDA-like non-NVIDIA accelerators there is no real libcuda.so at all -- a
# Kunlun P800 image only ships the toolkit's stub under lib64/stubs. Linking
# that stub is worse than not linking it: the build would succeed and then abort
# at dlopen/first-call time with an unhelpful error. Dropping the flag keeps the
# link honest, and costs nothing because no symbol from it is referenced.
if not enable_kernel_transfer and "-lcuda" in extra_link_args:
    extra_link_args.remove("-lcuda")
    print("CE-only build: dropping -lcuda (driver API unused; no libcuda.so "
          "outside NVIDIA platforms)")


# Prometheus libraries only when metrics enabled
if enable_metrics:
    extra_link_args.extend(["-lprometheus-cpp-pull", "-lprometheus-cpp-core"])
else:
    print("FLEXKV_ENABLE_METRICS=0: building without Prometheus monitoring")

# TORCH_CUDA_ARCH_LIST drives nvcc's -gencode flags, so it is only meaningful
# when device code is actually compiled. On CE-only builds nvcc never runs;
# probing arches there would either fail (no nvcc) or, worse, succeed against an
# unrelated CUDA toolkit and bake in -gencode flags for GPUs that do not exist.
if enable_kernel_transfer:
    # Auto-detect GPU architecture if TORCH_CUDA_ARCH_LIST is not explicitly set
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        os.environ["TORCH_CUDA_ARCH_LIST"] = detect_cuda_arch()
    print(f"TORCH_CUDA_ARCH_LIST = {os.environ['TORCH_CUDA_ARCH_LIST']}")
else:
    print("CE-only build: skipping TORCH_CUDA_ARCH_LIST detection "
          "(no device code is compiled)")

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

# ---------------------------------------------------------------------------
# Apply the platform-capability macros to both compilers.
#
# Every macro must be passed to the cxx *and* nvcc arg lists: torch's
# CUDAExtension routes .cu files through nvcc and .cpp files through the host
# compiler, and headers such as csrc/layerwise.h (which declares
# nvtxRangeId_t parameters) are included from both kinds of translation unit.
# A macro visible to only one of them would produce mismatched declarations and
# fail at link time -- or worse, silently produce an ODR violation.
# ---------------------------------------------------------------------------
_capability_macros = []
if enable_kernel_transfer:
    _capability_macros.append("-DFLEXKV_ENABLE_KERNEL_TRANSFER")
    print("ENABLE_KERNEL_TRANSFER = true: compiling custom PTX copy kernels")
else:
    print("ENABLE_KERNEL_TRANSFER = false: Copy-Engine-only transfer "
          "(csrc/transfer.cu kernels are compiled out). "
          "Remember to set FLEXKV_USE_CE_TRANSFER_H2D=1 and "
          "FLEXKV_USE_CE_TRANSFER_D2H=1 at runtime.")

if enable_nvtx:
    _capability_macros.append("-DFLEXKV_ENABLE_NVTX")
    print("ENABLE_NVTX = true: compiling with real NVTX ranges")
else:
    print("ENABLE_NVTX = false: NVTX ranges compile to no-ops "
          "(csrc/flexkv_nvtx.h shim)")

extra_compile_args.extend(_capability_macros)
nvcc_compile_args.extend(_capability_macros)

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

def _parse_requirements(path):
    """Read a pip requirements file, dropping comments and blank lines.

    requirements.txt documents why nvtx is optional, so it contains comment and
    blank lines that must not reach ``install_requires`` -- setuptools would
    otherwise try to parse them as requirement specifiers and fail.
    """
    reqs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


install_requires = _parse_requirements("requirements.txt")

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
    extras_require={
        # NVIDIA-only profiling annotations; see requirements.txt and
        # flexkv/common/nvtx_compat.py. Omitted from install_requires so that
        # FlexKV installs cleanly on Copy-Engine-only accelerators.
        "nvtx": ["nvtx>=0.2.8"],
    },
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
