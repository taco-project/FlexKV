import os
import sys
import shutil
import importlib

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension


def get_version():
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        return f.read().strip()


build_dir = "build"
os.makedirs(build_dir, exist_ok=True)

# ─── Build configuration ─────────────────────────────────────────────────────
debug           = os.environ.get("FLEXKV_DEBUG",          "0") == "1"
enable_cfs      = os.environ.get("FLEXKV_ENABLE_CFS",     "0") == "1"
enable_p2p      = os.environ.get("FLEXKV_ENABLE_P2P",     "0") == "1"
enable_cputest  = os.environ.get("FLEXKV_ENABLE_CPUTEST", "0") == "1"
enable_metrics  = os.environ.get("FLEXKV_ENABLE_METRICS", "0") == "1"

# GPU backend selection (default: nvidia/cuda)
GPU_BACKEND = os.environ.get("FLEXKV_GPU_BACKEND", "nvidia").strip().lower()
# Normalise: "cuda" → "nvidia"
if GPU_BACKEND == "cuda":
    GPU_BACKEND = "nvidia"

# Storage backend: posix (default) | cufile (nvidia GDS) | mufile (musa GDS)
STORAGE_BACKEND = os.environ.get("FLEXKV_STORAGE_BACKEND", "posix").strip().lower()
# Legacy: FLEXKV_ENABLE_GDS=1 with nvidia backend → cufile
if os.environ.get("FLEXKV_ENABLE_GDS", "0") == "1" and GPU_BACKEND == "nvidia":
    STORAGE_BACKEND = "cufile"

if debug:
    print(f"[FlexKV] Debug mode enabled")

print(f"[FlexKV] GPU_BACKEND={GPU_BACKEND}  STORAGE_BACKEND={STORAGE_BACKEND}")

# ─── Load backend builder ────────────────────────────────────────────────────
_BUILDER_MAP = {
    "nvidia":  "build_backends.cuda_builder.CUDABuilder",
    "musa":    "build_backends.musa_builder.MUSABuilder",
    "generic": "build_backends.generic_builder.GenericBuilder",
}

if GPU_BACKEND not in _BUILDER_MAP:
    raise ValueError(
        f"Unsupported FLEXKV_GPU_BACKEND={GPU_BACKEND!r}. "
        f"Choices: {list(_BUILDER_MAP)}"
    )

_builder_path, _builder_cls = _BUILDER_MAP[GPU_BACKEND].rsplit(".", 1)
builder = getattr(importlib.import_module(_builder_path), _builder_cls)()

# Pre-build env setup (e.g. TORCH_CUDA_ARCH_LIST, MUSA_HOME check)
builder.configure_env()

# CPUTEST: for nvidia backend without GPU, remove -lcuda
if enable_cputest and GPU_BACKEND == "nvidia":
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;9.0")

# ─── Build options ───────────────────────────────────────────────────────────
build_opts = dict(
    storage_backend = STORAGE_BACKEND,
    enable_p2p      = enable_p2p,
    enable_cfs      = enable_cfs,
    enable_metrics  = enable_metrics,
    enable_cputest  = enable_cputest,
)

include_dirs = [
    os.path.abspath(os.path.join(build_dir, "include")),
    os.path.abspath("csrc"),
    os.path.abspath("third_party/xxHash"),
]

lib_dir = os.path.join(build_dir, "lib")
link_args = builder.get_link_args(**build_opts)
if os.path.exists(lib_dir):
    link_args += [f"-Wl,-rpath,{lib_dir}", "-Wl,-rpath,$ORIGIN",
                  "-Wl,-rpath,$ORIGIN/../lib"]

# All backends build into the same extension module name "flexkv.c_ext".
# GPU-vendor-specific code is selected at compile time via backend macros
# (FLEXKV_BACKEND_MUSA etc.) in csrc/bindings.cpp.
ext_name = "flexkv.c_ext"

ExtClass = builder.get_extension_class()
c_extension = ExtClass(
    name=ext_name,
    sources=builder.get_sources(**build_opts),
    library_dirs=[lib_dir],
    include_dirs=include_dirs,
    extra_compile_args=builder.get_compile_args(**build_opts),
    extra_link_args=link_args,
)

cpp_extensions = [c_extension]

# ─── Cython (release mode only) ──────────────────────────────────────────────
ext_modules = cpp_extensions

if not debug:
    python_files = ["flexkv/**/*.py"]
    excluded_files = [
        "flexkv/**/__init__.py",
        "flexkv/**/test_*.py",
        "flexkv/**/benchmark_*.py",
        "flexkv/benchmark/**/*.py",
        "flexkv/benchmark/test_kvmanager.py",
    ]
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
    print("[FlexKV] Release mode: Cython compilation enabled")
else:
    print("[FlexKV] Debug mode: Skipping Cython compilation")

# ─── CustomBuildExt: copy .so files after build ──────────────────────────────
BuildExtClass = builder.get_build_ext_class()


class CustomBuildExt(BuildExtClass):
    def run(self):
        super().run()
        self._copy_shared_libraries()

    def _copy_shared_libraries(self):
        source_lib_dir = os.path.join(build_dir, "lib")
        if not os.path.exists(source_lib_dir):
            return
        package_lib_dir = os.path.join("flexkv", "lib")
        os.makedirs(package_lib_dir, exist_ok=True)
        for fname in os.listdir(source_lib_dir):
            if fname.endswith(".so") or ".so." in fname:
                src = os.path.join(source_lib_dir, fname)
                dst = os.path.join(package_lib_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    print(f"[FlexKV] Copied {src} → {dst}")


with open("requirements.txt") as f:
    install_requires = [l.strip() for l in f if l.strip() and not l.startswith("#")]

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
    cmdclass={
        "build_ext": CustomBuildExt.with_options(
            include_dirs=os.path.join(build_dir, "include"),
            no_python_abi_suffix=True,
            build_temp=os.path.join(build_dir, "temp"),
        )
    },
    python_requires=">=3.6",
)
