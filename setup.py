"""FlexKV setup.py — vendor-aware build entry.

Selects a GPU builder based on ``FLEXKV_GPU_BACKEND`` (default: ``nvidia``)
and delegates source / compile / link decisions to ``build_backends/``.
The historical NVIDIA-only behavior is preserved.
"""
import os
import shutil
import sys

from setuptools import find_packages, setup

# Make build_backends importable (it lives at repo root).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_version() -> str:
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        return f.read().strip()


def get_install_requires():
    with open("requirements.txt") as f:
        return f.read().splitlines()


# ---------------------------------------------------------------------
# Build dispatch
# ---------------------------------------------------------------------
build_dir = "build"
os.makedirs(build_dir, exist_ok=True)

debug = os.environ.get("FLEXKV_DEBUG") == "1"
if debug:
    print("Running in debug mode - Cython compilation disabled")

GPU_BACKEND = os.environ.get("FLEXKV_GPU_BACKEND", "nvidia").lower().strip() or "nvidia"
print(f"FLEXKV_GPU_BACKEND={GPU_BACKEND}")

opts = dict(
    enable_gds=os.environ.get("FLEXKV_ENABLE_GDS", "0") == "1",
    enable_p2p=os.environ.get("FLEXKV_ENABLE_P2P", "0") == "1",
    enable_metrics=os.environ.get("FLEXKV_ENABLE_METRICS", "0") == "1",
    enable_cfs=os.environ.get("FLEXKV_ENABLE_CFS", "0") == "1",
    enable_cputest=os.environ.get("FLEXKV_ENABLE_CPUTEST", "0") == "1",
    build_dir=build_dir,
)

# Lazy import: avoids failing when build_backends is itself broken.
from build_backends import load_builder  # noqa: E402

builder = load_builder(GPU_BACKEND)
print(f"Selected builder: {builder.__class__.__name__}")
builder.configure_env()

ext_modules = []
if builder.get_extension_name():
    ExtCls = builder.get_extension_class()
    sources = builder.get_sources(**opts)
    if sources:
        ext = ExtCls(
            name=builder.get_extension_name(),
            sources=sources,
            library_dirs=[os.path.join(build_dir, "lib")],
            include_dirs=builder.get_include_dirs(**opts),
            extra_compile_args=builder.get_compile_args(**opts),
            extra_link_args=builder.get_link_args(**opts),
        )
        ext_modules.append(ext)


# ---------------------------------------------------------------------
# Cython compilation (release mode only)
# ---------------------------------------------------------------------
if not debug and ext_modules:
    from Cython.Build import cythonize  # noqa: E402

    python_files = ["flexkv/**/*.py"]
    excluded_files = [
        "flexkv/**/__init__.py",
        "flexkv/**/test_*.py",
        "flexkv/**/benchmark_*.py",
        "flexkv/benchmark/**/*.py",
        "flexkv/benchmark/test_kvmanager.py",
    ]
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
elif debug:
    print("Debug mode: Skipping Cython compilation")
else:
    print("No native extension produced (e.g. FLEXKV_GPU_BACKEND=generic)")


# ---------------------------------------------------------------------
# Custom build_ext that copies build/lib/*.so into the package dir
# ---------------------------------------------------------------------
BuildExtBase = builder.get_build_ext_class()


class CustomBuildExt(BuildExtBase):  # type: ignore[misc, valid-type]
    def run(self):
        super().run()
        self.copy_shared_libraries()

    def copy_shared_libraries(self):
        source_lib_dir = os.path.join(build_dir, "lib")
        if not os.path.exists(source_lib_dir):
            print(f"Warning: Source library directory {source_lib_dir} does not exist")
            return

        package_lib_dir = os.path.join("flexkv", "lib")
        os.makedirs(package_lib_dir, exist_ok=True)

        for fname in os.listdir(source_lib_dir):
            if fname.endswith(".so") or fname.endswith(".so.*"):
                src = os.path.join(source_lib_dir, fname)
                dst = os.path.join(package_lib_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    print(f"Copied {src} to {dst}")


cmdclass = {}
if ext_modules:
    # Build_ext is only needed when we have native extensions to build.
    cmdclass["build_ext"] = CustomBuildExt.with_options(
        include_dirs=os.path.join(build_dir, "include"),
        no_python_abi_suffix=True,
        build_temp=os.path.join(build_dir, "temp"),
    )


setup(
    name="flexkv",
    description="A global KV-Cache manager for LLM inference",
    version=get_version(),
    packages=find_packages(exclude=("benchmarks", "csrc", "examples", "tests", "build_backends")),
    package_data={
        "flexkv": ["*.so", "lib/*.so", "lib/*.so.*"],
    },
    include_package_data=True,
    install_requires=get_install_requires(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.6",
)
