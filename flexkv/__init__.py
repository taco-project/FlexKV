import ctypes
import glob
import os
import sys


# Add package lib directory to system library path
def _setup_library_path() -> None:
    """Setup library path to find shared libraries in the package.

    Note: Modifying LD_LIBRARY_PATH at runtime does NOT affect the current
    process's dynamic linker (ld.so reads it only at startup). We still set it
    for child processes, but for the current process we must pre-load required
    shared libraries via ctypes.CDLL with RTLD_GLOBAL so that subsequent
    dlopen() calls (e.g. when importing c_ext) can resolve them.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(package_dir, "lib")

    if os.path.exists(lib_dir):
        # Set LD_LIBRARY_PATH for child processes
        if sys.platform.startswith('linux'):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if lib_dir not in current_ld_path:
                if current_ld_path:
                    os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{current_ld_path}"
                else:
                    os.environ['LD_LIBRARY_PATH'] = lib_dir

        # Pre-load shared libraries into the current process so that
        # c_ext (loaded via dlopen) can find them.
        for so_file in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
            try:
                ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass  # non-critical: library may not be needed

        # Add to sys.path for loading
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)


# Call the setup function when the package is imported
_setup_library_path()

# ``flexkv.c_ext`` is a PyTorch C++ extension and dynamically links against
# ``libc10.so`` / ``libtorch*.so`` from the installed ``torch`` package. Those
# libraries live under ``<site-packages>/torch/lib`` and are NOT on the system
# linker search path. Importing ``torch`` here causes Python to ``dlopen`` them
# (with RTLD_GLOBAL), so any subsequent ``import flexkv.c_ext`` can resolve
# them without requiring the caller to ``import torch`` first or to set
# ``LD_LIBRARY_PATH``.
import torch  # noqa: E402,F401  (side-effect import: load libtorch/libc10)
