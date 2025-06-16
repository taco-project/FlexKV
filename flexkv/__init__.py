import os
import sys


# Add package lib directory to system library path
def _setup_library_path() -> None:
    """Setup library path to find shared libraries in the package"""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(package_dir, "lib")

    if os.path.exists(lib_dir):
        # Add to LD_LIBRARY_PATH for Linux
        if sys.platform.startswith('linux'):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if lib_dir not in current_ld_path:
                if current_ld_path:
                    os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{current_ld_path}"
                else:
                    os.environ['LD_LIBRARY_PATH'] = lib_dir

        # Add to sys.path for loading
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)


# Call the setup function when the package is imported
_setup_library_path()
