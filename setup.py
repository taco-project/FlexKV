import os
import shutil
import sys
import sysconfig
import subprocess
import glob as glob_module


from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
import torch
from torch.utils.cpp_extension import CUDA_HOME

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
    "csrc/transfer.cu",
    "csrc/hash.cpp",
    "csrc/tp_transfer_thread_group.cpp",
    "csrc/transfer_ssd.cpp",
    "csrc/radix_tree.cpp",
    "csrc/layerwise.cpp"
]

hpp_sources = [
    "csrc/cache_utils.h",
    "csrc/tp_transfer_thread_group.h",
    "csrc/transfer_ssd.h",
    "csrc/radix_tree.h",
    "csrc/layerwise.h",
]

# nvCOMP paths for compression support
nvcomp_root = os.environ.get("NVCOMP_ROOT", os.path.abspath("../cuda_test/nvcomp/nvcomp"))
nvcomp_include = os.path.join(nvcomp_root, "include")
nvcomp_lib = os.path.join(nvcomp_root, "lib")

extra_link_args = ["-lcuda", "-lxxhash", "-lpthread", "-lrt", "-luring"]
# Add nvCOMP library if available - link the full static library for device code
if os.path.exists(nvcomp_lib):
    nvcomp_device_static = os.path.join(nvcomp_lib, "libnvcomp_device_static.a")
    extra_link_args.extend([
        f"-L{nvcomp_lib}",
        "-lnvcomp",
        "-Wl,--whole-archive",
        nvcomp_device_static,  # Link static library directly for device code
        "-Wl,--no-whole-archive",
        f"-Wl,-rpath,{nvcomp_lib}"
    ])

extra_compile_args = ["-std=c++17"]
include_dirs = [os.path.abspath(os.path.join(build_dir, "include"))]
# Add nvCOMP include if available
if os.path.exists(nvcomp_include):
    include_dirs.append(nvcomp_include)

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

nvcc_compile_args = ["-O3", "-Xcompiler", "-fPIC"]
# Enable relocatable device code for nvCOMP device API linking
enable_rdc = os.path.exists(nvcomp_lib)
print(f"enable_rdc: {enable_rdc}; nvcomp_lib: {nvcomp_lib}")
if enable_rdc:
    nvcc_compile_args.extend(["-rdc=true", "--extended-lambda"])
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

    def build_extension(self, ext):
        """Override to handle RDC device linking for nvCOMP"""
        if not enable_rdc:
            return super().build_extension(ext)
        
        # Find all .cu sources
        cu_sources = [s for s in ext.sources if s.endswith('.cu')]
        other_sources = [s for s in ext.sources if not s.endswith('.cu')]
        
        if not cu_sources:
            return super().build_extension(ext)
        
        print(f"=== nvCOMP RDC build: performing device linking for {len(cu_sources)} CUDA files ===")
        
        # Get build directories
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        
        # Get nvcc path
        nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
        
        # Get CUDA architecture from torch
        cuda_arch = self._get_cuda_arch()
        
        # Compile .cu files to .o with RDC
        cu_objects = []
        for cu_src in cu_sources:
            obj_name = os.path.splitext(os.path.basename(cu_src))[0] + '.o'
            obj_path = os.path.join(build_temp, obj_name)
            
            compile_cmd = [
                nvcc, '-c', cu_src, '-o', obj_path,
                '-Xcompiler', '-fPIC',
                '-rdc=true',
                '--extended-lambda',
                '-O3',
                '-std=c++17',
            ]
            # Add architecture flags
            for arch in cuda_arch:
                compile_cmd.extend(['-gencode', arch])
            # Add include directories
            for inc_dir in ext.include_dirs:
                compile_cmd.extend(['-I', inc_dir])
            # Add Python include directory
            python_include = sysconfig.get_path('include')
            compile_cmd.extend(['-I', python_include])
            # Add torch include directories
            import torch
            for path in torch.utils.cpp_extension.include_paths():
                compile_cmd.extend(['-I', path])
            
            print(f"Compiling {cu_src} -> {obj_path}")
            subprocess.check_call(compile_cmd)
            cu_objects.append(obj_path)
        
        # Perform device linking (required for RDC)
        dlink_obj = os.path.join(build_temp, 'dlink.o')
        dlink_cmd = [
            nvcc, '-dlink',
            '-Xcompiler', '-fPIC',
            '-o', dlink_obj,
        ]
        for arch in cuda_arch:
            dlink_cmd.extend(['-gencode', arch])
        dlink_cmd.extend(cu_objects)
        # Include nvCOMP device static library in device linking
        if os.path.exists(nvcomp_lib):
            nvcomp_device_static = os.path.join(nvcomp_lib, "libnvcomp_device_static.a")
            if os.path.exists(nvcomp_device_static):
                dlink_cmd.append(nvcomp_device_static)
        dlink_cmd.extend(['-lcudadevrt'])
        
        print(f"Device linking -> {dlink_obj}")
        subprocess.check_call(dlink_cmd)
        cu_objects.append(dlink_obj)
        
        # Compile C++ sources manually
        print(f"=== Compiling {len(other_sources)} C++ files ===")
        cpp_objects = []
        # Use g++ for C++ files, not gcc
        cpp_compiler = 'g++'
        
        cpp_dir = os.path.join(build_temp, 'csrc')
        os.makedirs(cpp_dir, exist_ok=True)
        
        # Get include paths
        inc_paths = []
        for inc_dir in ext.include_dirs:
            inc_paths.extend(['-I', inc_dir])
        import torch
        for path in torch.utils.cpp_extension.include_paths():
            inc_paths.extend(['-I', path])
        inc_paths.extend(['-I', sysconfig.get_path('include')])
        
        for cpp_src in other_sources:
            obj_name = os.path.splitext(os.path.basename(cpp_src))[0] + '.o'
            obj_path = os.path.join(cpp_dir, obj_name)
            
            compile_cmd = [
                cpp_compiler, '-c', cpp_src, '-o', obj_path,
                '-fPIC', '-O2', '-std=c++17',
                '-DNDEBUG',
                '-DTORCH_API_INCLUDE_EXTENSION_H',
                '-DTORCH_EXTENSION_NAME=c_ext',
                '-D_GLIBCXX_USE_CXX11_ABI=0',  # Match PyTorch's ABI
            ]
            compile_cmd.extend(inc_paths)
            
            print(f"Compiling {cpp_src} -> {obj_path}")
            subprocess.check_call(compile_cmd)
            cpp_objects.append(obj_path)
        
        # Now we do the final linking with nvcc to handle the non-PIC static library
        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)
        
        print(f"=== Final linking with nvcc for nvCOMP support ===")
        
        # Collect all object files
        all_objects = cu_objects + cpp_objects
        
        # Build final shared library with nvcc
        final_link_cmd = [
            nvcc, '-shared',
            '-Xcompiler', '-fPIC',
            '-o', ext_fullpath,
        ]
        # Add architecture flags
        for arch in cuda_arch:
            final_link_cmd.extend(['-gencode', arch])
        # Add all object files
        final_link_cmd.extend(all_objects)
        # Add nvCOMP device static library (nvcc can handle non-PIC)
        if os.path.exists(nvcomp_lib):
            nvcomp_device_static = os.path.join(nvcomp_lib, "libnvcomp_device_static.a")
            if os.path.exists(nvcomp_device_static):
                final_link_cmd.extend(['-Xlinker', '--whole-archive', nvcomp_device_static, '-Xlinker', '--no-whole-archive'])
        # Add library paths
        final_link_cmd.extend(['-L' + d for d in ext.library_dirs])
        final_link_cmd.extend([f'-L{nvcomp_lib}'])
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        final_link_cmd.extend([f'-L{torch_lib}'])
        cuda_lib = os.path.join(CUDA_HOME, 'lib64')
        final_link_cmd.extend([f'-L{cuda_lib}'])
        # Add build lib directory for xxhash
        final_link_cmd.extend([f'-L{build_dir}/lib'])
        # Add libraries
        final_link_cmd.extend([
            '-lnvcomp', '-lcudadevrt', '-lcudart',
            '-lc10', '-ltorch', '-ltorch_cpu', '-ltorch_python', '-lc10_cuda', '-ltorch_cuda',
            '-lcuda', '-lxxhash', '-lpthread', '-lrt', '-luring',
        ])
        # Add rpath
        final_link_cmd.extend([
            f'-Xlinker,-rpath,{nvcomp_lib}',
            f'-Xlinker,-rpath,{torch_lib}',
            f'-Xlinker,-rpath,{build_dir}/lib',
            '-Xlinker,-rpath,$ORIGIN',
            '-Xlinker,-rpath,$ORIGIN/../lib',
        ])
        
        print(f"Final link: {ext_fullpath}")
        subprocess.check_call(final_link_cmd)
    
    def _get_cuda_arch(self):
        """Get CUDA architecture flags from torch or detect from current GPU"""
        # Try to get from environment
        cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '')
        if cuda_arch_list:
            archs = []
            for arch in cuda_arch_list.replace(' ', '').split(';'):
                arch = arch.replace('.', '')
                if '+PTX' in arch:
                    arch = arch.replace('+PTX', '')
                    archs.append(f'arch=compute_{arch},code=sm_{arch}')
                    archs.append(f'arch=compute_{arch},code=compute_{arch}')
                else:
                    archs.append(f'arch=compute_{arch},code=sm_{arch}')
            return archs
        
        # Default: detect from current GPU or use common architectures
        try:
            import torch
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                arch = f'{major}{minor}'
                return [f'arch=compute_{arch},code=sm_{arch}']
        except:
            pass
        
        # Fallback to common architectures (Ampere + Hopper)
        return [
            'arch=compute_80,code=sm_80',
            'arch=compute_90,code=sm_90',
        ]

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
    python_requires=">=3.8",
)
