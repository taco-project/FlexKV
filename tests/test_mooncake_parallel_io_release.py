"""Compile the complete production module, then run its real spawn suite.

Requires Cython/setuptools/compiler and local Unix sockets. No GPU/SDK.
"""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    import Cython  # required, not a silent skip
    with tempfile.TemporaryDirectory(prefix='flexkv-parallel-io-release-') as directory:
        work = Path(directory)
        shutil.copy2(ROOT/'flexkv/external/mooncake_parallel_io.py', work/'mooncake_parallel_io.py')
        tree = ast.parse((ROOT/'setup.py').read_text())
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'cythonize')
        directives = ast.literal_eval(next(kw.value for kw in call.keywords if kw.arg == 'compiler_directives'))
        (work/'setup.py').write_text(
            'from setuptools import setup\nfrom Cython.Build import cythonize\n'
            f"setup(ext_modules=cythonize('mooncake_parallel_io.py', compiler_directives={directives!r}))\n")
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=work, check=True, timeout=180)
        binary = next(work.glob('mooncake_parallel_io*.so'))
        print(f'Testing Cython {Cython.__version__}: {binary}', flush=True)
        env = dict(os.environ, FLEXKV_TEST_PARALLEL_IO_EXTENSION_DIR=str(work))
        subprocess.run([sys.executable, '-B', '-X', 'faulthandler', str(ROOT/'tests/test_mooncake_parallel_io_standalone.py'), '-v'],
                       env=env, check=True, timeout=180)


if __name__ == '__main__':
    main()
