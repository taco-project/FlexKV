"""
Safe subprocess utilities for FlexKV.

This module provides utilities for creating subprocesses that are safe to use
when the parent process was launched with MPI. It prevents MPI re-initialization
issues in spawned child processes.

Main APIs:
    - create_safe_process(): Main API for creating safe subprocesses
    - safe_spawn_process(): Convenience function with built-in spawn context
"""

import os
from typing import Any, Callable, Optional, Dict, Tuple
from multiprocessing import Process
from multiprocessing.context import BaseContext


def _safe_subprocess_entry(target: Callable, args: tuple, kwargs: dict) -> None:
    """
    Safe entry point for all spawned subprocesses in FlexKV.
    
    This function is the actual target of mp.Process. It sets environment
    variables BEFORE executing the user's target function to prevent MPI
    initialization issues.
    
    This is critical when the parent process was launched with MPI (e.g., via
    mpirun), as spawned child processes will attempt to re-import and re-initialize
    MPI, which causes them to hang.
    
    Args:
        target: The actual function to run in the subprocess
        args: Positional arguments for the target function
        kwargs: Keyword arguments for the target function
    """
    # Set environment variables BEFORE any imports that might trigger mpi4py
    # This prevents MPI from auto-initializing when imported by other modules
    os.environ['MPI4PY_RC_INITIALIZE'] = 'false'
    os.environ['OMPI_MCA_mpi_warn_on_fork'] = '0'
    
    # Now safe to call the actual target function
    # Even if it imports modules that contain mpi4py, MPI won't be initialized
    target(*args, **kwargs)


def create_safe_process(mp_ctx: BaseContext,
                       target: Callable,
                       args: Optional[tuple] = None,
                       kwargs: Optional[dict] = None,
                       daemon: Optional[bool] = None,
                       name: Optional[str] = None) -> Process:
    """
    Create a subprocess that is safe to use when parent was launched with MPI.
    
    This is a drop-in replacement for mp.Process() that automatically wraps
    the target function to prevent MPI initialization issues in child processes.
    
    Usage:
        # Instead of:
        process = mp_ctx.Process(target=my_func, args=(arg1, arg2), daemon=True)
        
        # Use:
        process = create_safe_process(mp_ctx, target=my_func, args=(arg1, arg2), daemon=True)
        process.start()
    
    Args:
        mp_ctx: Multiprocessing context (e.g., mp.get_context('spawn'))
        target: The function to run in the subprocess
        args: Positional arguments for the target function (default: ())
        kwargs: Keyword arguments for the target function (default: {})
        daemon: Whether the process should be a daemon process
        name: Name for the subprocess (for debugging)
    
    Returns:
        Process object ready to be started
    
    Example:
        >>> import torch.multiprocessing as mp
        >>> mp_ctx = mp.get_context('spawn')
        >>> 
        >>> def worker_func(x, y, option=None):
        >>>     print(f"Worker: {x}, {y}, {option}")
        >>> 
        >>> process = create_safe_process(
        >>>     mp_ctx,
        >>>     target=worker_func,
        >>>     args=(1, 2),
        >>>     kwargs={'option': 'test'},
        >>>     daemon=True
        >>> )
        >>> process.start()
        >>> process.join()
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    
    # Wrap the target function with our safe entry point
    process = mp_ctx.Process(
        target=_safe_subprocess_entry,
        args=(target, args, kwargs),
        daemon=daemon,
        name=name
    )
    
    return process


# Convenience function for common use case
def safe_spawn_process(target: Callable,
                      args: Optional[tuple] = None,
                      kwargs: Optional[dict] = None,
                      daemon: Optional[bool] = None,
                      name: Optional[str] = None) -> Process:
    """
    Convenience function to create and return a safe subprocess using 'spawn' context.
    
    This is equivalent to:
        mp_ctx = mp.get_context('spawn')
        process = create_safe_process(mp_ctx, target=target, ...)
    
    Args:
        target: The function to run in the subprocess
        args: Positional arguments for the target function
        kwargs: Keyword arguments for the target function
        daemon: Whether the process should be a daemon process
        name: Name for the subprocess
    
    Returns:
        Process object ready to be started
    
    Example:
        >>> def worker(x):
        >>>     print(f"Worker: {x}")
        >>> 
        >>> p = safe_spawn_process(target=worker, args=(42,), daemon=True)
        >>> p.start()
        >>> p.join()
    """
    import torch.multiprocessing as mp
    mp_ctx = mp.get_context('spawn')
    return create_safe_process(mp_ctx, target, args, kwargs, daemon, name)

