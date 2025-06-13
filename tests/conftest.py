"""
Pytest configuration file for FlexKV tests.
This file contains shared fixtures and setup code for all tests.
"""

import multiprocessing as mp

# Set the start method for multiprocessing to 'spawn'
# This ensures consistent behavior across different platforms
mp.set_start_method("spawn", force=True) 