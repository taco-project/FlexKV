"""
Pytest configuration file for FlexKV tests.
This file contains shared fixtures and setup code for all tests.
"""
# Import fixtures from test_utils so pytest can discover them
from test_utils import model_config, cache_config, test_config

import multiprocessing as mp

# Set the start method for multiprocessing to 'spawn'
# This ensures consistent behavior across different platforms
mp.set_start_method("spawn", force=True)
