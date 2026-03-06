"""
Pytest configuration file for FlexKV tests.
This file contains shared fixtures and setup code for all tests.
"""
# Import fixtures from common_utils so pytest can discover them
from common_utils import model_config, cache_config, test_config

# Register custom markers (e.g. for SIMM integration tests)
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "simm: mark test as SIMM integration (requires running SIMM manager)"
    )
