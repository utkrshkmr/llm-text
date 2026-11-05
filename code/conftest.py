import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU (skipped automatically without one)")
    config.addinivalue_line("markers", "slow: long-running test (GPU test plan)")
