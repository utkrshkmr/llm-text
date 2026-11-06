"""Helper so every lab's tests can target either the starter or the reference solution.

    LAB_IMPL=starter  pytest labs/lab09_attention     # default: grade your own work
    LAB_IMPL=solution pytest labs/lab09_attention     # check the reference solution
"""
import importlib.util
import os
import sys


def load_impl(test_file: str):
    which = os.environ.get("LAB_IMPL", "starter")
    lab_dir = os.path.dirname(os.path.abspath(test_file))
    path = os.path.join(lab_dir, f"{which}.py")
    name = f"{os.path.basename(lab_dir)}_{which}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # needed for dataclasses + postponed annotations
    spec.loader.exec_module(mod)
    return mod
