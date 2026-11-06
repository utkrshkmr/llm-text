"""Project 11: load the implementation to run or grade (Chapter 5).

    P11_IMPL=/path/to/your_package    your code (a package with the same modules as reference/)
    LAB_IMPL=solution                 the reference solution in reference/
Imported under a unique module name so it never collides with other kits' ``reference`` packages.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "reference")


def impl_path() -> Optional[str]:
    """Directory of the package to use, or None if neither variable selects one."""
    if os.environ.get("P11_IMPL"):
        return os.path.abspath(os.path.expanduser(os.environ["P11_IMPL"]))
    if os.environ.get("LAB_IMPL") == "solution":
        return REFERENCE
    return None


def load(path: Optional[str] = None):
    """Import the package at ``path`` (default: ``impl_path()``, falling back to reference/)."""
    path = path or impl_path() or REFERENCE
    name = "p11_impl_" + str(abs(hash(path)))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
