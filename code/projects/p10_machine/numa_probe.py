"""Project 10: one host<->device measurement bound to one NUMA node (Chapter 4).

Run by ``bandwidth.numa_h2d`` in a fresh process per placement, so that no page and no cached
pinned block is shared between placements. Binds CPUs and memory policy to ``--node`` before
touching any host memory, measures, then reports the nodes that actually hold the buffer.

    python numa_probe.py --node 1 --nbytes 268435456 --pinned --gpu 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--nbytes", type=int, default=256 * 2**20)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--direction", default="h2d", choices=["h2d", "d2h"])
    ap.add_argument("--pinned", action="store_true")
    ap.add_argument(
        "--impl", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")
    )
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location(
        "p10_numa_impl", os.path.join(a.impl, "__init__.py"), submodule_search_locations=[a.impl]
    )
    p10 = importlib.util.module_from_spec(spec)
    sys.modules["p10_numa_impl"] = p10
    spec.loader.exec_module(p10)
    p10.bandwidth.bind_to_node(a.node)  # before any host buffer exists
    row = p10.bandwidth.host_device_bandwidth(
        a.nbytes, a.direction, a.pinned, f"cuda:{a.gpu}", a.reps
    )
    row.update(bound_node=a.node, gpu_node=p10.bandwidth.gpu_numa_node(a.gpu))
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
