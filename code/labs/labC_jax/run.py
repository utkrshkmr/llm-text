"""Lab C driver: dump StableHLO and optimised HLO for every sharded-matmul case, count collectives.

    python labs/labC_jax/run.py --out /tmp/labC_hlo            # 8 simulated CPU devices
    python labs/labC_jax/run.py --out /tmp/labC_hlo --profile  # also record a Perfetto trace
    python labs/labC_jax/run.py --help

For each case the driver writes ``<out>/<name>.stablehlo.mlir`` (what JAX handed to XLA) and
``<out>/<name>.hlo.txt`` (what XLA will run, after SPMD partitioning and fusion) and prints one
row per case with the collectives found in the optimised HLO. The same script runs unchanged on
real GPUs or TPUs: set JAX_PLATFORMS=gpu and drop the simulated-device flag.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import llmsys.jaxlab  # noqa: E402,F401  sets XLA_FLAGS before jax is imported


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"labC_{name}", os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", default="solution", choices=["solution", "starter"])
    ap.add_argument("--out", default=os.path.join(HERE, "hlo_out"), help="directory for the dumps")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--profile", action="store_true", help="record a jax.profiler trace of case 3")
    ap.add_argument("--mental-model", action="store_true", help="print the PyTorch-to-JAX table")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from llmsys.jaxlab import basics, hlo
    from llmsys.jaxlab import sharding as S

    lab = load(args.impl)
    print(f"jax {jax.__version__}, {len(jax.devices())} {jax.devices()[0].platform} devices")
    if args.mental_model:
        width = max(len(a) for a, _ in basics.MENTAL_MODEL)
        for torch_side, jax_side in basics.MENTAL_MODEL:
            print(f"  {torch_side:<{width}}  ->  {jax_side}")

    mesh = lab.make_mesh((2, 4), ("dp", "tp"))
    a = jnp.asarray(jax.random.normal(jax.random.key(0), (args.m, args.k)))
    b = jnp.asarray(jax.random.normal(jax.random.key(1), (args.k, args.n)))
    x = jnp.asarray(jax.random.normal(jax.random.key(2), (args.m, args.k)))
    w = jnp.asarray(jax.random.normal(jax.random.key(3), (args.k, args.m)))

    # (name, function, (array, spec) pairs): the automatic versions first, then shard_map
    cases = [
        ("auto_case1", lab.sharded_matmul_auto(mesh, ("dp", None), (None, "tp")),
         [(a, ("dp", None)), (b, (None, "tp"))]),
        ("auto_case2", lab.sharded_matmul_auto(mesh, ("dp", "tp"), ()),
         [(a, ("dp", "tp")), (b, ())]),
        ("auto_case3", lab.sharded_matmul_auto(mesh, ("dp", "tp"), ("tp", None)),
         [(a, ("dp", "tp")), (b, ("tp", None))]),
        ("auto_case3_rs", lab.sharded_matmul_auto(mesh, ("dp", "tp"), ("tp", None), ("dp", "tp")),
         [(a, ("dp", "tp")), (b, ("tp", None))]),
        ("auto_case4", lab.sharded_matmul_auto(mesh, ("tp", None), (None, "tp")),
         [(a, ("tp", None)), (b, (None, "tp"))]),
        ("smap_case1_local", jax.jit(lab.case1_local(mesh)),
         [(a, ("dp", None)), (b, (None, "tp"))]),
        ("smap_case2_gather", jax.jit(lab.case2_gather(mesh)), [(a, ("dp", "tp")), (b, ())]),
        ("smap_case3_allreduce", jax.jit(lab.case3_allreduce(mesh)),
         [(a, ("dp", "tp")), (b, ("tp", None))]),
        ("smap_case3_reducescatter", jax.jit(lab.case3_reducescatter(mesh)),
         [(a, ("dp", "tp")), (b, ("tp", None))]),
        ("smap_case4_gather", jax.jit(lab.case4_gather(mesh)),
         [(a, ("tp", None)), (b, (None, "tp"))]),
        ("collective_matmul_ring", jax.jit(lab.collective_matmul_allgather(mesh)),
         [(x, ("tp", None)), (w, (None, "tp"))]),
    ]

    os.makedirs(args.out, exist_ok=True)
    header = f"{'case':28s} {'out sharding':>16s} " + " ".join(f"{c:>12s}" for c in hlo.COLLECTIVES)
    print(header)
    print("-" * len(header))
    for name, fn, operands in cases:
        arrays = [S.shard(arr, mesh, *spec) for arr, spec in operands]
        ref = operands[0][0] @ operands[1][0]
        out = fn(*arrays)
        err = float(jnp.max(jnp.abs(out - ref)))
        assert err < 1e-3 * float(jnp.max(jnp.abs(ref))), f"{name}: max error {err}"
        hlo.dump_hlo(fn, arrays, os.path.join(args.out, name))
        counts = lab.count_collectives(hlo.compiled_text(fn, *arrays))
        spec = "(" + ",".join("-" if s is None else s for s in S.spec_of(out)) + ")"
        print(f"{name:28s} {spec:>16s} " + " ".join(f"{counts[c]:>12d}" for c in hlo.COLLECTIVES))
    print(f"\nwrote *.stablehlo.mlir and *.hlo.txt for {len(cases)} cases to {args.out}")

    if args.profile:
        fn = cases[2][1]
        arrays = [S.shard(arr, mesh, *spec) for arr, spec in cases[2][2]]
        files = hlo.profile_trace(fn, arrays, os.path.join(args.out, "profile"))
        traces = [f for f in files if f.endswith("trace.json.gz")]
        print(f"\nprofile written: {len(files)} files; open {traces[0] if traces else '?'} at "
              "ui.perfetto.dev")
        if traces:
            for event, total_us, count in hlo.summarize_trace(traces[0], top=8):
                print(f"  {total_us:10.1f} us  x{count:<4d} {event}")


if __name__ == "__main__":
    main()
