"""llmsys.ops: one Python entry point per fused operator, with switchable backends (Chapter 20).

Backends
    python  pure PyTorch reference (always available; the ground truth in tests)
    triton  Triton kernels (GPU, or the CPU interpreter when TRITON_INTERPRET=1 is set before import)
    cpp     compiled C++ kernels of the ``llmsys`` operator library, running on the CPU
    cuda    compiled CUDA kernels of the ``llmsys`` operator library

Selection: ``LLMSYS_BACKEND=auto|python|triton|cpp|cuda`` (default ``auto``), or ``set_backend``.
``auto`` picks, per call, the fastest backend that is usable for the operator *and* the tensor's
device: cuda > triton > python on a GPU, cpp > python on the CPU. A backend is usable when its
registered operator (``requires_op``) exists in ``torch.ops.llmsys``.

Where the compiled operators come from
    * the installed extension ``llmsys._C`` (``pip install -e .`` in ``code/``), or
    * a just-in-time build of a subset of ``csrc/`` through :func:`jit_load` / :func:`ensure`
      (``LLMSYS_JIT=1``), which is how each operator is developed and tested on its own.
Both register into the same ``torch.ops.llmsys`` namespace.

Adding an operator: create ``llmsys/ops/<name>.py`` (copy ``rmsnorm.py``). Every module in this
package is imported automatically, so nothing in this file changes.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import pkgutil
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch

BACKENDS = ("python", "triton", "cpp", "cuda")
HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.abspath(os.path.join(HERE, "..", "..", "csrc"))

HAS_C = False
_C = None
try:  # pragma: no cover - depends on whether the extension was built
    from llmsys import _C as _C_mod  # type: ignore
    _C = _C_mod
    HAS_C = True
except Exception:  # ImportError, or an ABI mismatch after a torch upgrade
    _C = None
    HAS_C = False
HAS_CUDA_EXT = bool(HAS_C and _C.has_cuda())

try:  # pragma: no cover
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False
TRITON_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"


# ----------------------------------------------------------------------------- operator presence
def has_op(name: str) -> bool:
    """True when ``torch.ops.llmsys.<name>`` exists (from ``_C`` or a JIT build)."""
    try:
        getattr(torch.ops.llmsys, name)
        return True
    except (AttributeError, RuntimeError):
        return False


_fake_hooks: Dict[str, Callable] = {}
_fakes_done = set()


def register_fake(opname: str):
    """Like ``torch.library.register_fake("llmsys::<op>")`` but deferred until the op exists."""
    def deco(fn):
        _fake_hooks[opname] = fn
        _apply_fakes()
        return fn
    return deco


def _apply_fakes():
    for op, fn in list(_fake_hooks.items()):
        if op not in _fakes_done and has_op(op):
            try:
                torch.library.register_fake(f"llmsys::{op}")(fn)
            except RuntimeError:  # already registered (e.g. by C++)
                pass
            _fakes_done.add(op)


# ----------------------------------------------------------------------------- JIT builds
def jit_load(sources: Sequence[str], name: Optional[str] = None, verbose: bool = False,
             with_cuda: Optional[bool] = None) -> str:
    """Compile a subset of ``csrc/`` into a shared library and load it.

    ``sources`` are paths relative to ``csrc/`` (or absolute). ``.cu`` files are compiled when a CUDA
    toolkit and a GPU are present (or ``with_cuda=True``), otherwise dropped. The operators the files
    register appear in ``torch.ops.llmsys`` exactly as they would from ``llmsys._C``. Build
    directory: ``$LLMSYS_JIT_DIR`` (default: the system temp dir) / ``<name>``.
    """
    from torch.utils import cpp_extension

    srcs = [s if os.path.isabs(s) else os.path.join(CSRC, s) for s in sources]
    has_cu = any(s.endswith(".cu") for s in srcs)
    if with_cuda is None:
        with_cuda = has_cu and cpp_extension.CUDA_HOME is not None and torch.cuda.is_available()
    if not with_cuda:
        srcs = [s for s in srcs if not s.endswith(".cu")]
    if not srcs:
        raise ValueError("jit_load: nothing to compile (CUDA sources without a CUDA toolkit?)")
    name = name or "llmsys_jit_" + hashlib.md5(" ".join(sorted(srcs)).encode()).hexdigest()[:10]
    build_dir = os.path.join(os.environ.get("LLMSYS_JIT_DIR", tempfile.gettempdir()), name)
    os.makedirs(build_dir, exist_ok=True)
    cflags = ["-O3", "-std=c++20", "-fopenmp", "-Wall"]
    if os.environ.get("LLMSYS_CXXFLAGS"):
        cflags += os.environ["LLMSYS_CXXFLAGS"].split()
    else:
        cflags += ["-march=native"]
    cuda_flags = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo"]
    if with_cuda:
        cflags.append("-DLLMSYS_WITH_CUDA")
        cuda_flags.append("-DLLMSYS_WITH_CUDA")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    cpp_extension.load(name=name, sources=srcs, extra_cflags=cflags, extra_cuda_cflags=cuda_flags,
                       extra_include_paths=[CSRC], extra_ldflags=["-fopenmp"] + (["-lcublas"] if with_cuda else []),
                       build_directory=build_dir, with_cuda=with_cuda, is_python_module=False,
                       verbose=verbose)
    _apply_fakes()
    return name


def ensure(opname: str, sources: Optional[Sequence[str]] = None, verbose: bool = False) -> bool:
    """Make ``torch.ops.llmsys.<opname>`` available if we can; return whether it is.

    Order: already present (``_C`` or an earlier JIT load) -> JIT build of ``sources`` when
    ``LLMSYS_JIT=1`` -> False. Tests use the result to skip cleanly on machines without a compiler.
    """
    if has_op(opname):
        return True
    if sources and os.environ.get("LLMSYS_JIT", "0") == "1":
        try:
            jit_load(sources, verbose=verbose)
        except Exception as e:  # compiler missing, build error: report, don't crash the import
            if verbose or os.environ.get("LLMSYS_JIT_STRICT"):
                raise
            print(f"[llmsys.ops] JIT build for {opname} failed: {type(e).__name__}: {str(e)[:300]}")
            return False
    _apply_fakes()
    return has_op(opname)


# ----------------------------------------------------------------------------- registry
@dataclass
class _Impl:
    fn: Callable
    backend: str
    requires_op: Optional[str] = None


_registry: Dict[str, Dict[str, _Impl]] = {}
_backend = os.environ.get("LLMSYS_BACKEND", "auto")


def register(name: str, backend: str, requires_op: Optional[str] = None):
    """Decorator: register ``fn`` as the ``backend`` implementation of operator ``name``.

    ``requires_op`` names the compiled operator the implementation calls (default: ``name`` for the
    cpp and cuda backends); the implementation counts as usable only when that op exists.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if requires_op is None and backend in ("cpp", "cuda"):
        requires_op = name

    def deco(fn):
        _registry.setdefault(name, {})[backend] = _Impl(fn, backend, requires_op)
        return fn
    return deco


def set_backend(name: str) -> None:
    global _backend
    if name != "auto" and name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}")
    _backend = name


def get_backend() -> str:
    return _backend


def _usable(impl: _Impl, is_cuda: bool) -> bool:
    if impl.backend == "cuda" and not is_cuda:
        return False
    if impl.backend == "cpp" and is_cuda:
        return False
    if impl.backend == "triton" and not (is_cuda or TRITON_INTERPRET):
        return False
    if impl.requires_op and not has_op(impl.requires_op):
        return False
    return True


def available(name: str, device="cpu") -> tuple:
    """Backends registered for ``name`` that are usable on ``device`` right now."""
    is_cuda = torch.device(device).type == "cuda"
    return tuple(b for b, impl in _registry.get(name, {}).items() if _usable(impl, is_cuda))


def resolve(name: str, device, backend: Optional[str] = None) -> str:
    """Concrete backend for operator ``name`` on ``device`` under the current policy."""
    want = backend or _backend
    impls = _registry.get(name)
    if not impls:
        raise KeyError(f"no implementations registered for op {name!r}")
    is_cuda = torch.device(device).type == "cuda"
    if want != "auto":
        if want not in impls or not _usable(impls[want], is_cuda):
            raise RuntimeError(f"backend {want!r} is not usable for op {name!r} on {device} "
                               f"(usable: {available(name, device)})")
        return want
    order = ("cuda", "triton", "python") if is_cuda else ("cpp", "python")
    for b in order:
        if b in impls and _usable(impls[b], is_cuda):
            return b
    raise RuntimeError(f"no usable backend for op {name!r} on {device}")


def dispatch(name: str, device, backend: Optional[str] = None) -> Callable:
    return _registry[name][resolve(name, device, backend)].fn


def _autoload():
    for m in pkgutil.iter_modules(__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"{__name__}.{m.name}")


_autoload()
_apply_fakes()
