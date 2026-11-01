"""GPU kernels (Appendix A).

``reference``      -- PyTorch reference implementations (run anywhere).
``triton_kernels`` -- Triton versions. They run on a GPU, or on the CPU through
                      Triton's interpreter when ``TRITON_INTERPRET=1`` is set *before*
                      ``triton`` is imported (slow, but great for debugging).
"""
try:  # pragma: no cover - depends on the environment
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:  # ImportError or driver problems
    HAS_TRITON = False
