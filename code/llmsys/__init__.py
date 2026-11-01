"""llmsys: the companion code for *LLM Systems from Scratch*.

Core modules (written in the opening chapters, reused everywhere):
    layers, kvcache, gpt, qwen3, tokenizer, data, train, generate, flops, utils
Later chapters add sub-packages (e.g. ``llmsys.eval``, ``llmsys.rl``, ``llmsys.serving``,
``llmsys.dist``) which build only on these core modules.
"""
__version__ = "1.0.0"
