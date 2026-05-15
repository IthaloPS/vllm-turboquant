#!/usr/bin/env python3
"""
Wrapper to launch vLLM OpenAI API server with TurboQuant enabled.

This must run BEFORE vLLM initializes the engine so hooks can be installed
as part of KV cache spec setup (no_alloc path).
"""

from __future__ import annotations

import os
import runpy
import sys


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def main() -> None:
    # TurboQuant config (can be overridden via env vars)
    key_bits = _env_int("TQ_KEY_BITS", 3)
    value_bits = _env_int("TQ_VALUE_BITS", 2)
    buffer_size = _env_int("TQ_BUFFER_SIZE", 128)
    initial_layers_count = _env_int("TQ_INITIAL_LAYERS_COUNT", 8)

    # Patch vLLM v1 executor to auto-install hooks during engine init.
    try:
        from turboquant.vllm_attn_backend import enable_no_alloc
    except ModuleNotFoundError as e:
        if e.name != "turboquant":
            raise
        msg = (
            "TurboQuant não encontrado (import turboquant falhou).\n"
            "Dicas:\n"
            "- Rode ./bootstrap.sh ou pip install -r requirements.txt no mesmo Python\n"
            "- Ou instale manualmente: pip install -e vllm/third_party/turboquant[vllm]\n"
            "- Ou defina TURBOQUANT_ROOT / PYTHONPATH apontando pro pacote\n"
        )
        print(msg, file=sys.stderr)
        raise

    enable_no_alloc(
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
        initial_layers_count=initial_layers_count,
    )

    # Re-run vLLM's entrypoint as if we executed:
    #   python -m vllm.entrypoints.openai.api_server <args...>
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()

