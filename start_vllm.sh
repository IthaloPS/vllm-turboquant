#!/bin/bash

set -euo pipefail

VLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${VLLM_ROOT}" || exit 1

# Make TurboQuant importable when running from this repo layout:
#   <root>/vllm (this script)
#   <root>/turboquant (python package)
TURBOQUANT_ROOT="${TURBOQUANT_ROOT:-}"
if [[ -z "${TURBOQUANT_ROOT}" ]]; then
  if [[ -d "${VLLM_ROOT}/third_party/turboquant" && -f "${VLLM_ROOT}/third_party/turboquant/setup.py" ]]; then
    TURBOQUANT_ROOT="$(cd "${VLLM_ROOT}/third_party/turboquant" && pwd)"
  elif [[ -d "${VLLM_ROOT}/../turboquant" ]]; then
    TURBOQUANT_ROOT="$(cd "${VLLM_ROOT}/../turboquant" && pwd)"
  fi
fi
if [[ -n "${TURBOQUANT_ROOT}" ]]; then
  export PYTHONPATH="${TURBOQUANT_ROOT}:${PYTHONPATH:-}"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Default to GPU 0 (override if needed)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_PATH="${CUDA_HOME}"
export CUDA_NVCC_EXECUTABLE="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export NVCC="${CUDA_HOME}/bin/nvcc"
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
export TORCH_NVCC_FLAGS="-I${CUDA_HOME}/include ${TORCH_NVCC_FLAGS:-}"
export NVCC_FLAGS="-I${CUDA_HOME}/include ${NVCC_FLAGS:-}"
export TORCH_COMPILE_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PORT=${VLLM_PORT:-8000}

echo "Lançando Qwen3.5-9B com Motor Soberano v14.0..."

export TQ_KEY_BITS=${TQ_KEY_BITS:-3}
export TQ_VALUE_BITS=${TQ_VALUE_BITS:-2}
export TQ_BUFFER_SIZE=${TQ_BUFFER_SIZE:-128}
export TQ_INITIAL_LAYERS_COUNT=${TQ_INITIAL_LAYERS_COUNT:-8}

echo "TurboQuant: key_bits=${TQ_KEY_BITS}, value_bits=${TQ_VALUE_BITS}, buffer=${TQ_BUFFER_SIZE}, initial_layers=${TQ_INITIAL_LAYERS_COUNT}"

PYTHON_BIN="${VLLM_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" && -x "${VLLM_ROOT}/venv/bin/python" ]]; then
  PYTHON_BIN="${VLLM_ROOT}/venv/bin/python"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

 "${PYTHON_BIN}" "${VLLM_ROOT}/run_api_server_turboquant.py" \
  --config config.yaml \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}"
