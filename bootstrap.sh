#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${VLLM_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

echo "Bootstrap vLLM+TurboQuant"
echo "- VLLM_ROOT: ${VLLM_ROOT}"
echo "- Python: ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERRO: Python não encontrado: ${PYTHON_BIN}"
  exit 1
fi

echo "Atualizando pip/setuptools/wheel..."
"${PYTHON_BIN}" -m pip install -U pip setuptools wheel

echo "Verificando turboquant..."
if "${PYTHON_BIN}" -c "import turboquant" >/dev/null 2>&1; then
  echo "OK: turboquant já está instalado."
  exit 0
fi

TURBOQUANT_ROOT="${TURBOQUANT_ROOT:-}"
if [[ -z "${TURBOQUANT_ROOT}" ]]; then
  if [[ -d "${VLLM_ROOT}/third_party/turboquant" && -f "${VLLM_ROOT}/third_party/turboquant/setup.py" ]]; then
    TURBOQUANT_ROOT="$(cd "${VLLM_ROOT}/third_party/turboquant" && pwd)"
  elif [[ -d "${VLLM_ROOT}/../turboquant" ]]; then
    TURBOQUANT_ROOT="$(cd "${VLLM_ROOT}/../turboquant" && pwd)"
  fi
fi

if [[ -z "${TURBOQUANT_ROOT}" ]]; then
  cat <<'EOF'
ERRO: turboquant não está instalado e eu não encontrei o diretório.

Opções:
- Este repositório inclui cópia em third_party/turboquant; confira se a pasta existe, OU
- Clone o repo do turboquant ao lado deste (../turboquant), OU
- Exporte TURBOQUANT_ROOT apontando pro clone, OU
- Instale o pacote no Python do serviço (pip install -e /caminho/turboquant)
EOF
  exit 1
fi

echo "Instalando turboquant em modo editable de: ${TURBOQUANT_ROOT}"
"${PYTHON_BIN}" -m pip install -e "${TURBOQUANT_ROOT}[vllm]"

echo "OK: bootstrap finalizado."

