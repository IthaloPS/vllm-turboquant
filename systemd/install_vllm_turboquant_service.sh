#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="vllm-turboquant.service"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SRC_DIR}/.." && pwd)"
UNIT_SRC="${SRC_DIR}/${SERVICE_NAME}"
ENV_SRC="${SRC_DIR}/vllm-turboquant.env"

if [[ $EUID -ne 0 ]]; then
  echo "Execute como root (ex: sudo $0)"
  exit 1
fi

if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "Unit não encontrado: ${UNIT_SRC}"
  exit 1
fi

install -m 0644 "${UNIT_SRC}" "/etc/systemd/system/${SERVICE_NAME}"

if [[ ! -f "/etc/default/vllm-turboquant" ]]; then
  install -m 0644 "${ENV_SRC}" "/etc/default/vllm-turboquant"
  echo "Criado /etc/default/vllm-turboquant (ajuste se precisar)."
fi

if ! grep -q '^VLLM_ROOT=' /etc/default/vllm-turboquant 2>/dev/null; then
  echo "VLLM_ROOT=${VLLM_ROOT}" >> /etc/default/vllm-turboquant
fi

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

echo "OK. Ver logs com: journalctl -u ${SERVICE_NAME} -f"
