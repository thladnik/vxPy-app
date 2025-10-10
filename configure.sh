#!/usr/bin/env bash

# Resolve script and venv paths
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
VENV_DIR="${SCRIPT_DIR}/.venv"
VENV_PY="${VENV_DIR}/bin/python"
ACTIVATE="${VENV_DIR}/bin/activate"

# Default config or first argument
CONFIG="${1:-${SCRIPT_DIR}/configurations/example.yaml}"

# Validate venv
if [ ! -x "${VENV_PY}" ]; then
  echo "[ERROR] Virtual environment not found at '${VENV_DIR}'."
  echo "Re-run install to create it."
  exit 1
fi
if [ ! -f "${ACTIVATE}" ]; then
  echo "[ERROR] Activation script not found at '${ACTIVATE}'."
  exit 1
fi

# Activate venv
# shellcheck disable=SC1090
source "${ACTIVATE}" || {
  echo "[ERROR] Failed to activate the virtual environment."
  exit 1
}

# Prefer calling the installed console script
if command -v vxpy >/dev/null 2>&1; then
  vxpy -c "${CONFIG}" configure
  EXITCODE=$?
else
  "${VENV_PY}" -m vxpy -c "${CONFIG}" configure
  EXITCODE=$?
fi

echo "VxPy closed with exit code ${EXITCODE}."
sleep 1

exit ${EXITCODE}
