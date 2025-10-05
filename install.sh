#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
VENV_DIR="${SCRIPT_DIR}/.venv"
VENV_PY="${VENV_DIR}/bin/python"
PYTHON_EXE=""

find_python38_plus() {
  local c
  for c in python3.13 python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3, 8) else 1)
PY
      then
        PYTHON_EXE="$(command -v "$c")"
        return 0
      fi
    fi
  done
  return 1
}

ensure_venv_module() {
  if "$PYTHON_EXE" - <<'PY' >/dev/null 2>&1
import venv
PY
  then
    return 0
  fi

  if command -v apt-get >/dev/null 2>&1; then
    local ver
    ver="$("$PYTHON_EXE" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    echo "Installing venv support for Python ${ver}..."
    if sudo apt-get update && sudo apt-get install -y "python${ver}-venv"; then
      :
    else
      echo "Falling back to installing python3-venv..."
      sudo apt-get install -y python3-venv || true
    fi
  fi

  "$PYTHON_EXE" - <<'PY' >/dev/null 2>&1
import venv
PY
}

if ! find_python38_plus; then
  echo "[ERROR] Python 3.8+ not found. Install it: sudo apt-get install -y python3 python3-venv"
  exit 1
fi
echo "Using Python: ${PYTHON_EXE}"

if ! ensure_venv_module; then
  echo "[ERROR] Python venv module is missing and could not be installed."
  exit 1
fi

if [ ! -x "${VENV_PY}" ]; then
  echo "Creating virtual environment at '${VENV_DIR}'..."
  "${PYTHON_EXE}" -m venv "${VENV_DIR}"
else
  echo "Virtual environment already exists at '${VENV_DIR}'."
fi

echo "Upgrading pip..."
"${VENV_PY}" -m pip install --upgrade pip

echo "Installing package 'vxpy'..."
"${VENV_PY}" -m pip install vxpy

echo "Done."
