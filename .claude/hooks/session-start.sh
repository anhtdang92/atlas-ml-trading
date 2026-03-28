#!/bin/bash
set -euo pipefail

# Only run in Claude Code remote (web) environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install Python dependencies (prefer binary wheels to avoid build failures)
pip install -r requirements.txt --quiet --prefer-binary --break-system-packages --ignore-installed

# Set PYTHONPATH so imports work from project root
echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"
