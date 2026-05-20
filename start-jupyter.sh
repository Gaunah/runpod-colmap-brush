#!/bin/bash
set -e

if [ -z "${JUPYTER_PASSWORD:-}" ]; then
    echo "WARNING: JUPYTER_PASSWORD not set. Generating a random token."
    JUPYTER_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
    echo "================================================"
    echo "Jupyter token: $JUPYTER_PASSWORD"
    echo "================================================"
fi

exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=True \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.token="$JUPYTER_PASSWORD" \
    --ServerApp.password=''
