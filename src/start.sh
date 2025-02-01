#!/usr/bin/env bash

# Use libtcmalloc for better memory management
#TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
#export LD_PRELOAD="${TCMALLOC}"

# Serve the API and don't shutdown the container
echo "runpod-worker-kohya: Starting RunPod Handler"

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python3 -u ./handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python3 -u ./handler.py
fi
