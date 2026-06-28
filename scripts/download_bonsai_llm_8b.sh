#!/bin/sh
# Download Bonsai LLM 8B ONNX (~2.2 GB q2f16 weights across two shards).
exec "$(dirname "$0")/download_bonsai_llm.sh" 8b "$@"
