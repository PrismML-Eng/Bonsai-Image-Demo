#!/bin/sh
# Download Bonsai LLM 1.7B ONNX (~470 MB q2f16 weights).
exec "$(dirname "$0")/download_bonsai_llm.sh" 1.7b "$@"
