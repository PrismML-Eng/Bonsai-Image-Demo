#!/bin/sh
# Download Bonsai LLM ONNX weights for local use (same repos as the SvelteKit speech demo).
#
# Uses curl (not huggingface_hub snapshot_download) because hf_transfer / Xet often
# hang on multi-GB .onnx_data files from UK residential links.
#
# Usage:
#   ./scripts/download_bonsai_llm.sh              # Bonsai 4B, q2f16 (web default)
#   ./scripts/download_bonsai_llm.sh 1.7b
#   ./scripts/download_bonsai_llm.sh 8b
#   ./scripts/download_bonsai_llm.sh 4b --all-dtypes
#
# Files land under models/bonsai-llm-<size>-onnx/ in standard HF layout.
# Re-run to resume interrupted downloads (curl -C -).
set -e

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DEMO_DIR/scripts/common.sh"

_usage() {
    cat <<EOF
Usage: $0 [<size>] [--all-dtypes]

  size: 1.7b | 4b (default) | 8b

  Downloads q2f16 ONNX weights (matches the browser WebGPU demo).
  --all-dtypes also fetches the q2 variant (roughly doubles ONNX size).

  Optional: HF_TOKEN=hf_... for authenticated Hub requests.
  Interrupted downloads resume on the next run.
EOF
}

_size="4b"
_all_dtypes=0
while [ $# -gt 0 ]; do
    case "$1" in
        1.7b|4b|8b) _size="$1"; shift ;;
        --all-dtypes) _all_dtypes=1; shift ;;
        -h|--help) _usage; exit 0 ;;
        *) err "Unknown argument: $1"; _usage; exit 1 ;;
    esac
done

case "$_size" in
    1.7b)
        _hf_repo="onnx-community/Ternary-Bonsai-1.7B-ONNX"
        _saved_dir="$DEMO_DIR/models/bonsai-llm-1.7b-onnx"
        _display="Bonsai LLM 1.7B ONNX"
        ;;
    4b)
        _hf_repo="onnx-community/Ternary-Bonsai-4B-ONNX"
        _saved_dir="$DEMO_DIR/models/bonsai-llm-4b-onnx"
        _display="Bonsai LLM 4B ONNX"
        ;;
    8b)
        _hf_repo="onnx-community/Ternary-Bonsai-8B-ONNX"
        _saved_dir="$DEMO_DIR/models/bonsai-llm-8b-onnx"
        _display="Bonsai LLM 8B ONNX"
        ;;
esac

_files() {
    curl -s "https://huggingface.co/api/models/${_hf_repo}/tree/main?recursive=1" \
        | python3 -c "
import json, sys
all_dtypes = sys.argv[1] == '1'
data = json.load(sys.stdin)
meta = [
    'tokenizer.json',
    'tokenizer_config.json',
    'config.json',
    'generation_config.json',
    'chat_template.jinja',
]
paths = []
for entry in data:
    if entry.get('type') != 'file':
        continue
    path = entry['path']
    if path in meta:
        paths.append(path)
    elif path == 'onnx/model_q2f16.onnx' or path.startswith('onnx/model_q2f16.onnx_data'):
        paths.append(path)
    elif all_dtypes and (path == 'onnx/model_q2.onnx' or path.startswith('onnx/model_q2.onnx_data')):
        paths.append(path)
for path in sorted(paths):
    print(path)
" "$_all_dtypes"
}

# Drop stale locks from a hung huggingface_hub run.
rm -rf "$_saved_dir/.cache"

_files_list="$(_files)"
if [ -z "$_files_list" ]; then
    err "Could not resolve file list for ${_hf_repo}"
    exit 1
fi

_file_size() {
    if [ -f "$1" ]; then
        stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

_expected_size() {
    _path="$1"
    curl -s "https://huggingface.co/api/models/${_hf_repo}/tree/main?recursive=1" \
        | python3 -c "
import json, sys
path = sys.argv[1]
data = json.load(sys.stdin)
for entry in data:
    if entry.get('type') == 'file' and entry.get('path') == path:
        print(entry.get('size', 0))
        break
else:
    print(0)
" "$_path"
}

_curl_fetch() {
    _rel="$1"
    _dest="$_saved_dir/$_rel"
    _url="https://huggingface.co/${_hf_repo}/resolve/main/${_rel}"
    _expected="$(_expected_size "$_rel")"
    _have="$(_file_size "$_dest")"

    mkdir -p "$(dirname "$_dest")"

    if [ "$_expected" -gt 0 ] && [ "$_have" -eq "$_expected" ]; then
        info "Already have $_rel ($(_human_bytes "$_have"))"
        return 0
    fi

    if [ "$_have" -gt 0 ] && [ "$_expected" -gt 0 ] && [ "$_have" -lt "$_expected" ]; then
        step "Resuming $_rel ($(_human_bytes "$_have") / $(_human_bytes "$_expected")) ..."
    else
        step "Downloading $_rel ..."
    fi

    _auth=()
    if [ -n "${HF_TOKEN:-}" ]; then
        _auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi

    # aria2c is faster than curl on HF Xet CDN, but 16 parallel range requests often
    # get 403 from short-lived signed URLs — use 4 connections and retry aggressively.
    # Occasional aria2 ERROR/403 lines are normal; aria2 retries other segments.
    if command -v aria2c >/dev/null 2>&1 && [ "$_expected" -gt 52428800 ]; then
        _aria_header=()
        if [ -n "${HF_TOKEN:-}" ]; then
            _aria_header=(--header="Authorization: Bearer ${HF_TOKEN}")
        fi
        aria2c -x4 -s4 -k1M -c --file-allocation=none \
            --console-log-level=warn --max-tries=10 --retry-wait=3 \
            "${_aria_header[@]}" -d "$(dirname "$_dest")" -o "$(basename "$_dest")" "$_url" \
            || true
        rm -f "${_dest}.aria2"
    fi

    _have="$(_file_size "$_dest")"
    if [ "$_expected" -gt 0 ] && [ "$_have" -lt "$_expected" ]; then
        step "Finishing $_rel with curl ($(_human_bytes "$_have") / $(_human_bytes "$_expected")) ..."
        if ! curl -L -f -C - "${_auth[@]}" --retry 5 --retry-delay 2 \
            -o "$_dest" "$_url"; then
            err "Failed to download $_rel"
            exit 1
        fi
        _have="$(_file_size "$_dest")"
    elif [ "$_expected" -eq 0 ] || [ "$_have" -eq 0 ]; then
        if ! curl -L -f -C - "${_auth[@]}" --retry 3 --retry-delay 2 \
            -o "$_dest" "$_url"; then
            err "Failed to download $_rel"
            exit 1
        fi
        _have="$(_file_size "$_dest")"
    fi

    if [ "$_expected" -gt 0 ] && [ "$_have" -ne "$_expected" ]; then
        err "Size mismatch for $_rel: got $_have bytes, expected $_expected"
        exit 1
    fi
    info "Saved $_rel ($(_human_bytes "$_have"))"
}

_human_bytes() {
    _b="$1"
    if [ "$_b" -ge 1073741824 ]; then
        awk -v b="$_b" 'BEGIN { printf "%.2f GB", b / 1073741824 }'
    elif [ "$_b" -ge 1048576 ]; then
        awk -v b="$_b" 'BEGIN { printf "%.0f MB", b / 1048576 }'
    elif [ "$_b" -ge 1024 ]; then
        awk -v b="$_b" 'BEGIN { printf "%.0f KB", b / 1024 }'
    else
        printf '%s B' "$_b"
    fi
}

step "Downloading ${_display} to ${_saved_dir} ..."
if command -v aria2c >/dev/null 2>&1; then
    echo "  (aria2c + curl fallback; ignore transient aria2 403 errors on HF Xet URLs)"
else
    echo "  (curl with resume — install aria2 for faster large-file downloads: brew install aria2)"
fi

for _rel in $_files_list; do
    _curl_fetch "$_rel"
done

info "Model saved to $_saved_dir"
echo ""
echo "  Weights:      $_saved_dir/onnx/model_q2f16.onnx_data*"
echo "  Graph stub:   $_saved_dir/onnx/model_q2f16.onnx"
