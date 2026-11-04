#!/usr/bin/env bash
# Start the llama.cpp router server for Pi (/login llama.cpp -> http://127.0.0.1:14100).
#
# Model and context window settings live in models-router.ini (per-model sections).
# Router CLI flags would override ALL models, so per-model values (ctx-size, ctk/ctv,
# spec settings) must be edited there.
#
# Active config (from models-router.ini):
#   5090  (CUDA0) : qwen3.8-flash-next-113b-q4km, 262144 ctx, q8_0 KV, experts from blk.10 in RAM
#   R9700 (ROCm0) : swift-qwen3.8-27b-q4, 153600 ctx, q8_0 KV (+ monitors)
# Both are load-on-startup, hence --models-max 2.
#
# Pi thinking settings are configured in ~/.pi/agent/models.json with
# thinkingFormat "qwen-chat-template" for these Qwen3.8 models.
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-14100}"

# models-router.ini keeps model paths under ${HF_CACHE} so the file stays portable.
# llama.cpp does not expand variables in preset values, so the server is pointed at a
# rendered copy instead.
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface/hub}"
HF_MARKER='${HF_CACHE}'
ROUTER_INI="models-router.ini"
ROUTER_INI_RESOLVED="${XDG_CACHE_HOME:-$HOME/.cache}/llama.cpp/models-router.resolved.ini"

# fork perf env vars, inherited by all model instances
export GGML_CUDA_REGISTER_HOST=1
# GGML_SCHED_PREFETCH_EXPERTS=1 halves pp speed on the 113B (104 vs 169 t/s): with 512
# experts and a full ubatch nearly every expert is hot anyway, so the whole-tensor
# staging just adds slot rotation overhead.
# export GGML_SCHED_PREFETCH_EXPERTS=1

# register GGUF files from the HF cache that are not in models-router.ini yet.
# new models get [name] + model = path + a computed ctx-size, all other options
# come from the [*] defaults.

# max ctx that fits on the GPU: (VRAM ceiling - buffer - weights) / kv-per-token
# kv-per-token from GGUF metadata, assuming ctk/ctv = turbo4 (66 B per 128 values).
# prints the ctx on stdout, empty when python3/gguf is unavailable.
calc_ctx_size() {
    python3 - "$1" <<'PYEOF'
import os, sys
from gguf import GGUFReader

VRAM = 29 * 1024**3  # 32 GB card minus 3 GB buffer
TURBO4_BPV = 66 / 128  # bytes per kv value element

path = sys.argv[1]
r = GGUFReader(path)
arch = r.fields["general.architecture"].contents()

def g(key, default=None):
    f = r.fields.get(f"{arch}.{key}")
    return f.contents() if f else default

n_kv = g("attention.head_count_kv")
if not n_kv:
    sys.exit(0)

dk = g("attention.key_length") or g("attention.head_length")
dv = g("attention.value_length") or g("attention.head_length")
if not dk or not dv:
    n_heads = g("attention.head_count")
    emb = g("embedding_length")
    if not n_heads or not emb:
        sys.exit(0)
    dk = dv = emb // n_heads

# hybrid models (e.g. qwen35): only non-recurrent layers keep KV
rl = g("attention.recurrent_layers")
if rl is not None:
    n_layers = rl.count(False)
else:
    interval = g("full_attention_interval")
    n_layers = g("block_count") // interval if interval else g("block_count")

per_token = n_layers * n_kv * (dk + dv) * TURBO4_BPV
weights = os.path.getsize(path)
ctx = min(g("context_length") or float("inf"), (VRAM - weights) // per_token)
print(int(ctx) // 512 * 512)
PYEOF
}

sync_models_router() {
    local ini="$ROUTER_INI"
    local found_new=0

    while IFS= read -r -d '' f; do
        case "$(basename "$f")" in
            mmproj*) continue ;;   # vision projectors, not standalone models
            *-of-*.gguf)
                # split GGUF: only the first shard is a loadable entry point
                [[ "$f" == *-00001-of-*.gguf ]] || continue
                ;;
        esac
        [[ "$f" == */MTP/* ]] && continue   # MTP draft models, referenced via spec-draft-model

        # compare against the ${HF_CACHE} form so re-running does not re-register a model
        local p="${HF_MARKER}${f#"$HF_CACHE"}"
        grep -Fq -- "$p" "$ini" && continue

        # section name = sanitized lowercase filename, make it unique
        local base name ctx n=2
        base=$(basename "$f" .gguf | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-' | sed 's/^-//;s/-$//')
        name="$base"
        while grep -Fq "[${name}]" "$ini"; do
            name="${base}-${n}"; ((n++))
        done

        ctx=$(calc_ctx_size "$f")
        if [[ -z "$ctx" ]]; then
            echo "  ! skipped $(basename "$f"): could not read GGUF metadata (need python3 + gguf)" >&2
            continue
        fi
        if (( ctx < 4096 )); then
            echo "  ! skipped $(basename "$f"): weights leave no VRAM for KV (max ctx $ctx)" >&2
            continue
        fi

        printf '\n[%s]\nmodel = %s\nctx-size = %s\n' "$name" "$p" "$ctx" >> "$ini"
        echo "  + registered new model [$name] (ctx-size $ctx, turbo4 KV)"
        found_new=1
    done < <(find -L "$HF_CACHE" -name '*.gguf' -print0 2>/dev/null | sort -z)

    if [[ $found_new -eq 0 ]]; then
        echo "  no new models in HF cache"
    fi
}

render_models_router() {
    local ini
    ini=$(<"$ROUTER_INI")
    mkdir -p "$(dirname "$ROUTER_INI_RESOLVED")"
    printf '%s\n' "${ini//"$HF_MARKER"/$HF_CACHE}" > "$ROUTER_INI_RESOLVED"
}

echo "=== scanning Hugging Face cache for new models ==="
sync_models_router

echo "=== llama.cpp router (Pi: /login llama.cpp -> http://${HOST}:${PORT}) ==="
echo "models-router.ini sections:"
grep -E '^\[|load-on-startup|ctx-size|^ctk|^ctv|spec-type' "$ROUTER_INI" | sed 's/^/  /'
echo "----------------------------------------------------------------"
render_models_router
echo "HF cache: $HF_CACHE -> $ROUTER_INI_RESOLVED"

# build-dl has GGML_BACKEND_DL=ON, which is what lets the CUDA (5090) and HIP (R9700)
# backends coexist in one process - with DL=OFF both export ggml_backend_cuda_reg and
# CUDA shadows ROCm, so the R9700 never shows up in --list-devices.
exec ./build-dl/bin/llama-server \
    --models-preset "$ROUTER_INI_RESOLVED" \
    --models-max 2 \
    --host "$HOST" \
    --port "$PORT"
