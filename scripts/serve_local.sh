#!/usr/bin/env bash
set -euo pipefail

component="${1:-}"
if [[ -z "$component" ]]; then
    printf '%s\n' "Usage: bash scripts/serve_local.sh {builder|linker|embedding} [vLLM arguments]" >&2
    exit 2
fi
shift
command -v vllm >/dev/null || {
    printf '%s\n' "Install vLLM in the model-serving environment first." >&2
    exit 1
}

case "$component" in
    builder)
        checkpoint="${GRAPHPI_BUILDER_CHECKPOINT:?Set the local fine-tuned graph builder checkpoint}"
        exec vllm serve "$checkpoint" --served-model-name graphpi-builder \
            --host 127.0.0.1 --port "${GRAPHPI_BUILDER_PORT:-8001}" \
            --generation-config vllm "$@"
        ;;
    linker)
        checkpoint="${GRAPHPI_LINKER_CHECKPOINT:?Set the local fine-tuned cross-graph linker checkpoint}"
        exec vllm serve "$checkpoint" --served-model-name graphpi-linker \
            --host 127.0.0.1 --port "${GRAPHPI_LINKER_PORT:-8002}" \
            --generation-config vllm "$@"
        ;;
    embedding)
        checkpoint="${GRAPHPI_EMBEDDING_CHECKPOINT:-BAAI/bge-m3}"
        exec vllm serve "$checkpoint" --served-model-name BAAI/bge-m3 \
            --runner pooling --host 127.0.0.1 --port "${GRAPHPI_EMBEDDING_PORT:-8003}" "$@"
        ;;
    *)
        printf '%s\n' "Unknown component: $component" >&2
        exit 2
        ;;
esac
