
# Auxiliary router data: 0 keeps the current negative-only training objective.
USE_DATA_MODE="${USE_DATA_MODE:-0}"
AUX_INTERP_COUNT="${AUX_INTERP_COUNT:-8}"
# Shared experiment model switch. Set MODEL_FAMILY=LLAMA or MODEL_FAMILY=QWEN.
MODEL_FAMILY="${MODEL_FAMILY:-LLAMA}"
case "${MODEL_FAMILY^^}" in
  LLAMA)
    EXP_MODEL="llama3.1-8B-Instruct"
    EXP_MODEL_DIR="/mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct"
    EXP_LAYER=14
    ;;
  QWEN)
    EXP_MODEL="Qwen2.5-7B-Instruct"
    EXP_MODEL_DIR="/mnt/nas-public-bing/pinxuanli11/Qwen2.5-7B-Instruct"
    EXP_LAYER=19
    ;;
  *) echo "MODEL_FAMILY must be LLAMA or QWEN" >&2; exit 2 ;;
esac
MODEL="$EXP_MODEL"
MODEL_NAME="$EXP_MODEL"
MODEL_DIR="${MODEL_DIR:-$EXP_MODEL_DIR}"
LAYER="${LAYER:-$EXP_LAYER}"
export MODEL MODEL_NAME MODEL_DIR LAYER MODEL_FAMILY
