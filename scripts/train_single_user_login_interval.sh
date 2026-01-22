#!/usr/bin/env bash
set -euo pipefail

USER_CSV=""
CKPT=""
SEQ_LEN="64"
PRED_LEN="1"
BATCH_SIZE="16"
LR="1e-4"
EPOCHS="30"
PATIENCE="5"
OUTPUT_ROOT="./outputs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user_csv)
      USER_CSV="$2"; shift 2 ;;
    --ckpt)
      CKPT="$2"; shift 2 ;;
    --seq_len)
      SEQ_LEN="$2"; shift 2 ;;
    --pred_len)
      PRED_LEN="$2"; shift 2 ;;
    --batch_size)
      BATCH_SIZE="$2"; shift 2 ;;
    --learning_rate)
      LR="$2"; shift 2 ;;
    --train_epochs)
      EPOCHS="$2"; shift 2 ;;
    --patience)
      PATIENCE="$2"; shift 2 ;;
    --output_root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$USER_CSV" || -z "$CKPT" ]]; then
  echo "Usage: $0 --user_csv <path> --ckpt <path> [--seq_len 64] [--pred_len 1]"
  exit 1
fi

if [[ ! -f "$USER_CSV" ]]; then
  echo "user_csv not found: $USER_CSV"; exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ckpt not found: $CKPT"; exit 1
fi

ROW_COUNT=$(python - <<PY
import pandas as pd
print(len(pd.read_csv('$USER_CSV')))
PY
)

if [[ "$ROW_COUNT" -lt 80 ]]; then
  SEQ_LEN="16"
elif [[ "$ROW_COUNT" -lt 150 ]]; then
  if [[ "$SEQ_LEN" -gt 32 ]]; then
    SEQ_LEN="32"
  fi
fi

if [[ "$ROW_COUNT" -lt 300 ]]; then
  LR="5e-5"
fi

ROOT_PATH=$(python - <<PY
import os
print(os.path.dirname('$USER_CSV'))
PY
)
DATA_PATH=$(python - <<PY
import os
print(os.path.basename('$USER_CSV'))
PY
)
USER_ID=$(python - <<PY
import os
print(os.path.splitext(os.path.basename('$USER_CSV'))[0])
PY
)

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --model_id "$USER_ID" \
  --model timer_xl \
  --data LoginIntervalUser \
  --seq_len "$SEQ_LEN" \
  --input_token_len "$SEQ_LEN" \
  --output_token_len "$PRED_LEN" \
  --test_seq_len "$SEQ_LEN" \
  --test_pred_len "$PRED_LEN" \
  --d_model 1024 \
  --e_layers 8 \
  --d_ff 2048 \
  --n_heads 8 \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" \
  --train_epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --num_workers 0 \
  --gpu 0 \
  --use_norm \
  --nonautoregressive \
  --adaptation \
  --pretrain_model_path "$CKPT" \
  --output_root "$OUTPUT_ROOT" \
  --user_id "$USER_ID"
