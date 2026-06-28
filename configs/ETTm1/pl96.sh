#!/bin/bash
# Best config for ETTm1 pred_len=96
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${1:-0}

for seed in 2021 2022 2023; do
  python -m gse_xlstm fit+test \
    --data ForecastingDataModule --data.dataset_name "ETTm1" \
    --data.seq_len 768 --data.pred_len 96 --data.label_len 0 \
    --data.batch_size 32 --data.num_workers 2 --data.persistent_workers true \
    --model LongTermForecastingExp --model.criterion torch.nn.L1Loss \
    --model.architecture GSEXlstm \
    --model.architecture.patch_size 16 \
    --model.architecture.xlstm_embedding_dim 128 \
    --model.architecture.xlstm_num_heads 8 \
    --model.architecture.xlstm_num_blocks 1 \
    --model.architecture.xlstm_dropout 0.05 \
    --model.architecture.xlstm_conv1d_kernel_size 0 \
    --model.architecture.num_mem_tokens 2 \
    --model.architecture.time_branch_weight 1.0 \
    --model.architecture.spatial_branch_weight 1.0 \
    --model.architecture.spatial_dim 128 \
    --model.architecture.spatial_num_groups 0 \
    --model.architecture.spatial_hidden_mult 4 \
    --model.architecture.use_nlinear_baseline false \
    --optimizer.lr 0.001 \
    --lr_scheduler.warmup_epochs 5 \
    --lr_scheduler.constant_gamma_epochs 2 \
    --lr_scheduler.gamma 0.98 \
    --lr_scheduler.cosine_epochs 35 \
    --trainer.max_epochs 60 --seed_everything $seed \
    --trainer.logger.name "ETTm1_pl96_seed${seed}" \
    --trainer.logger.project gse-xlstm \
    --trainer.enable_progress_bar true \
    --model_checkpoint.dirpath "/tmp/gse_xlstm_ckpt/ETTm1_pl96_seed${seed}" \
    --model_checkpoint.save_top_k 1 --model_checkpoint.monitor val/loss
done

echo "=== ETTm1 pl=96 DONE (3 seeds) ==="
