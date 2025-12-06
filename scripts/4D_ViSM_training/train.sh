export MODEL_NAME="models/Wan2.1-Fun-V1.1-14B-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=DEBUG

# accelerate launch --mixed_precision="bf16" scripts/wan2.1_fun/train_lora.py \
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/4D_ViSM_training/train.py \
  --config_path="config/wan2.1/wan_civital.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=456 \
  --checkpoints_total_limit 5 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=49 \
  --train_batch_size=2 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=10 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir/wan2.1_gs" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="inpaint" \
  --resume_from_checkpoint="latest" \
  --save_state \
  --use_3dgs 