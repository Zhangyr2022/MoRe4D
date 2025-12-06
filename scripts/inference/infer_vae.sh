python scripts/inference/infer_vae.py  \
  --vae_model_path "models/Wan2.1-Fun-V1.1-14B-InP/Wan2.1_VAE.pth" \
  --ckpt_dir "/path/to/your/finetuned/vae" \
  --input_sceneflow /path/to/your/input_sceneflow \
  --output_dir "/path/to/your/output" \
  --num_frames 49 \
  --normalize_track_z  \
  --mixed_precision "bf16"