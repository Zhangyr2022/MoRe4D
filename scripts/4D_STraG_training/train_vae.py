import imageio
import os
import argparse
import contextlib
import gc
import logging
import math
import pickle
import shutil
import sys
from collections import deque
from pathlib import Path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from MoRe4D.models.wan_vae import AutoencoderKLWan
from MoRe4D.models.trajectory_module import VAEEncoderadaptor, VAEDecoderadaptor
from MoRe4D.data.vae_dataset import VAEDataset

logger = get_logger(__name__)


class LossTracker:
    """Tracks loss statistics for training monitoring and batch skipping."""
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.total_loss = 0.0
        self.total_samples = 0
        
    def update(self, loss_value, batch_size=1):
        """Update tracker with new loss value."""
        if torch.is_tensor(loss_value):
            loss_value = loss_value.item()
        
        self.loss_history.append(loss_value)
        self.total_loss += loss_value * batch_size
        self.total_samples += batch_size
    
    def get_global_average(self):
        """Get average loss across all samples."""
        return self.total_loss / self.total_samples if self.total_samples > 0 else 0.0
    
    def get_window_average(self):
        """Get average loss in current window."""
        return sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0.0
    
    def get_window_std(self):
        """Get standard deviation of loss in current window."""
        if len(self.loss_history) <= 1:
            return 0.0
        
        mean_loss = self.get_window_average()
        variance = sum((loss - mean_loss) ** 2 for loss in self.loss_history) / len(self.loss_history)
        return math.sqrt(variance)


def should_skip_batch(loss_value, loss_tracker, args):
    """Determine if batch should be skipped based on loss outlier detection."""
    if torch.is_tensor(loss_value):
        loss_value = loss_value.item()
    
    # Skip if loss is not finite
    if not torch.isfinite(torch.tensor(loss_value)):
        logger.warning(f"Skip batch: loss is not finite (loss={loss_value})")
        return True
    
    # Skip if absolute threshold exceeded
    if loss_value > args.loss_skip_absolute_threshold:
        logger.warning(f"Skip batch: loss={loss_value:.6f} > absolute_threshold={args.loss_skip_absolute_threshold}")
        return True
    
    # Need minimum samples for statistical outlier detection
    if len(loss_tracker.loss_history) < args.loss_skip_min_samples:
        return False
    
    window_mean = loss_tracker.get_window_average()
    window_std = loss_tracker.get_window_std()
    
    # Calculate dynamic threshold
    if window_std < 1e-6:
        threshold = window_mean * args.loss_skip_multiplier
    else:
        threshold = window_mean + args.loss_skip_std_multiplier * window_std
    
    if loss_value > threshold:
        logger.warning(
            f"Skip batch: loss={loss_value:.6f} > threshold={threshold:.6f} "
            f"(mean={window_mean:.6f}, std={window_std:.6f})"
        )
        return True
    
    return False


def collate_fn(examples):
    """Custom collate function for batching data."""
    sample = {}
    for key in examples[0].keys():
        sample[key] = torch.cat([example[key] for example in examples], dim=0)
    return sample


def normalize_coordinates(batch, args, H_ori=720, W_ori=960, H=368, W=512):
    """Normalize 3D coordinates based on specified strategy."""
    weight_dtype = torch.float32
    
    # Calculate scale factors
    fx = 1 if W_ori / W > H_ori / H else H_ori / W_ori / (H / W)
    fy = W_ori / H_ori / (W / H) if W_ori / W > H_ori / H else 1
    
    if args.normalize_track:
        return batch["coords_normalized"].to(dtype=weight_dtype)
    
    elif args.normalize_track_first_frame:
        flow = batch["coords"][:, :, :args.num_frames, :, :].to(dtype=weight_dtype)
        targets = []
        
        for b in range(flow.size(0)):
            frame0 = flow[b, :, 0, :, :]  # [3, H, W]
            max_vals = frame0.view(3, -1).max(dim=1)[0]  # [3]
            min_vals = frame0.view(3, -1).min(dim=1)[0]  # [3]
            diff = (max_vals - min_vals).max().repeat(3)  # [3]
            diff[diff == 0] = 1.0  # Avoid division by zero
            
            normalized = batch["coords_delta"][b, :, :args.num_frames, :, :].to(dtype=weight_dtype) / diff.view(3, 1, 1, 1)
            targets.append(normalized)
        
        return torch.stack(targets, dim=0)
    
    elif args.normalize_track_z:
        flow = batch["coords"][:, :, :args.num_frames, :, :].to(dtype=weight_dtype)
        targets = []
        
        for b in range(flow.size(0)):
            frame0 = flow[b, :, 0, :, :].clone()  # [3, H, W]
            
            # Handle invalid z values
            frame0[2, :, :][torch.isnan(frame0[2, :, :])] = 1.0
            frame0[2, :, :][frame0[2, :, :] == 0] = 1.0
            frame0[2, :, :][torch.isinf(frame0[2, :, :])] = 1.0
            
            current_x_norm = frame0[2, :, :] / fx
            current_y_norm = frame0[2, :, :] / fy
            
            temp = batch["coords_delta"][b, :, :args.num_frames, :, :].to(dtype=weight_dtype)
            temp[0:1, :, :, :] = temp[0:1, :, :, :] / current_x_norm
            temp[1:2, :, :, :] = temp[1:2, :, :, :] / current_y_norm
            temp[2:3, :, :, :] = temp[2:3, :, :, :] / frame0[2:3, :, :]
            targets.append(temp)
        
        return torch.stack(targets, dim=0)
    
    else:
        # Default: relative to first frame
        targets = batch["coords"][:, :, :args.num_frames, :, :].to(dtype=weight_dtype)
        return targets - targets[:, :, 0:1, :, :]


def compute_loss(reconstructions, targets, posterior, args):
    """Compute reconstruction and KL divergence losses."""
    if args.rec_loss == "l2":
        rec_loss = F.mse_loss(reconstructions.float(), targets.float(), reduction="none")
    elif args.rec_loss == "l1":
        rec_loss = F.l1_loss(reconstructions.float(), targets.float(), reduction="none")
    else:
        raise ValueError(f"Invalid reconstruction loss type: {args.rec_loss}")
    
    nll_loss = torch.sum(rec_loss) / rec_loss.shape[0]
    kl_loss = torch.sum(posterior.kl()) / posterior.kl().shape[0]
    
    total_loss = nll_loss + args.kl_scale * kl_loss
    
    return total_loss, nll_loss, kl_loss


def create_projected_video(data, reconstructions, args, accelerator, H_ori=720, W_ori=960, H=368, W=512):
    """Create projected video from 3D coordinates."""
    from MoRe4D.utils.project_utils import project
    from torch_scatter import scatter
    
    # Calculate intrinsic parameters
    fx = 1 if W_ori / W > H_ori / H else H_ori / W_ori / (H / W)
    fy = W_ori / H_ori / (W / H) if W_ori / W > H_ori / H else 1
    
    intrinsic = torch.tensor([
        [fx, 0, 0.5],
        [0, fy, 0.5],
        [0, 0, 1]
    ]).to(accelerator.device)
    
    extrinsic = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).to(accelerator.device)
    
    n_frames = min(49, (len(data["coords"])//4)*4+1)
    recon_coords = reconstructions[0].permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, 3]
    
    # Apply inverse normalization if needed
    if args.normalize_track_first_frame or args.normalize_track_z:
        # Note: This would need the normalization parameters from validation
        pass  # Implementation depends on stored normalization parameters
    
    frames = []
    for i in range(n_frames):
        world_points = torch.from_numpy(recon_coords[i]).reshape(-1, 3).to(accelerator.device)
        
        predicted_2D, depth_2D = project(world_points, extrinsic, intrinsic)
        mask = ((predicted_2D[..., 0] >= 0) & (predicted_2D[..., 0] <= 1) &
                (predicted_2D[..., 1] >= 0) & (predicted_2D[..., 1] <= 1) & (depth_2D >= 0))
        
        if not mask.any():
            frames.append(np.zeros((H, W, 3), dtype=np.uint8))
            continue
        
        color_pc = torch.from_numpy(data["colors"]).to(accelerator.device)[mask, :]
        depth_2D_masked = depth_2D[mask]
        idx_pc = predicted_2D[mask, :]
        idx_xy = (idx_pc[:, 0]*W).floor().clamp(0, W-1) * H + (idx_pc[:, 1]*H).floor().clamp(0, H-1)
        
        # Handle depth conflicts
        unique_indices, inverse_indices = torch.unique(idx_xy, return_inverse=True)
        min_depth = torch.ones_like(unique_indices, dtype=depth_2D_masked.dtype) * depth_2D_masked.max()
        min_depth.index_reduce_(0, inverse_indices, depth_2D_masked, 'amin')
        mask_depth = (depth_2D_masked == min_depth[inverse_indices])
        
        color_pc = color_pc[mask_depth, :]
        idx_xy = idx_xy[mask_depth]
        
        color_image = scatter(color_pc, idx_xy.long(), dim=0, reduce="mean")
        if len(color_image) < H*W:
            color_image = torch.cat([color_image, torch.zeros((H*W-len(color_image), 3), device=accelerator.device)], dim=0)
        
        color_image = color_image.reshape(W, H, 3).transpose(0, 1)
        image_proj = color_image.cpu().numpy().astype(np.uint8)
        
        frames.append(image_proj)
    
    return np.stack(frames, axis=0)


@torch.no_grad()
def log_validation(encoder_prompt, decoder_prompt, vae, args, accelerator, weight_dtype, step, is_final_validation=False):
    """Run validation and save results."""
    logger.info("Running validation...")
    
    if is_final_validation:
        # Load models for final validation
        encoder_prompt = VAEEncoderadaptor()
        decoder_prompt = VAEDecoderadaptor()
        vae = AutoencoderKLWan.from_pretrained(args.vae_model_path)
        
        encoder_state_dict = torch.load(os.path.join(args.output_dir, "encoder_prompt", "pytorch_model.bin"))
        decoder_state_dict = torch.load(os.path.join(args.output_dir, "decoder_prompt", "pytorch_model.bin"))
        encoder_prompt.load_state_dict(encoder_state_dict)
        decoder_prompt.load_state_dict(decoder_state_dict)
    else:
        encoder_prompt = accelerator.unwrap_model(encoder_prompt)
        decoder_prompt = accelerator.unwrap_model(decoder_prompt)
        vae = accelerator.unwrap_model(vae)
    
    videos = []
    projected_videos = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    
    for i, data_path in enumerate(args.validation_sceneflow):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        targets = torch.from_numpy(data["coords"]).to(accelerator.device, weight_dtype)
        n_frames = min(49, (len(data["coords"])//4)*4+1)
        
        # Apply normalization (simplified version for validation)
        if not args.normalize_track:
            targets = targets - targets[0:1, :, :]
        
        targets = targets.reshape(targets.shape[0], 384, 512, 3)[:n_frames, :, :, :].permute(3, 0, 1, 2).unsqueeze(0)
        
        if args.normalize_track:
            targets = targets / targets.abs().amax(dim=(1, 2, 3, 4), keepdim=True)
        
        with inference_ctx:
            pseudo_video = encoder_prompt(targets)
            pseudo_video = pseudo_video * 2 - 1
            posterior = vae.encode(pseudo_video).latent_dist
            latents = posterior.sample()
            recon_video = vae.decode(latents).sample
            reconstructions = decoder_prompt(recon_video)
        
        videos.append(torch.cat([pseudo_video.cpu(), recon_video.cpu(), reconstructions.cpu()], dim=-2))
        
        # Create projected video
        projected_video = create_projected_video(data, reconstructions, args, accelerator)
        projected_videos.append(torch.from_numpy(projected_video).permute(0, 3, 1, 2).unsqueeze(0) / 255.0)
    
    # Save videos
    save_validation_videos(videos, projected_videos, args, step, is_final_validation)
    
    return videos, projected_videos


def save_validation_videos(videos, projected_videos, args, step, is_final_validation):
    """Save validation videos to disk."""
    tracker_key = "test" if is_final_validation else "validation"
    os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
    
    # Save regular videos
    for i, video in enumerate(videos):
        video = ((video[0].permute(1, 0, 2, 3) + 1) / 2).clamp(0, 1).to(torch.float32)
        video_path = os.path.join(args.output_dir, "sample", f"{tracker_key}_video_{step}_{i}.mp4")
        video_np = (video.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        imageio.mimwrite(video_path, video_np, fps=8)
    
    # Save projected videos
    for i, video in enumerate(projected_videos):
        video = video[0].to(torch.float32)
        video_path = os.path.join(args.output_dir, "sample", f"{tracker_key}_projected_video_{step}_{i}.mp4")
        video_np = (video.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        imageio.mimwrite(video_path, video_np, fps=8)


def setup_models_and_optimizer(args):
    """Initialize models and optimizer."""
    encoder_prompt = VAEEncoderadaptor()
    decoder_prompt = VAEDecoderadaptor()
    
    vae = AutoencoderKLWan.from_pretrained(
        args.vae_model_path,
        additional_kwargs={
            "latent_channels": 16,
            "temporal_compression_ratio": 4,
            "spatial_compression_ratio": 8
        }
    )
    
    # Set training/eval modes and gradient requirements
    encoder_prompt.requires_grad_(True).train()
    decoder_prompt.requires_grad_(True).train()
    vae.model.encoder.requires_grad_(False).eval()
    
    if args.finetune_vae_decoder:
        vae.model.decoder.requires_grad_(True).train()
    else:
        vae.model.decoder.requires_grad_(False).eval()
    
    # Setup optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
    else:
        optimizer_cls = torch.optim.AdamW
    
    params_to_optimize = [
        {"params": encoder_prompt.parameters(), "lr": args.learning_rate},
        {"params": decoder_prompt.parameters(), "lr": args.learning_rate},
    ]
    
    if args.finetune_vae_decoder:
        params_to_optimize.append({"params": vae.model.decoder.parameters(), "lr": args.learning_rate * 0.1})
    
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    return encoder_prompt, decoder_prompt, vae, optimizer


def setup_save_load_hooks(accelerator, args):
    """Setup model save/load hooks for checkpointing."""
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            while len(weights) > 0:
                weights.pop()
                model = models.pop()
                
                if isinstance(model, AutoencoderKLWan):
                    if args.finetune_vae_decoder:
                        sub_dir = os.path.join(output_dir, "vae")
                        os.makedirs(sub_dir, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(sub_dir, "pytorch_model.bin"))
                elif isinstance(model, VAEEncoderadaptor):
                    sub_dir = os.path.join(output_dir, "encoder_prompt")
                    os.makedirs(sub_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(sub_dir, "pytorch_model.bin"))
                elif isinstance(model, VAEDecoderadaptor):
                    sub_dir = os.path.join(output_dir, "decoder_prompt")
                    os.makedirs(sub_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(sub_dir, "pytorch_model.bin"))
    
    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            
            if isinstance(model, AutoencoderKLWan):
                vae_path = os.path.join(input_dir, "vae", "pytorch_model.bin")
                if os.path.exists(vae_path):
                    model.load_state_dict(torch.load(vae_path))
            elif isinstance(model, VAEDecoderadaptor):
                decoder_path = os.path.join(input_dir, "decoder_prompt", "pytorch_model.bin")
                if os.path.exists(decoder_path):
                    model.load_state_dict(torch.load(decoder_path))
            elif isinstance(model, VAEEncoderadaptor):
                encoder_path = os.path.join(input_dir, "encoder_prompt", "pytorch_model.bin")
                if os.path.exists(encoder_path):
                    model.load_state_dict(torch.load(encoder_path))
    
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def train_step(batch, encoder_prompt, decoder_prompt, vae, optimizer, accelerator, loss_tracker, args):
    """Execute single training step."""
    targets = normalize_coordinates(batch, args)
    
    with accelerator.accumulate([encoder_prompt, decoder_prompt, vae]):
        # Forward pass
        pseudo_video = accelerator.unwrap_model(encoder_prompt)(targets)
        pseudo_video = pseudo_video * 2 - 1
        
        with torch.no_grad():
            posterior = accelerator.unwrap_model(vae).encode_memory_saver(pseudo_video).latent_dist
            latents = posterior.sample()
            del pseudo_video
            torch.cuda.empty_cache()
        
        if not args.finetune_vae_decoder:
            with torch.no_grad():
                recon_video = accelerator.unwrap_model(vae).decode_memory_saver(latents).sample
        else:
            recon_video = accelerator.unwrap_model(vae).decode_memory_saver(latents).sample
        
        del latents
        torch.cuda.empty_cache()
        
        reconstructions = accelerator.unwrap_model(decoder_prompt)(recon_video)
        
        # Compute loss
        loss, nll_loss, kl_loss = compute_loss(reconstructions, targets, posterior, args)
        
        # Check if batch should be skipped
        if should_skip_batch(loss, loss_tracker, args):
            accelerator.backward(loss)
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            return None, True  # Skip flag
        
        loss_tracker.update(loss, batch_size=targets.size(0))
        
        # Backward pass
        accelerator.backward(loss)
        
        # Gradient clipping
        if accelerator.sync_gradients:
            unwrapped_vae = accelerator.unwrap_model(vae).model
            params_to_clip = list(encoder_prompt.parameters()) + list(decoder_prompt.parameters())
            if args.finetune_vae_decoder:
                params_to_clip += list(unwrapped_vae.decoder.parameters())
            
            grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            
            if grad_norm is None or not math.isfinite(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm):
                logger.warning(f"Skip batch: invalid grad_norm={grad_norm}")
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                return None, True
        
        logs = {
            "loss": loss.detach().item(),
            "nll_loss": nll_loss.detach().item(),
            "kl_loss": (kl_loss.detach().item() * args.kl_scale),
            "window_avg_loss": loss_tracker.get_window_average(),
        }
        
        return logs, False


def parse_args(input_args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WAN VAE fine-tuning script.")
    
    # Model and data paths
    parser.add_argument("--vae_model_path", type=str, required=True, help="Path to pretrained VAE model.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of scene flow data.")
    parser.add_argument("--video_column", type=str, required=True, help="Path to video list file.")
    parser.add_argument("--data_posfix", type=str, default="_render", help="Postfix for data files.")
    parser.add_argument("--output_dir", type=str, default="outputs/wan_vae_finetuned", help="Output directory.")
    
    # Training configuration
    parser.add_argument("--finetune_vae_decoder", action="store_true", help="Whether to finetune the VAE decoder.")
    parser.add_argument("--normalize_track", action="store_true", help="Whether to normalize the track coordinates.")
    parser.add_argument("--normalize_track_first_frame", action="store_true", help="Whether to normalize coordinates by first frame.")
    parser.add_argument("--normalize_track_z", action="store_true", help="Whether to normalize coordinates by depth.")
    
    # Data parameters
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames to use.")
    
    # Loss skipping parameters
    parser.add_argument("--loss_skip_min_samples", type=int, default=100, help="Minimum samples before enabling loss skipping.")
    parser.add_argument("--loss_skip_std_multiplier", type=float, default=6.0, help="Skip batch if loss > mean + std_multiplier * std.")
    parser.add_argument("--loss_skip_multiplier", type=float, default=10.0, help="Skip batch if loss > multiplier * mean (when std is small).")
    parser.add_argument("--loss_skip_absolute_threshold", type=float, default=1e7, help="Absolute threshold for skipping batches.")
    parser.add_argument("--loss_tracker_window_size", type=int, default=1000, help="Window size for loss tracking.")
    
    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=4.5e-6, help="Initial learning rate.")
    parser.add_argument("--scale_lr", action="store_true", help="Scale learning rate by batch size and GPUs.")
    
    # Learning rate scheduling
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="The scheduler type to use.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps.")
    
    # Optimizer parameters
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1 parameter.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 parameter.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Adam weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon parameter.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether to use 8-bit Adam.")
    parser.add_argument("--set_grads_to_none", action="store_true", help="Set grads to None instead of zero.")
    
    # Loss parameters
    parser.add_argument("--rec_loss", type=str, default="l1", choices=["l1", "l2"], help="Reconstruction loss type.")
    parser.add_argument("--kl_scale", type=float, default=1e-6, help="Scale factor for KL divergence loss.")
    
    # Validation and checkpointing
    parser.add_argument("--validation_steps", type=int, default=500, help="Run validation every X steps.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save checkpoint every X steps.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--validation_sceneflow", type=str, default=None, nargs="+", help="Path to validation data.")
    
    # System parameters
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Mixed precision mode.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Logging integration.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Enable xformers optimization.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to pretrained models.")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    # Handle environment variables
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args


def main(args):
    """Main training function."""
    # Setup accelerator and logging
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directories
    if accelerator.is_main_process and args.output_dir is not None:
        for subdir in ["", "encoder_prompt", "decoder_prompt", "vae"]:
            os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    # Setup models and optimizer
    encoder_prompt, decoder_prompt, vae, optimizer = setup_models_and_optimizer(args)
    
    # Setup save/load hooks
    setup_save_load_hooks(accelerator, args)
    
    # Enable optimizations
    if args.gradient_checkpointing and hasattr(vae, "enable_gradient_checkpointing"):
        vae.enable_gradient_checkpointing()
    
    if args.enable_xformers_memory_efficient_attention and hasattr(vae, "enable_xformers_memory_efficient_attention"):
        vae.enable_xformers_memory_efficient_attention()
    
    # Log parameter counts
    total_params = sum(p.numel() for p in encoder_prompt.parameters()) + \
                  sum(p.numel() for p in decoder_prompt.parameters()) + \
                  sum(p.numel() for p in vae.model.parameters())
    
    trainable_params = sum(p.numel() for p in encoder_prompt.parameters() if p.requires_grad) + \
                      sum(p.numel() for p in decoder_prompt.parameters() if p.requires_grad)
    
    if args.finetune_vae_decoder:
        trainable_params += sum(p.numel() for p in vae.model.decoder.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Setup dataset and dataloader
    train_dataset = VAEDataset(
        data_root=args.data_root,
        video_column=args.video_column,
        posfix=args.data_posfix,
        max_frames=args.num_frames,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        overrode_max_train_steps = False
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with accelerator
    encoder_prompt, decoder_prompt, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        encoder_prompt, decoder_prompt, vae, optimizer, train_dataloader, lr_scheduler
    )
    
    # Setup mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Recalculate training parameters
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Initialize tracking
    if accelerator.is_main_process:
        tracker_config = {k: v for k, v in vars(args).items() if k != "validation_sceneflow"}
        accelerator.init_trackers("wan_vae_finetuning", config=tracker_config)
        writer = SummaryWriter(log_dir=logging_dir)
    
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Initialize training state
    global_step = 0
    first_epoch = 0
    loss_tracker = LossTracker(window_size=args.loss_tracker_window_size)
    skipped_batches = 0
    
    # Handle checkpoint resuming
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        
        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new training.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
    
    # Training loop
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        # Set training modes
        encoder_prompt.train()
        decoder_prompt.train()
        unwrapped_vae = accelerator.unwrap_model(vae).model
        unwrapped_vae.encoder.eval()
        
        if args.finetune_vae_decoder:
            unwrapped_vae.decoder.train()
        else:
            unwrapped_vae.decoder.eval()
        
        for step, batch in enumerate(train_dataloader):
            try:
                # Execute training step
                logs, skip_flag = train_step(batch, encoder_prompt, decoder_prompt, vae, optimizer, accelerator, loss_tracker, args)
                
                if skip_flag:
                    skipped_batches += 1
                    continue
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                # Update progress
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    # Add learning rate to logs
                    logs["lr"] = lr_scheduler.get_last_lr()[0]
                    accelerator.log({"train_loss": logs["loss"]}, step=global_step)
                    
                    # Checkpointing
                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # Clean up old checkpoints
                            if args.checkpoints_total_limit is not None:
                                checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    for checkpoint in checkpoints[:num_to_remove]:
                                        shutil.rmtree(os.path.join(args.output_dir, checkpoint))
                            
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                    
                    # Validation
                    if args.validation_sceneflow and global_step % args.validation_steps == 0:
                        log_validation(encoder_prompt, decoder_prompt, vae, args, accelerator, weight_dtype, global_step)
                
                progress_bar.set_postfix(**logs)
                
                if global_step >= args.max_train_steps:
                    break
                    
            except Exception as e:
                logger.error(f"Exception in step {step} of epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Training completion
    if accelerator.is_main_process:
        logger.info(f"Training completed. Total skipped batches: {skipped_batches}")
        logger.info(f"Final global average loss: {loss_tracker.get_global_average():.6f}")
        logger.info(f"Final window average loss: {loss_tracker.get_window_average():.6f}")
    
    # Final model saving
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        encoder_prompt = accelerator.unwrap_model(encoder_prompt)
        decoder_prompt = accelerator.unwrap_model(decoder_prompt)
        vae = accelerator.unwrap_model(vae)
        
        torch.save(encoder_prompt.state_dict(), os.path.join(args.output_dir, "encoder_prompt", "pytorch_model.bin"))
        torch.save(decoder_prompt.state_dict(), os.path.join(args.output_dir, "decoder_prompt", "pytorch_model.bin"))
        
        if args.finetune_vae_decoder:
            torch.save(vae.state_dict(), os.path.join(args.output_dir, "vae", "pytorch_model.bin"))
    
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)