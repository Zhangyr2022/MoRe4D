#!/usr/bin/env python
# coding=utf-8

import os
import torch
import pickle
import argparse
import numpy as np
import imageio
import glob
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from torch_scatter import scatter
from MoRe4D.models.wan_vae import AutoencoderKLWan
from MoRe4D.models.trajectory_module import VAEEncoderadaptor, VAEDecoderadaptor
from MoRe4D.utils.gaussian_splatting import gs_render
from MoRe4D.utils.project_utils import project


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inference for WAN VAE model")
    parser.add_argument("--vae_model_path", type=str, required=True, help="Path to pretrained VAE model")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory with trained encoder/decoder checkpoints")
    parser.add_argument("--input_sceneflow", type=str, nargs="+", required=True, help="Path to input scene flow data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for visualizations")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames to process")
    parser.add_argument("--normalize_track", action="store_true", help="Globally normalize track coordinates")
    parser.add_argument("--normalize_track_first_frame", action="store_true", help="Normalize by first frame")
    parser.add_argument("--normalize_track_z", action="store_true", help="Normalize by Z axis")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--render_type", type=str, default="both", choices=["project", "3dgs", "both"])
    parser.add_argument("--gs_scale", type=float, default=0.0001, help="Scale parameter for 3D Gaussian Splatting")
    return parser.parse_args()


def get_pkl_files(input_paths):
    """Retrieve all .pkl files from the given paths."""
    pkl_files = []
    for path in input_paths:
        if os.path.isdir(path):
            dir_pkls = glob.glob(os.path.join(path, "*.pkl"))
            if dir_pkls:
                pkl_files.extend(dir_pkls)
            else:
                for subdir in os.listdir(path):
                    subdir_path = os.path.join(path, subdir)
                    if os.path.isdir(subdir_path):
                        sub_pkls = glob.glob(os.path.join(subdir_path, "*.pkl"))
                        pkl_files.extend([p for p in sub_pkls if "dense_3d_track.pkl" in p])
        else:
            pkl_files.append(path)
    return pkl_files


def load_models(args, device, weight_dtype):
    """Load VAE model and associated encoder/decoder components."""
    encoder_prompt = VAEEncoderadaptor().to(device).to(weight_dtype).eval()
    decoder_prompt = VAEDecoderadaptor().to(device).to(weight_dtype).eval()
    vae = AutoencoderKLWan.from_pretrained(args.vae_model_path).to(device).to(weight_dtype).eval()

    encoder_path = os.path.join(args.ckpt_dir, "encoder_prompt", "pytorch_model.bin")
    decoder_path = os.path.join(args.ckpt_dir, "decoder_prompt", "pytorch_model.bin")
    
    encoder_prompt.load_state_dict(torch.load(encoder_path))
    decoder_prompt.load_state_dict(torch.load(decoder_path))

    vae_path = os.path.join(args.ckpt_dir, "vae", "pytorch_model.bin")
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path))

    return encoder_prompt, decoder_prompt, vae


def setup_camera_params(H, W, H_ori, W_ori, device):
    """Set up intrinsic and extrinsic camera parameters."""
    if W_ori / W > H_ori / H:
        fx, fy = 1, W_ori / H_ori / (W / H)
    else:
        fy, fx = 1, H_ori / W_ori / (H / W)

    intrinsic = torch.Tensor([[fx, 0, 0.5], [0, fy, 0.5], [0, 0, 1]]).to(device)
    extrinsic = torch.eye(4).to(device)
    return intrinsic, extrinsic


def prepare_data(data, device, weight_dtype, n_frames, H=384, W=512):
    """Load and preprocess coordinate and color data."""
    for key in data:
        if isinstance(data[key], np.ndarray):
            if data[key].shape[0] == 1:
                data[key] = data[key][0]
            data[key] = torch.from_numpy(data[key])
        elif torch.is_tensor(data[key]) and data[key].shape[0] == 1:
            data[key] = data[key][0]
    
    targets = data["coords"].to(device, weight_dtype)
    first_frame = targets[0:1, :, :].clone()
    
    targets = targets.reshape(targets.shape[0], H, W, 3)[:n_frames, :, :, :]
    targets = targets.permute(3, 0, 1, 2).unsqueeze(0)  # [B, C, T, H, W]
    
    return targets, first_frame, data["colors"]


def normalize_targets(targets, first_frame, args, H=384, W=512):
    """Apply normalization based on args configuration."""
    if args.normalize_track:
        return targets
    
    targets = targets - first_frame.reshape(1, 3, 1, H, W)
    
    if args.normalize_track_first_frame:
        frame0 = first_frame.reshape(H, W, 3).permute(2, 0, 1)
        max_vals = frame0.view(3, -1).max(dim=1)[0]
        min_vals = frame0.view(3, -1).min(dim=1)[0]
        diff = (max_vals - min_vals).max().repeat(3)
        diff[diff == 0] = 1.0
        targets = targets / diff.view(3, 1, 1, 1, 1)
        return targets, diff
    
    elif args.normalize_track_z:
        H_ori, W_ori = 720, 960
        fx, fy = _compute_focal_lengths(H, W, H_ori, W_ori)
        frame0 = first_frame.reshape(H, W, 3).permute(2, 0, 1)
        frame0[2, :, :] = _safe_depth(frame0[2, :, :])
        
        targets[:, 0, :, :, :] /= (frame0[2, :, :] / fx)
        targets[:, 1, :, :, :] /= (frame0[2, :, :] / fy)
        targets[:, 2, :, :, :] /= frame0[2:3, :, :]
        return targets, frame0
    
    return targets, None


def _compute_focal_lengths(H, W, H_ori, W_ori):
    """Compute focal lengths based on aspect ratios."""
    if W_ori / W > H_ori / H:
        return 1, W_ori / H_ori / (W / H)
    else:
        return H_ori / W_ori / (H / W), 1


def _safe_depth(depth):
    """Handle invalid depth values."""
    depth[torch.isnan(depth) | (depth == 0) | torch.isinf(depth)] = 1.0
    return depth


def denormalize_reconstruction(reconstructions, targets, args, diff_or_frame0):
    """Reverse normalization on reconstructed data."""
    if args.normalize_track_first_frame and diff_or_frame0 is not None:
        return reconstructions * diff_or_frame0.view(3, 1, 1, 1), targets * diff_or_frame0.view(3, 1, 1, 1)
    return reconstructions, targets


def render_with_project(world_points, extrinsic, intrinsic, colors, H, W, device):
    """Render point cloud using projection."""
    predicted_2D, depth_2D = project(world_points, extrinsic, intrinsic)
    mask = (predicted_2D[..., 0] >= 0) & (predicted_2D[..., 0] <= 1) & \
           (predicted_2D[..., 1] >= 0) & (predicted_2D[..., 1] <= 1) & (depth_2D >= 0)

    color_pc = colors[mask, :]
    depth_2D = depth_2D[mask]
    idx_xy = ((predicted_2D[mask, 0] * W).floor().clamp(0, W - 1) * H + 
              (predicted_2D[mask, 1] * H).floor().clamp(0, H - 1)).long()

    unique_indices, inverse_indices = torch.unique(idx_xy, return_inverse=True)
    min_depth = torch.ones_like(unique_indices, dtype=depth_2D.dtype) * depth_2D.max()
    min_depth.index_reduce_(0, inverse_indices, depth_2D, 'amin')
    mask_depth = depth_2D == min_depth[inverse_indices]

    color_pc = color_pc[mask_depth, :]
    idx_xy = idx_xy[mask_depth]

    color_image = scatter(color_pc, idx_xy, dim=0, reduce="mean")
    if len(color_image) < H * W:
        padding = torch.zeros((H * W - len(color_image), 3), device=device)
        color_image = torch.cat([color_image, padding], dim=0)

    color_image = color_image.reshape(W, H, 3).transpose(0, 1)
    return Image.fromarray(color_image.cpu().numpy().astype(np.uint8))


def render_with_gs(world_points, extrinsic, intrinsic, colors, H, W, device, scale):
    """Render point cloud using 3D Gaussian Splatting."""
    scale_tensor = torch.Tensor([scale, scale, scale]).to(device)
    rotation = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(device)
    
    color_normalized = colors.float() / 255.0 if colors.max() > 1.0 else colors.float()
    rendered_images = gs_render(
        intrinsic, extrinsic, [H, W], world_points, scale_tensor, rotation,
        color_normalized, torch.ones((H * W,)).to(device)
    )

    color_image = rendered_images[0].permute(1, 2, 0).detach().cpu() * 255
    return Image.fromarray(color_image.numpy().astype(np.uint8))


def save_reconstructions(reconstructions, output_dir, name, n_frames):
    """Save reconstruction frames as PNG images."""
    recon_vis = (reconstructions - reconstructions.min()) / \
                (reconstructions.max() - reconstructions.min()) * 255.0
    recon_vis = recon_vis.numpy().astype(np.uint8)
    
    output_path = os.path.join(output_dir, name)
    os.makedirs(output_path, exist_ok=True)
    
    for t in range(n_frames):
        frame = Image.fromarray(recon_vis[0, :, t, :, :].transpose(1, 2, 0))
        frame.save(os.path.join(output_path, f"frame_{t:03d}_reconstruction.png"))


def save_videos(gt_frames, pred_frames, output_dir, name, render_type):
    """Save rendered frames as video files."""
    if render_type in ["project", "both"] and gt_frames.get("project"):
        gt_video = np.stack(gt_frames["project"], axis=0)
        pred_video = np.stack(pred_frames["project"], axis=0)
        imageio.mimwrite(os.path.join(output_dir, f"{name}_gt_project.mp4"), gt_video, fps=8)
        imageio.mimwrite(os.path.join(output_dir, f"{name}_pred_project.mp4"), pred_video, fps=8)
        side_by_side = np.concatenate([gt_video, pred_video], axis=2)
        imageio.mimwrite(os.path.join(output_dir, f"{name}_comparison_project.mp4"), side_by_side, fps=8)
    
    if render_type in ["3dgs", "both"] and gt_frames.get("3dgs"):
        gt_video = np.stack(gt_frames["3dgs"], axis=0)
        pred_video = np.stack(pred_frames["3dgs"], axis=0)
        imageio.mimwrite(os.path.join(output_dir, f"{name}_gt_3dgs.mp4"), gt_video, fps=8)
        imageio.mimwrite(os.path.join(output_dir, f"{name}_pred_3dgs.mp4"), pred_video, fps=8)
        side_by_side = np.concatenate([gt_video, pred_video], axis=2)
        imageio.mimwrite(os.path.join(output_dir, f"{name}_comparison_3dgs.mp4"), side_by_side, fps=8)


def inference(args):
    """Main inference function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    pkl_files = get_pkl_files(args.input_sceneflow)
    if not pkl_files:
        print(f"No .pkl files found in: {args.input_sceneflow}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    encoder_prompt, decoder_prompt, vae = load_models(args, device, weight_dtype)
    intrinsic, extrinsic = setup_camera_params(368, 512, 720, 960, device)

    for idx, data_path in enumerate(pkl_files):
        print(f"Processing [{idx + 1}/{len(pkl_files)}]: {data_path}")
        
        # Extract output name
        name = data_path.split("/")[-2] if "dense_3d_track.pkl" in data_path else data_path.split("/")[-1].split("_")[0]
        
        # Load data
        try:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {data_path}: {e}")
            continue
        
        if "coords" not in data or "colors" not in data:
            print(f"Missing required fields in {data_path}")
            continue
        
        # Prepare data
        n_frames = min(args.num_frames, (data["coords"].shape[1] // 4) * 4 + 1)
        targets, first_frame, colors = prepare_data(data, device, weight_dtype, n_frames)
        
        # Normalize
        norm_result = normalize_targets(targets, first_frame, args)
        targets = norm_result[0]
        norm_params = norm_result[1] if len(norm_result) > 1 else None
        
        # Inference
        with torch.no_grad(), torch.autocast(device.type, dtype=weight_dtype):
            pseudo_video = encoder_prompt(targets)
            pseudo_video = pseudo_video * 2 - 1
            latents = vae.encode(pseudo_video).latent_dist.sample()
            recon_video = vae.decode(latents).sample
            reconstructions = decoder_prompt(recon_video).float().cpu()
            
        # Save reconstructions
        save_reconstructions(reconstructions, args.output_dir, name, n_frames)
        
        # Prepare rendering data
        original_coords = (targets.float().cpu()[0].permute(1, 2, 3, 0).numpy())  # [T, H, W, 3]
        recon_coords = (reconstructions.float()[0].permute(1, 2, 3, 0).numpy())  # [T, H, W, 3]
        
        gt_frames = {"project": [], "3dgs": []}
        pred_frames = {"project": [], "3dgs": []}
        colors = colors.to(device)
        
        # Render frames
        for t in range(n_frames):
            world_points_gt = torch.from_numpy(original_coords[t]).reshape(-1, 3).to(device)
            world_points_pred = torch.from_numpy(recon_coords[t]).reshape(-1, 3).to(device)
            
            if args.render_type in ["project", "both"]:
                gt_frames["project"].append(render_with_project(world_points_gt, extrinsic, intrinsic, colors, 368, 512, device))
                pred_frames["project"].append(render_with_project(world_points_pred, extrinsic, intrinsic, colors, 368, 512, device))
            
            if args.render_type in ["3dgs", "both"]:
                gt_frames["3dgs"].append(render_with_gs(world_points_gt, extrinsic, intrinsic, colors, 368, 512, device, args.gs_scale))
                pred_frames["3dgs"].append(render_with_gs(world_points_pred, extrinsic, intrinsic, colors, 368, 512, device, args.gs_scale))
        
        # Save videos
        save_videos(gt_frames, pred_frames, args.output_dir, name, args.render_type)
    
    print("Inference completed!")


if __name__ == "__main__":
    args = parse_args()
    inference(args)