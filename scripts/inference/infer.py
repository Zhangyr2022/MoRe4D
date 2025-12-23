#!/usr/bin/env python
# coding=utf-8

import os
import sys
import gc
import math
import cv2
import numpy as np
import torch
import argparse
import logging
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from typing import Dict, Any, List, Tuple
import torch.nn.functional as F
from torchvision import transforms
from omegaconf import OmegaConf
import imageio

from transformers import AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from torch_scatter import scatter

# Add project paths
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), 
                os.path.dirname(os.path.dirname(current_file_path)), 
                os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from MoRe4D.models import (AutoencoderKLWan, WanT5EncoderModel, 
                          WanTransformer3DModel, CLIPModel, WanTransformer4DModel)
from MoRe4D.pipeline import WanFunControlPipeline, WanFunInpaintPipeline
from MoRe4D.utils.lora_utils import create_network, merge_lora, unmerge_lora
from MoRe4D.utils.utils import (filter_kwargs, get_image_latent, get_video_to_video_latent, 
                                save_videos_grid, get_image_to_video_latent, get_image_to_flow_video_latent)
from MoRe4D.models.cache_utils import get_teacache_coefficients
from MoRe4D.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from MoRe4D.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from MoRe4D.models.trajectory_module import VAEEncoderadaptor, VAEDecoderadaptor
from MoRe4D.utils.project_utils import project
from unidepth.models import UniDepthV2old
from MoRe4D.utils.gaussian_splatting import gs_render

# Constants
TRAJECTORY_TYPES = ["mix1", "mix2", "surrounding", "anti-surrounding", "circular", 
                   "forward_backward", "y_moving", "x_moving", "circle_rotating", "static", "camera_rotate"]
DEFAULT_H_ORI, DEFAULT_W_ORI = 540, 960


def get_logger(name, log_level="INFO"):
    """Initialize logger with StreamHandler."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger


def load_prompts(file_path: str) -> List[str]:
    """Load prompts from a text file."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_videos(file_path: str) -> List[Path]:
    """Load video paths from a text file."""
    data_dir = Path(file_path).parent
    with open(file_path, "r") as f:
        return [Path(str(data_dir / line.strip())) for line in f.readlines() if line.strip()]


class TwoStageDataset:
    """Dataset for two-stage video generation pipeline."""
    
    def __init__(self, data_root: str, caption_column: str, video_column: str,
                 device: torch.device, max_num_frames: int, height: int, width: int,
                 max_samples: int = None):
        super().__init__()
        self.logger = get_logger("inference", "INFO")
        data_root = Path(data_root)
        
        # Load and shuffle data
        import random
        self.prompts = load_prompts(data_root / caption_column)
        self.videos = load_videos(data_root / video_column)
        
        rand_idx = list(range(len(self.videos)))
        random.shuffle(rand_idx)
        self.prompts = [self.prompts[i] for i in rand_idx]
        self.videos = [self.videos[i] for i in rand_idx]
        
        # Sort by video names
        self.video_names = [video.stem for video in self.videos]
        sorted_indices = sorted(range(len(self.video_names)), key=lambda i: self.video_names[i])
        self.prompts = [self.prompts[i] for i in sorted_indices]
        self.video_names = [self.video_names[i] for i in sorted_indices]
        self.videos = [self.videos[i] for i in sorted_indices]
        
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.device = device
        self.__image_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        
        # Validation
        if len(self.videos) != len(self.prompts):
            raise ValueError(f"Number of prompts and videos must match: {len(self.prompts)} vs {len(self.videos)}")
        
        if any(not path.is_file() for path in self.videos):
            missing_file = next(path for path in self.videos if not path.is_file())
            raise ValueError(f"Video file not found: {missing_file}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index
        
        prompt = self.prompts[index]
        video = self.videos[index]
        _, image, fps = self.load_first_frame(video)
        image = self.image_transform(image)
        
        return {
            "prompt": prompt,
            "image": image,
            "video_path": video,
            "video_fps": fps
        }
    
    def __len__(self) -> int:
        return len(self.videos)
    
    def load_first_frame(self, video: Path) -> Tuple[None, torch.Tensor, None]:
        """Extract and process the first frame from a video."""
        cap = cv2.VideoCapture(str(video))
        success, first_frame = cap.read()
        cap.release()
        
        if not success:
            raise ValueError(f"Cannot read video file: {video}")
        
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_frame = Image.fromarray(first_frame)
        first_frame = first_frame.resize((self.width, self.height), Image.BILINEAR)
        first_frame = np.array(first_frame).astype(np.float32)
        image = torch.from_numpy(first_frame).permute(2, 0, 1).unsqueeze(0)
        return None, image, None
    
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


def get_intrinsic_matrix(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Compute intrinsic camera matrix based on resolution."""
    H_ori, W_ori = DEFAULT_H_ORI, DEFAULT_W_ORI
    if W_ori / W > H_ori / H:
        fx = 1.0
        fy = W_ori / H_ori / (W / H)
    else:
        fy = 1.0
        fx = H_ori / W_ori / (H / W)
    
    intrinsic = torch.Tensor([
        [fx, 0, 0.5],
        [0, fy, 0.5],
        [0, 0, 1]
    ]).to(device)
    return intrinsic


def back_project_coords(depth_map: torch.Tensor, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Back-project depth map to 3D coordinates."""
    depth_map = F.interpolate(depth_map.unsqueeze(0).unsqueeze(0), size=(H, W), 
                             mode='bilinear', align_corners=False)[0, 0]
    
    intrinsic = get_intrinsic_matrix(H, W, device)
    
    u = torch.linspace(0, 1, W, device=device)
    v = torch.linspace(0, 1, H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    
    pixels = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
    K_inv = torch.inverse(intrinsic.cpu()).to(device)
    rays = pixels @ K_inv.T
    
    points_3d = rays * depth_map.unsqueeze(-1)
    return points_3d


def inverse_flow_norm_transform_no_diff(rel_flow: torch.Tensor, 
                                        first_frame_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse normalize flow using first frame coordinates."""
    rel_flow = rel_flow.permute((0, 2, 1, 3, 4))
    B, F, C, H, W = rel_flow.shape
    device = rel_flow.device
    
    first_frame_coords = first_frame_coords[:, :, 0, :, :].permute((0, 2, 3, 1))
    if first_frame_coords.dim() == 4 and first_frame_coords.size(0) == 1:
        first_frame_coords = first_frame_coords.expand(B, H, W, 3)
    
    frame0_flat = first_frame_coords.permute(0, 3, 1, 2).reshape(B, 3, -1)
    max_vals = frame0_flat.max(dim=2).values
    min_vals = frame0_flat.min(dim=2).values
    diff = (max_vals - min_vals).max(dim=1)[0].repeat((1, 3))
    diff = torch.where(diff == 0, torch.ones_like(diff), diff)
    
    frame0_normalized = first_frame_coords.permute(0, 3, 1, 2) / diff.view(B, 3, 1, 1)
    recovered_flow_normalized = rel_flow + frame0_normalized.unsqueeze(1)
    recovered_flow = recovered_flow_normalized * diff.view(B, 1, 3, 1, 1)
    recovered_flow = recovered_flow.permute((0, 2, 1, 3, 4))
    return recovered_flow, diff


def render_with_project(world_points: torch.Tensor, extrinsic: torch.Tensor, 
                       intrinsic: torch.Tensor, colors: torch.Tensor, 
                       H: int, W: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Render 3D points using projection."""
    predicted_2D, depth_2D = project(world_points, extrinsic, intrinsic)
    
    mask = (predicted_2D[..., 0] >= 0) * (predicted_2D[..., 0] <= 1) * \
           (predicted_2D[..., 1] >= 0) * (predicted_2D[..., 1] <= 1) * (depth_2D >= 0)
    
    if mask.sum() > 0:
        color_pc = colors[mask, :]
        depth_2D = depth_2D[mask]
        idx_pc = predicted_2D[mask, :]
        idx_xy = (idx_pc[:, 0]*W).floor().clamp(0, W-1) * H + (idx_pc[:, 1]*H).floor().clamp(0, H-1)
        
        unique_indices, inverse_indices = torch.unique(idx_xy, return_inverse=True)
        min_depth = torch.ones_like(unique_indices, dtype=depth_2D.dtype) * depth_2D.max()
        min_depth.index_reduce_(0, inverse_indices, depth_2D, 'amin')
        mask_depth = (depth_2D == min_depth[inverse_indices])
        
        color_pc = color_pc[mask_depth, :]
        idx_xy = idx_xy[mask_depth]
        
        color_image = scatter(color_pc, idx_xy.long(), dim=0, reduce="mean")
        if len(color_image) < H*W:
            color_image = torch.cat([color_image, torch.zeros((H*W-len(color_image), 3), device=device)], dim=0)
        
        color_image = color_image.reshape(W, H, 3).transpose(0, 1)
    else:
        color_image = torch.zeros((H, W, 3), device=device)
    
    image_proj = color_image.cpu().numpy().astype(np.uint8)
    mask = (image_proj.sum(-1) == 0)
    
    return image_proj, mask


def render_with_gs(world_points: torch.Tensor, extrinsic: torch.Tensor, 
                  intrinsic: torch.Tensor, colors: torch.Tensor, 
                  H: int, W: int, device: torch.device, scale: float = 0.0001) -> np.ndarray:
    """Render 3D points using Gaussian splatting."""
    scale_tensor = torch.Tensor([scale, scale, scale]).to(device)
    rotation = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(device)
    
    rendered_images = gs_render(
        intrinsic, extrinsic, [H, W], world_points, scale_tensor, rotation,
        colors.float()/255.0 if colors.max() > 1.0 else colors.float(),
        torch.ones((H*W,)).to(device)
    )
    
    color_image = rendered_images[0].permute(1, 2, 0).detach().cpu() * 255
    return color_image.numpy().astype(np.uint8)


# Camera trajectory functions
def generate_static_trajectory(n_frames: int) -> List[torch.Tensor]:
    """Generate static camera trajectory."""
    extrinsic = torch.eye(4).float()
    return [extrinsic for _ in range(n_frames)]


def generate_forward_backward_trajectory(center: np.ndarray, n_frames: int, 
                                        radius_base: float = 0.3, z_progress: bool = True) -> List[torch.Tensor]:
    """Generate forward-backward camera motion."""
    extrinsics = []
    for i in range(n_frames):
        cam_x = cam_y = 0.0
        if i < n_frames // 4:
            cam_z = radius_base * i / n_frames if z_progress else 0.0
        elif i < 3 * n_frames // 4:
            cam_z = 0.5 * radius_base - radius_base * i / n_frames if z_progress else 0.0
        else:
            cam_z = -radius_base + radius_base * i / n_frames if z_progress else 0.0
        
        extrinsic = torch.eye(4).float()
        extrinsic[:3, 3] = torch.tensor([cam_x, cam_y, cam_z])
        extrinsics.append(extrinsic)
    
    return extrinsics


def generate_circle_rotating_trajectory(center: np.ndarray, n_frames: int, 
                                       radius_base: float = 0.3, z_progress: bool = True) -> List[torch.Tensor]:
    """Generate circular rotation camera trajectory."""
    extrinsics = []
    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        
        cam_x = radius_base * math.cos(angle)
        cam_y = radius_base * math.sin(angle)
        cam_z = 3 * radius_base * i / n_frames if z_progress else 0.0
        
        cam_pos = np.array([cam_x, cam_y, cam_z])
        target = np.array([center[0], center[1], center[2]])
        up = np.array([0, 1, 0])
        
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, forward)
        up_corrected = up_corrected / np.linalg.norm(up_corrected)
        
        R = np.array([-right, up_corrected, forward]).T
        
        extrinsic = torch.eye(4).float()
        extrinsic[:3, :3] = torch.from_numpy(R)
        extrinsic[:3, 3] = torch.from_numpy(cam_pos)
        extrinsics.append(extrinsic)
    
    return extrinsics


def generate_surrounding_trajectory(center: np.ndarray, n_frames: int) -> List[torch.Tensor]:
    """Generate surrounding camera trajectory."""
    extrinsics = []
    for i in range(n_frames):
        angle = -math.pi * i / n_frames / 4 - math.atan2(center[2], center[0])
        radius = math.sqrt(center[0]**2 + center[2]**2)
        cam_x = center[0] + radius * math.cos(angle)
        cam_y = 0
        cam_z = center[2] + radius * math.sin(angle)
        
        cam_pos = np.array([cam_x, cam_y, cam_z])
        target = center
        up = np.array([0, 1, 0])
        
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, forward)
        up_corrected = up_corrected / np.linalg.norm(up_corrected)
        
        R = np.array([-right, up_corrected, forward]).T
        
        extrinsic = torch.eye(4).float()
        extrinsic[:3, :3] = torch.from_numpy(R)
        extrinsic[:3, 3] = torch.from_numpy(cam_pos)
        extrinsics.append(extrinsic)
    
    return extrinsics


def generate_camera_rotate_trajectory(center: np.ndarray, n_frames: int, 
                                     rotate_max_degree: float = 30, z_progress: bool = True) -> List[torch.Tensor]:
    """Generate in-place camera rotation."""
    extrinsics = []
    for i in range(n_frames):
        extrinsic = torch.eye(4).float()
        angle = math.radians(rotate_max_degree) * math.sin(math.pi * i / n_frames)
        
        R_z = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
        extrinsic[:3, :3] = torch.from_numpy(R_z)
        extrinsics.append(extrinsic)
    
    return extrinsics


def get_trajectory_generator(trajectory_type: str):
    """Get trajectory generator function by type."""
    generators = {
        "static": generate_static_trajectory,
        "forward_backward": generate_forward_backward_trajectory,
        "circle_rotating": generate_circle_rotating_trajectory,
        "surrounding": generate_surrounding_trajectory,
        "camera_rotate": generate_camera_rotate_trajectory,
        # Add other trajectory generators as needed
    }
    return generators.get(trajectory_type)


def render_trajectory(coords_data: torch.Tensor, colors: torch.Tensor, trajectory_type: str,
                     n_frames: int, H: int, W: int, device: torch.device) -> Tuple[List, List, List]:
    """Render video frames along a trajectory."""
    intrinsic = get_intrinsic_matrix(H, W, device)
    
    first_frame_coords = coords_data[0, :, 0].permute(1, 2, 0).reshape(-1, 3)
    center = first_frame_coords.mean(dim=0).cpu().numpy()
    
    # Generate extrinsics based on trajectory type
    if trajectory_type == "surrounding":
        extrinsics = generate_surrounding_trajectory(center, n_frames)
    elif trajectory_type == "camera_rotate":
        extrinsics = generate_camera_rotate_trajectory(center, n_frames, rotate_max_degree=30)
    elif trajectory_type == "forward_backward":
        radius_scaled = 0.4 * abs(center[2])
        extrinsics = generate_forward_backward_trajectory(center, n_frames, radius_scaled)
    elif trajectory_type == "circle_rotating":
        radius_scaled = 0.05 * abs(center[2])
        extrinsics = generate_circle_rotating_trajectory(center, n_frames, radius_scaled)
    elif trajectory_type == "static":
        extrinsics = generate_static_trajectory(n_frames)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    gs_frames, project_frames, project_masks = [], [], []
    
    for frame_idx in range(min(n_frames, coords_data.shape[2])):
        world_points = coords_data[0, :, frame_idx].permute(1, 2, 0).reshape(-1, 3)
        extrinsic = extrinsics[frame_idx].to(device)
        
        try:
            gs_image = render_with_gs(world_points, extrinsic, intrinsic, colors[0], H, W, device)
            gs_frames.append(gs_image)
        except Exception as e:
            print(f"GS rendering failed for frame {frame_idx}: {e}")
            gs_frames.append(np.zeros((H, W, 3), dtype=np.uint8))
        
        try:
            project_image, project_mask = render_with_project(world_points, extrinsic, intrinsic, colors[0], H, W, device)
            project_frames.append(project_image)
            project_masks.append(project_mask)
        except Exception as e:
            print(f"Project rendering failed for frame {frame_idx}: {e}")
            project_frames.append(np.zeros((H, W, 3), dtype=np.uint8))
            project_masks.append(np.ones((H, W), dtype=bool))
    
    return gs_frames, project_frames, project_masks


def save_pointcloud_data(recon_flow: torch.Tensor, colors: torch.Tensor, 
                         video_name: str, output_dir: str, seed: int) -> None:
    """Save 3D point cloud data to disk."""
    pts_dir = os.path.join(output_dir, "pts", f"seed_{seed}")
    os.makedirs(pts_dir, exist_ok=True)
    
    B, C, F, H, W = recon_flow.shape
    for frame_idx in range(F):
        frame_coords = recon_flow[0, :, frame_idx].permute(1, 2, 0).reshape(-1, 3)
        frame_colors = colors[0].reshape(-1, 3)
        pointcloud_data = torch.cat([frame_coords.cpu(), frame_colors.cpu().float()], dim=1)
        
        pts_file = os.path.join(pts_dir, f"{video_name}_frame_{frame_idx:04d}.txt")
        np.savetxt(pts_file, pointcloud_data.numpy())


def load_pointcloud_data(video_name: str, input_dir: str, 
                        device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Load 3D point cloud data from disk."""
    pts_dir = os.path.join(input_dir, "pts")
    
    frame_files = [f for f in os.listdir(pts_dir) 
                  if f.startswith(f"{video_name}_frame_") and f.endswith(".txt")]
    frame_files.sort()
    
    if not frame_files:
        raise ValueError(f"No point cloud files found for video {video_name}")
    
    H, W, F = 368, 512, len(frame_files)
    recon_flow = torch.zeros(1, 3, F, H, W, device=device)
    colors = torch.zeros(1, 3, H, W, device=device)
    
    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(pts_dir, frame_file)
        frame_data = np.loadtxt(frame_path)
        
        coords_tensor = torch.from_numpy(frame_data[:, :3].reshape(H, W, 3)).permute(2, 0, 1).to(device)
        colors_tensor = torch.from_numpy(frame_data[:, 3:6].reshape(H, W, 3)).permute(2, 0, 1).to(device)
        
        recon_flow[0, :, frame_idx] = coords_tensor
        if frame_idx == 0:
            colors[0] = colors_tensor
    
    return recon_flow, colors


def create_control_video_from_image(image_path: str, video_length: int, 
                                   sample_size: List[int], fps: int = 16) -> torch.Tensor:
    """Create control video from a single image."""
    input_video, _, _, _ = get_video_to_video_latent(
        image_path, video_length=video_length, sample_size=sample_size, 
        fps=fps, ref_image=None
    )
    return input_video


def merge_safetensors(model_dir: str) -> Dict:
    """Merge multiple safetensors files into one state dict."""
    state_dict = {}
    for filename in os.listdir(model_dir):
        if filename.endswith(".safetensors"):
            filepath = os.path.join(model_dir, filename)
            partial_state_dict = load_file(filepath)
            state_dict.update(partial_state_dict)
            print(f"Loaded {filename} with {len(partial_state_dict)} keys.")
    return state_dict


def setup_depth_model(device: torch.device) -> UniDepthV2old:
    """Initialize depth estimation model."""
    depth_model = UniDepthV2old.from_pretrained("xxx")
    depth_model.to(device)
    depth_model.eval()
    return depth_model


def setup_decoder_prompt(vae_ckpt_dir: str, device: torch.device, 
                        weight_dtype: torch.dtype) -> VAEDecoderadaptor:
    """Initialize decoder prompt model if checkpoint exists."""
    decoder_path = os.path.join(vae_ckpt_dir, "decoder_prompt", "pytorch_model.bin")
    if not (vae_ckpt_dir and os.path.exists(decoder_path)):
        return None
    
    decoder_prompt = VAEDecoderadaptor().to(device).to(weight_dtype)
    state_dict = torch.load(decoder_path, map_location="cpu")
    decoder_prompt.load_state_dict(state_dict, strict=True)
    decoder_prompt.eval()
    return decoder_prompt


def load_stage1_models(args, logger):
    """Load all models for stage 1 (video generation with depth)."""
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(args.config_path)
    
    # Load transformer
    transformer = WanTransformer4DModel.from_pretrained(
        os.path.join(args.pretrained_model_path, 
                    config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        low_cpu_mem_usage=True,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        torch_dtype=weight_dtype,
    )
    
    # Adapt transformer for depth input if needed
    if args.use_depth:
        _adapt_transformer_for_depth(transformer, weight_dtype)
    
    # Load transformer checkpoint if provided
    if hasattr(args, 'transformer_path') and args.transformer_path is not None:
        state_dict = _load_state_dict(args.transformer_path)
        m, u = transformer.load_state_dict(state_dict, strict=False)
        logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    # Load VAE
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_path, 
                    config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)
    
    if hasattr(args, 'vae_path') and args.vae_path is not None:
        state_dict = _load_state_dict(args.vae_path)
        m, u = vae.load_state_dict(state_dict, strict=False)
        logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    # Load text encoder
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_path, 
                    config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_path, 
                    config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()
    
    # Load image encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.pretrained_model_path, 
                    config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype).eval()
    
    # Setup scheduler
    scheduler = _setup_scheduler(args, config)
    
    # Create pipeline
    pipeline = WanFunControlPipeline(
        transformer=transformer, vae=vae, tokenizer=tokenizer,
        text_encoder=text_encoder, scheduler=scheduler, clip_image_encoder=clip_image_encoder
    )
    
    # Apply optional optimizations
    if hasattr(args, 'compile_dit') and args.compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    
    # Setup memory optimization
    _setup_pipeline_memory_optimization(pipeline, transformer, args, device, weight_dtype)
    
    # Apply optional techniques
    if hasattr(args, 'enable_teacache') and args.enable_teacache:
        _enable_teacache(pipeline, args)
    
    if hasattr(args, 'cfg_skip_ratio') and args.cfg_skip_ratio > 0:
        pipeline.transformer.enable_cfg_skip(args.cfg_skip_ratio, args.num_inference_steps)
    
    if args.lora_path is not None:
        pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device)
    
    # Load auxiliary models
    depth_model = setup_depth_model(device)
    decoder_prompt = setup_decoder_prompt(args.vae_ckpt_dir, device, weight_dtype)
    
    return pipeline, depth_model, decoder_prompt


def _adapt_transformer_for_depth(transformer, weight_dtype):
    """Adapt transformer's patch embedding to accept depth input (64 channels instead of 48)."""
    old_conv = transformer.patch_embedding
    old_w = old_conv.weight.data.clone()
    old_b = old_conv.bias.data.clone() if old_conv.bias is not None else None
    
    if old_w.shape[1] == 48:
        out_c, kernel_size = old_w.shape[0], old_conv.kernel_size
        stride, device, dtype = old_conv.stride, old_w.device, old_w.dtype
        
        from torch import nn
        new_conv = nn.Conv3d(64, out_c, kernel_size, stride, 
                            bias=(old_b is not None), dtype=weight_dtype, device=device)
        
        new_w = torch.zeros((out_c, 64, *old_w.shape[2:]), device=device, dtype=dtype)
        new_w[:, :48] = old_w
        new_w[:, 48:].normal_(0, old_w.std().item())
        
        new_conv.weight.data.copy_(new_w)
        if old_b is not None:
            new_conv.bias.data.copy_(old_b)
        
        transformer.patch_embedding = new_conv


def _load_state_dict(path: str) -> Dict:
    """Load state dict from various formats."""
    if os.path.isdir(path):
        return merge_safetensors(path)
    elif path.endswith("safetensors"):
        return load_file(path)
    else:
        return torch.load(path, map_location="cpu")


def _setup_scheduler(args, config) -> object:
    """Setup scheduler based on configuration."""
    scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }
    sampler_name = getattr(args, 'sampler_name', 'Flow')
    Choosen_Scheduler = scheduler_dict[sampler_name]
    
    if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        config['scheduler_kwargs']['shift'] = 1
    
    return Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )


def _setup_pipeline_memory_optimization(pipeline, transformer, args, device, weight_dtype):
    """Setup memory optimization for pipeline."""
    gpu_memory_mode = getattr(args, 'gpu_memory_mode', 'model_full_load')
    
    if gpu_memory_mode == "sequential_cpu_offload":
        from MoRe4D.utils.fp8_optimization import replace_parameters_by_name
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        from MoRe4D.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif gpu_memory_mode == "model_full_load_and_qfloat8":
        from MoRe4D.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)


def _enable_teacache(pipeline, args):
    """Enable TeaCache optimization if coefficients are available."""
    coefficients = get_teacache_coefficients(args.pretrained_model_path)
    if coefficients is not None:
        teacache_threshold = getattr(args, 'teacache_threshold', 0.10)
        num_skip_start_steps = getattr(args, 'num_skip_start_steps', 5)
        teacache_offload = getattr(args, 'teacache_offload', False)
        pipeline.transformer.enable_teacache(
            coefficients, args.num_inference_steps, teacache_threshold,
            num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )


def load_stage2_models(args, logger):
    """Load all models for stage 2 (video inpainting/completion)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16
    
    config = OmegaConf.load(args.stage2_config_path)
    model_name = args.stage2_model_path
    
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=weight_dtype,
    )
    
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)
    
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=weight_dtype,
    ).eval()
    
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype).eval()
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, 
                       OmegaConf.to_container(config['scheduler_kwargs']))
    )
    
    pipeline = WanFunInpaintPipeline(
        vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
        transformer=transformer, scheduler=scheduler, clip_image_encoder=clip_image_encoder,
    )
    
    # Setup memory optimization
    if args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "sequential_cpu_offload":
        from MoRe4D.utils.fp8_optimization import replace_parameters_by_name
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    else:
        pipeline.to(device=device)
    
    if args.stage2_lora_path is not None:
        pipeline = merge_lora(pipeline, args.stage2_lora_path, args.stage2_lora_weight, device=device)
    
    return pipeline


def process_stage1_sample(args, sample, stage1_pipeline, depth_model, decoder_prompt, 
                         device, weight_dtype, H, W):
    """Process a single sample in stage 1."""
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompt = sample["prompt"]
    image = sample["image"].to(device).to(weight_dtype)
    video_path = sample["video_path"]
    video_name = f"{video_path.parent.name}_{video_path.stem}"
    
    # Calculate video length
    video_length = args.video_num_frames
    calc_video_length = int((video_length - 1) // stage1_pipeline.vae.config.temporal_compression_ratio * 
                           stage1_pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    
    if hasattr(args, 'enable_riflex') and args.enable_riflex:
        latent_frames = (calc_video_length - 1) // stage1_pipeline.vae.config.temporal_compression_ratio + 1
        stage1_pipeline.transformer.enable_riflex(k=args.riflex_k, L_test=latent_frames)
    
    # Prepare input image
    input_image = (image + 1.0) / 2.0
    input_image = input_image.squeeze(0)
    
    temp_image_path = os.path.join(args.output_dir, f"temp_{0}.png")
    temp_image = (input_image.float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(temp_image).save(temp_image_path)
    
    # Get latents
    ref_image = get_image_latent(temp_image_path, sample_size=[H, W])
    control_video = create_control_video_from_image(temp_image_path, calc_video_length, [H, W], args.fps)
    control_video = control_video.repeat(1, 1, calc_video_length, 1, 1)
    clip_image = Image.new("RGB", (W, H), (127, 127, 127))
    
    os.remove(temp_image_path)
    
    # Get depth and back-project to 3D
    depth_pred = depth_model.infer(image.to(torch.float32))["depth"].to(device)
    first_frame_coords = back_project_coords(depth_pred.squeeze(), H, W, device)
    first_frame_coords = first_frame_coords.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    
    # Prepare depth input
    depth_pixel_values = first_frame_coords[:, 2, :, :].unsqueeze(1).repeat(1, 3, 1, 1, 1)
    depth_pixel_values = torch.clamp(depth_pixel_values, min=0.0, max=10000.0)
    depth_pixel_values[torch.isinf(depth_pixel_values) | torch.isnan(depth_pixel_values) | (depth_pixel_values < 1e-5)] = 1
    depth_min, depth_max = depth_pixel_values.min(), depth_pixel_values.max()
    depth_pixel_values = 2 * (depth_pixel_values - depth_min) / (depth_max - depth_min + 1e-8) - 1
    
    # Generate video
    with torch.no_grad():
        sample_result = stage1_pipeline(
            prompt,
            num_frames=calc_video_length,
            negative_prompt=getattr(args, 'negative_prompt', ""),
            height=H, width=W, generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            control_video=control_video,
            control_camera_video=None,
            ref_image=ref_image,
            start_image=None,
            clip_image=clip_image,
            shift=getattr(args, 'shift', 3.0),
            output_type="no_normalize",
            depth_image=depth_pixel_values,
        ).videos
        
        decoder_prompt = decoder_prompt.to(device)
        recon_video = decoder_prompt(sample_result.to(torch.bfloat16).to(device)).float()
        recon_video_vis = (recon_video.squeeze(0) + 1) / 2
        recon_video_vis = recon_video_vis.permute(1, 2, 3, 0).cpu().numpy()
        
        recon_dir = os.path.join(args.output_dir, "recon")
        os.makedirs(recon_dir, exist_ok=True)
        imageio.mimwrite(os.path.join(recon_dir, f"{video_name}_recon.mp4"), recon_video_vis, fps=8)
    
    # Recover 3D coordinates from flow
    if args.normalize_track_z:
        recon_flow = recon_video.permute((0, 2, 1, 3, 4))[0].float().cpu()
        first_frame_expanded = first_frame_coords[0, :, 0].unsqueeze(0).float()
        recon_flow = (recon_flow.unsqueeze(0).permute((0, 2, 1, 3, 4)) + first_frame_expanded)
    else:
        recon_flow, _ = inverse_flow_norm_transform_no_diff(recon_video, first_frame_coords)
    
    # Prepare colors and save point cloud
    color = (image + 1) / 2
    color = color.reshape(color.shape[0], 3, -1).permute(0, 2, 1)
    colors = (color * 255).clamp(0, 255).to(torch.uint8)
    
    coords_data = torch.cat([first_frame_coords, recon_flow[:, :, 1:]], dim=2)
    save_pointcloud_data(coords_data, colors, video_name, args.output_dir, args.seed)
    
    return coords_data, colors


def process_stage1_all_samples(args, dataset, stage1_pipeline, depth_model, 
                              decoder_prompt, logger, only_render=False):
    """Process all samples in stage 1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16
    H, W = args.video_height, args.video_width
    
    stage1_dir = os.path.join(args.output_dir, "stage1_render_results", f"seed_{args.seed}")
    os.makedirs(stage1_dir, exist_ok=True)
    
    for traj_type in TRAJECTORY_TYPES:
        for render_type in ["gs", "project"]:
            os.makedirs(os.path.join(stage1_dir, f"{traj_type}_{render_type}"), exist_ok=True)
        os.makedirs(os.path.join(stage1_dir, f"{traj_type}_masks"), exist_ok=True)
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Stage1 Processing"):
            try:
                sample = dataset[i]
                video_path = sample["video_path"]
                video_name = f"{video_path.parent.name}_{video_path.stem}"
                
                if not only_render:
                    coords_data, colors = process_stage1_sample(
                        args, sample, stage1_pipeline, depth_model, decoder_prompt, device, weight_dtype, H, W
                    )
                else:
                    coords_data, colors = load_pointcloud_data(video_name, args.output_dir)
                
                # Render all trajectories
                for traj_type in TRAJECTORY_TYPES:
                    try:
                        gs_frames, project_frames, project_masks = render_trajectory(
                            coords_data, colors, traj_type, coords_data.shape[2], H, W, device
                        )
                        
                        imageio.mimwrite(
                            os.path.join(stage1_dir, f"{traj_type}_gs", f"{video_name}_render.mp4"),
                            gs_frames, fps=8
                        )
                        imageio.mimwrite(
                            os.path.join(stage1_dir, f"{traj_type}_project", f"{video_name}_render.mp4"),
                            project_frames, fps=8
                        )
                        mask_frames_uint8 = [(mask * 255).astype(np.uint8) for mask in project_masks]
                        imageio.mimwrite(
                            os.path.join(stage1_dir, f"{traj_type}_masks", f"{video_name}_mask.mp4"),
                            mask_frames_uint8, fps=8
                        )
                    except Exception as e:
                        print(f"Trajectory {traj_type} rendering failed: {e}")
                        continue
                        
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def process_stage2_all_samples(args, dataset, stage2_pipeline, logger):
    """Process all samples in stage 2 (inpainting)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = args.video_height, args.video_width
    
    stage1_dir = os.path.join(args.output_dir, "stage1_render_results", f"seed_{args.seed}")
    stage2_dir = os.path.join(args.output_dir, "stage2_completion_results", f"seed_{args.seed}")
    os.makedirs(stage2_dir, exist_ok=True)
    
    for traj_type in TRAJECTORY_TYPES:
        os.makedirs(os.path.join(stage2_dir, traj_type), exist_ok=True)
    
    for i in tqdm(range(len(dataset)), desc="Stage2 Processing"):
        try:
            sample = dataset[i]
            prompt = sample["prompt"]
            video_path = sample["video_path"]
            video_name = f"{video_path.parent.name}_{video_path.stem}"
            
            for traj_type in TRAJECTORY_TYPES:
                generator = torch.Generator(device=device).manual_seed(args.seed + 1)
                output_video_path = os.path.join(stage2_dir, traj_type, f"{video_name}.mp4")
                
                if os.path.exists(output_video_path):
                    continue
                
                gs_video_path = os.path.join(stage1_dir, f"{traj_type}_gs", f"{video_name}_render.mp4")
                mask_video_path = os.path.join(stage1_dir, f"{traj_type}_masks", f"{video_name}_mask.mp4")
                
                if not (os.path.exists(gs_video_path) and os.path.exists(mask_video_path)):
                    continue
                
                with torch.no_grad():
                    video_length = args.video_num_frames
                    calc_video_length = int((video_length - 1) // stage2_pipeline.vae.config.temporal_compression_ratio *
                                           stage2_pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
                    
                    input_video, input_video_mask, _, _ = get_video_to_video_latent(
                        gs_video_path, video_length=calc_video_length, sample_size=[H, W], fps=8,
                        validation_video_mask=mask_video_path, ref_image=None, video_mask=True
                    )
                    
                    sample = stage2_pipeline(
                        prompt,
                        num_frames=calc_video_length,
                        negative_prompt=args.stage2_negative_prompt,
                        height=H, width=W, generator=generator,
                        guidance_scale=args.stage2_guidance_scale,
                        num_inference_steps=args.stage2_num_inference_steps,
                        video=input_video,
                        mask_video=input_video_mask,
                        shift=3,
                    ).videos
                    
                    save_videos_grid(sample, output_video_path, fps=8)
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue


def cleanup_stage1_models(stage1_pipeline, depth_model, decoder_prompt, logger):
    """Clean up stage 1 models from GPU memory."""
    if stage1_pipeline is not None:
        stage1_pipeline.to("cpu")
        del stage1_pipeline
    
    if depth_model is not None:
        depth_model.to("cpu")
        del depth_model
    
    if decoder_prompt is not None:
        decoder_prompt.to("cpu")
        del decoder_prompt
    
    gc.collect()
    torch.cuda.empty_cache()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Two-stage video generation and rendering pipeline")
    
    # Model paths
    parser.add_argument("--pretrained_model_path", type=str, default="models/Wan2.1-Fun-V1.1-14B-Control")
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--vae_ckpt_dir", type=str, default="xxx")
    
    # Stage 2 models
    parser.add_argument("--stage2_model_path", type=str, default="models/Wan2.1-Fun-V1.1-14B-InP")
    parser.add_argument("--stage2_config_path", type=str, default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--stage2_lora_path", type=str, default="xxx.safetensors")
    parser.add_argument("--stage2_lora_weight", type=float, default=0.55)
    
    # Feature flags
    parser.add_argument("--use_depth", default=True)
    parser.add_argument("--only_render", action='store_true')
    parser.add_argument("--normalize_track_z", action='store_true')
    
    # Scheduler and sampling
    parser.add_argument("--sampler_name", type=str, default="Flow", 
                       choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    parser.add_argument("--shift", type=float, default=3.0)
    
    # Optimizations
    parser.add_argument("--enable_teacache", action='store_true', default=False)
    parser.add_argument("--teacache_threshold", type=float, default=0.10)
    parser.add_argument("--num_skip_start_steps", type=int, default=5)
    parser.add_argument("--teacache_offload", action='store_true')
    parser.add_argument("--cfg_skip_ratio", type=float, default=0.0)
    parser.add_argument("--compile_dit", action='store_true')
    parser.add_argument("--enable_riflex", action='store_true')
    parser.add_argument("--riflex_k", type=int, default=6)
    parser.add_argument("--gpu_memory_mode", type=str, default="model_full_load",
                       choices=["model_full_load", "model_full_load_and_qfloat8", 
                               "model_cpu_offload", "model_cpu_offload_and_qfloat8", 
                               "sequential_cpu_offload"])
    
    # Prompts and inference
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--stage2_negative_prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--stage2_guidance_scale", type=float, default=6.0)
    parser.add_argument("--stage2_num_inference_steps", type=int, default=50)
    parser.add_argument("--lora_weight", type=float, default=0.55)
    
    # Data paths
    parser.add_argument("--data_path", type=str, default="/xxx")
    parser.add_argument("--prompt_file_name", type=str, default="prompts_demo.txt")
    parser.add_argument("--video_file_name", type=str, default="videos_demo.txt")
    parser.add_argument("--output_dir", type=str, default="output_dir/infer")
    
    # Video parameters
    parser.add_argument("--video_height", type=int, default=368)
    parser.add_argument("--video_width", type=int, default=512)
    parser.add_argument("--video_num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    
    # Training parameters
    parser.add_argument("--mixed_precision", type=str, default="bf16", 
                       choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max_samples", type=int, default=800)
    
    # Pipeline control
    parser.add_argument("--run_stage1", action='store_true', default=False)
    parser.add_argument("--run_stage2_render", action='store_true', default=True)
    parser.add_argument("--run_stage2_complete", action='store_true', default=False)
    
    return parser.parse_args()


def main():
    """Main inference pipeline."""
    args = parse_args()
    logger = get_logger("two_stage_pipeline")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TwoStageDataset(
        data_root=args.data_path,
        caption_column=args.prompt_file_name,
        video_column=args.video_file_name,
        device=device,
        max_num_frames=args.video_num_frames,
        height=args.video_height,
        width=args.video_width,
        max_samples=args.max_samples,
    )
    
    if args.run_stage1:
        logger.info("=" * 50)
        logger.info("Running Stage 1: Video Generation with Depth")
        logger.info("=" * 50)
        
        stage1_pipeline, depth_model, decoder_prompt = load_stage1_models(args, logger)
        process_stage1_all_samples(args, dataset, stage1_pipeline, depth_model, 
                                  decoder_prompt, logger, only_render=args.only_render)
        cleanup_stage1_models(stage1_pipeline, depth_model, decoder_prompt, logger)
        logger.info("Stage 1 Completed!")
    
    if args.run_stage2_complete:
        logger.info("=" * 50)
        logger.info("Running Stage 2: Video Inpainting/Completion")
        logger.info("=" * 50)
        
        stage2_pipeline = load_stage2_models(args, logger)
        process_stage2_all_samples(args, dataset, stage2_pipeline, logger)
        
        if args.stage2_lora_path is not None:
            stage2_pipeline = unmerge_lora(stage2_pipeline, args.stage2_lora_path, args.stage2_lora_weight, device=device)
        
        stage2_pipeline.to("cpu")
        del stage2_pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("Stage 2 Completed!")
    
    logger.info("=" * 50)
    logger.info(f"Pipeline done! Save in {args.output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
