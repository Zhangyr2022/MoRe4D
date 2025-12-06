import csv
import gc
import json
import os
import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from packaging import version as pver
from PIL import Image
from torch.utils.data import BatchSampler, Sampler, Dataset
from ..utils.project_utils import project

VIDEO_READER_TIMEOUT = 20


def get_random_mask(shape: Tuple[int, ...], image_start_only: bool = False) -> torch.Tensor:
    """Generate random masks for inpainting training.
    
    Args:
        shape: Tensor shape (f, c, h, w)
        image_start_only: Whether to only mask from the second frame onwards
        
    Returns:
        Random mask tensor with values 0 or 1
    """
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        mask_type = _get_mask_type(f)
        mask = _apply_mask_strategy(mask, mask_type, f, h, w)
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    
    return mask


def _get_mask_type(num_frames: int) -> int:
    """Select mask strategy based on number of frames."""
    if num_frames != 1:
        return np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
            p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]
        )
    else:
        return np.random.choice([0, 1], p=[0.2, 0.8])


def _apply_mask_strategy(mask: torch.Tensor, mask_type: int, f: int, h: int, w: int) -> torch.Tensor:
    """Apply specific masking strategy."""
    if mask_type == 0:  # Random block mask
        mask = _apply_block_mask(mask, h, w)
    elif mask_type == 1:  # Full mask
        mask[:, :, :, :] = 1
    elif mask_type == 2:  # Temporal suffix mask
        mask_start = np.random.randint(1, 5)
        mask[mask_start:, :, :, :] = 1
    elif mask_type == 3:  # Temporal middle mask
        mask_start = np.random.randint(1, 5)
        mask[mask_start:-mask_start, :, :, :] = 1
    elif mask_type == 4:  # Spatio-temporal block mask
        mask = _apply_spatiotemporal_mask(mask, f, h, w)
    elif mask_type == 5:  # Random noise mask
        mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
    elif mask_type == 6:  # Scattered blocks
        mask = _apply_scattered_blocks(mask, f, h, w)
    elif mask_type == 7:  # Elliptical mask
        mask = _apply_elliptical_mask(mask, h, w)
    elif mask_type == 8:  # Circular mask
        mask = _apply_circular_mask(mask, h, w)
    elif mask_type == 9:  # Random frame mask
        for idx in range(f):
            if np.random.rand() > 0.5:
                mask[idx, :, :, :] = 1
    
    return mask


def _apply_block_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Apply block mask to all frames."""
    center_x = torch.randint(0, w, (1,)).item()
    center_y = torch.randint(0, h, (1,)).item()
    block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()
    block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()

    start_x = max(center_x - block_size_x // 2, 0)
    end_x = min(center_x + block_size_x // 2, w)
    start_y = max(center_y - block_size_y // 2, 0)
    end_y = min(center_y + block_size_y // 2, h)
    
    mask[:, :, start_y:end_y, start_x:end_x] = 1
    return mask


def _apply_spatiotemporal_mask(mask: torch.Tensor, f: int, h: int, w: int) -> torch.Tensor:
    """Apply spatio-temporal block mask."""
    center_x = torch.randint(0, w, (1,)).item()
    center_y = torch.randint(0, h, (1,)).item()
    block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()
    block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()

    start_x = max(center_x - block_size_x // 2, 0)
    end_x = min(center_x + block_size_x // 2, w)
    start_y = max(center_y - block_size_y // 2, 0)
    end_y = min(center_y + block_size_y // 2, h)

    mask_frame_before = np.random.randint(0, f // 2)
    mask_frame_after = np.random.randint(f // 2, f)
    mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
    return mask


def _apply_scattered_blocks(mask: torch.Tensor, f: int, h: int, w: int) -> torch.Tensor:
    """Apply scattered block masks."""
    num_frames_to_mask = random.randint(1, max(f // 2, 1))
    frames_to_mask = random.sample(range(f), num_frames_to_mask)

    for i in frames_to_mask:
        block_height = random.randint(1, h // 4)
        block_width = random.randint(1, w // 4)
        top_left_y = random.randint(0, h - block_height)
        top_left_x = random.randint(0, w - block_width)
        mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
    return mask


def _apply_elliptical_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Apply elliptical mask."""
    center_x = torch.randint(0, w, (1,)).item()
    center_y = torch.randint(0, h, (1,)).item()
    a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()
    b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()

    for i in range(h):
        for j in range(w):
            if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                mask[:, :, i, j] = 1
    return mask


def _apply_circular_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Apply circular mask."""
    center_x = torch.randint(0, w, (1,)).item()
    center_y = torch.randint(0, h, (1,)).item()
    radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
    
    for i in range(h):
        for j in range(w):
            if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                mask[:, :, i, j] = 1
    return mask


class Camera:
    """Camera parameters for 3D projection. Copied from CameraCtrl."""
    
    def __init__(self, entry: List[float]):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def custom_meshgrid(*args):
    """PyTorch version-compatible meshgrid."""
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_relative_pose(cam_params: List[Camera]) -> np.ndarray:
    """Get relative camera poses."""
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    return np.array(ret_poses, dtype=np.float32)


def ray_condition(K: torch.Tensor, c2w: torch.Tensor, H: int, W: int, device: str) -> torch.Tensor:
    """Compute ray conditions for camera control."""
    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5

    fx, fy, cx, cy = K.chunk(4, dim=-1)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)
    
    return plucker


def process_pose_params(cam_params: List[List[float]], width: int = 672, height: int = 384,
                       original_pose_width: int = 1280, original_pose_height: int = 720,
                       device: str = 'cpu') -> torch.Tensor:
    """Process camera parameters to generate Plucker embeddings."""
    cam_objects = [Camera(cam_param) for cam_param in cam_params]

    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height

    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_objects:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_objects:
            cam_param.fy = resized_ori_h * cam_param.fy / height

    intrinsic = np.asarray([[cam_param.fx * width,
                            cam_param.fy * height,
                            cam_param.cx * width,
                            cam_param.cy * height]
                            for cam_param in cam_objects], dtype=np.float32)

    K = torch.as_tensor(intrinsic)[None]
    c2ws = get_relative_pose(cam_objects)
    c2ws = torch.as_tensor(c2ws)[None]
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()
    plucker_embedding = plucker_embedding[None]
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
    
    return plucker_embedding


class ImageVideoSampler(BatchSampler):
    """Batch sampler for grouping images and videos separately."""

    def __init__(self, sampler: Sampler, dataset: Dataset, batch_size: int, drop_last: bool = False):
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of Sampler')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer')
            
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket = {'image': [], 'video': []}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # Yield batch when bucket is full
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    """Context manager for VideoReader with automatic cleanup."""
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


class ViSMDataset(Dataset):
    """Dataset for image and video data with flow information and 3DGS rendering."""

    def __init__(self, ann_path: str, data_root: Optional[str] = None, 
                 video_sample_size: Union[int, Tuple[int, int]] = 512,
                 video_sample_stride: int = 1, video_sample_n_frames: int = 16,
                 image_sample_size: Union[int, Tuple[int, int]] = 512,
                 video_repeat: int = 0, text_drop_ratio: float = 0.1,
                 enable_bucket: bool = True, video_length_drop_start: float = 0.0,
                 video_length_drop_end: float = 1.0, enable_inpaint: bool = False,
                 return_file_name: bool = False, max_num_frames: int = 49,
                 use_3dgs: bool = False):
        
        self.max_num_frames = max_num_frames
        self.use_3dgs = use_3dgs
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        
        # Load dataset
        self.dataset = self._load_dataset(ann_path, video_repeat)
        self.data_root = data_root or "/mnt/pfs/users/ziyi.wang/workspace/CogVideo/data/webvid"
        self.length = len(self.dataset)
        
        # Initialize transforms and parameters
        self._initialize_transforms(video_sample_size, image_sample_size)
        self._initialize_camera_params()
        
        print(f"Data scale: {self.length}")
        print(f"Using 3DGS rendered results: {self.use_3dgs}")

    def _load_dataset(self, ann_path: str, video_repeat: int) -> List[Dict]:
        """Load dataset from annotation file."""
        print(f"Loading annotations from {ann_path}...")
        
        if ann_path.endswith('prompt_clean_normalized_less.txt'):
            dataset = self._load_text_format(ann_path)
        elif ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
        else:
            raise ValueError(f"Unsupported annotation format: {ann_path}")
        
        return self._apply_video_repeat(dataset, video_repeat)

    def _load_text_format(self, ann_path: str) -> List[Dict]:
        """Load text format annotations."""
        with open(ann_path, 'r') as f:
            prompts = [p.strip() for p in f.readlines()]
        
        video_path = os.path.join(os.path.dirname(ann_path), 'videos_clean_normalized_less.txt')
        with open(video_path, 'r') as f:
            videos = [v.strip() for v in f.readlines()]
        
        return [{'file_path': videos[i], 'text': prompts[i], 'type': 'video'} 
                for i in range(len(prompts))]

    def _apply_video_repeat(self, dataset: List[Dict], video_repeat: int) -> List[Dict]:
        """Apply video repetition for balancing."""
        if video_repeat <= 0:
            return dataset
            
        result = [data for data in dataset if data.get('type', 'image') != 'video']
        video_data = [data for data in dataset if data.get('type', 'image') == 'video']
        
        for _ in range(video_repeat):
            result.extend(video_data)
            
        return result

    def _initialize_transforms(self, video_sample_size: Union[int, Tuple[int, int]], 
                              image_sample_size: Union[int, Tuple[int, int]]):
        """Initialize image and video transforms."""
        self.H, self.W = 384, 512
        self.H_ori, self.W_ori = 540, 960
        
        self.video_sample_size = (video_sample_size, video_sample_size) if isinstance(video_sample_size, int) else video_sample_size
        self.image_sample_size = (image_sample_size, image_sample_size) if isinstance(image_sample_size, int) else image_sample_size
        
        self.video_transforms = transforms.Compose([
            transforms.Resize((self.H, self.W)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.image_transforms = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _initialize_camera_params(self):
        """Initialize camera intrinsic and extrinsic parameters."""
        if self.W_ori / self.W > self.H_ori / self.H:
            self.fx = 1
            self.fy = self.W_ori / self.H_ori / (self.W / self.H)
        else:
            self.fy = 1
            self.fx = self.H_ori / self.W_ori / (self.H / self.W)
            
        self.intrinsic = torch.Tensor([
            [self.fx, 0, 0.5],
            [0, self.fy, 0.5],
            [0, 0, 1]
        ])
        
        self.extrinsic = torch.Tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def project_point_cloud(self, coords: torch.Tensor, colors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project point cloud to image space."""
        from torch_scatter import scatter
        
        predicted_2D, depth_2D = project(coords, self.extrinsic, self.intrinsic)
        mask = ((predicted_2D[..., 0] >= 0) & (predicted_2D[..., 0] <= 1) &
                (predicted_2D[..., 1] >= 0) & (predicted_2D[..., 1] <= 1) &
                (depth_2D >= 0))
               
        if mask.sum() == 0:
            color_image = torch.zeros((self.H, self.W, 3), dtype=colors.dtype, device=colors.device)
            mask_image = torch.ones_like(color_image)
            return color_image, mask_image

        color_pc = colors[mask, :]
        depth_2D = depth_2D[mask]
        idx_pc = predicted_2D[mask, :]
        idx_xy = ((idx_pc[:, 0] * self.W).floor().clamp(0, self.W - 1) * self.H + 
                 (idx_pc[:, 1] * self.H).floor().clamp(0, self.H - 1))
        
        # Handle depth conflicts
        unique_indices, inverse_indices = torch.unique(idx_xy, return_inverse=True)
        min_depth = torch.ones_like(unique_indices, dtype=depth_2D.dtype) * depth_2D.max()
        min_depth.index_reduce_(0, inverse_indices, depth_2D, 'amin')
        mask_depth = (depth_2D == min_depth[inverse_indices])
        
        color_pc = color_pc[mask_depth, :]
        idx_xy = idx_xy[mask_depth]
        
        color_image = scatter(color_pc, idx_xy.long(), dim=0, reduce="mean")
        if len(color_image) < self.H * self.W:
            padding = torch.zeros((self.H * self.W - len(color_image), 3), device=color_image.device)
            color_image = torch.cat([color_image, padding], dim=0)
            
        color_image = color_image.reshape(self.W, self.H, 3).transpose(0, 1)
        mask_image = (color_image.sum(dim=2) == 0).float().unsqueeze(2).expand_as(color_image)
        
        return color_image, mask_image

    def load_flow_data(self, video_path: str) -> Optional[Dict]:
        """Load point cloud flow data."""
        import pickle
        
        flow_path = video_path.replace("videos", "dt3d_render").replace(".mp4", "_dt3d_pred.pkl")
        if not os.path.exists(flow_path) and "webvid_tmp_13" in flow_path:
            flow_path = flow_path.replace("webvid_tmp_13", "webvid_tmp_13_vepfs2")

        try:
            with open(flow_path, "rb") as f:
                data = pickle.load(f)
                
            if "coords" not in data or "colors" not in data:
                print(f"Invalid flow data in {flow_path}")
                return None
                
            return data
        except Exception as e:
            print(f"Failed to load flow from {flow_path}: {e}")
            return None

    def load_3dgs_render(self, video_path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load pre-rendered 3DGS results."""
        render_path = video_path.replace("videos", "dt3d_render").replace(".mp4", "_dt3d_render.mp4")
        mask_path = video_path.replace("videos", "dt3d_render").replace(".mp4", "_mask_render.mp4")
        
        if "tmp_13" in render_path and not os.path.exists(render_path):
            render_path = render_path.replace("webvid_tmp_13", "webvid_tmp_13_vepfs2")
            mask_path = mask_path.replace("webvid_tmp_13", "webvid_tmp_13_vepfs2")
            
        try:
            render_frames = self._load_video_frames(render_path)
            mask_frames = self._load_video_frames(mask_path)
            
            # Process masks to binary format
            processed_masks = []
            for mask in mask_frames:
                mask_binary = (mask.sum(dim=2) > 0).float().unsqueeze(2).expand_as(mask)
                processed_masks.append(mask_binary)
                
            return torch.stack(render_frames), torch.stack(processed_masks)
            
        except Exception as e:
            print(f"Failed to load 3DGS render for {video_path}: {e}")
            return None, None

    def _load_video_frames(self, video_path: str) -> List[torch.Tensor]:
        """Load video frames with sampling and padding."""
        with VideoReader_contextmanager(video_path, width=self.W, height=self.H, num_threads=2) as reader:
            if len(reader) < 1:
                raise ValueError(f"No frames in video: {video_path}")
                
            # Sample frames
            if len(reader) > self.max_num_frames:
                indices = list(range(0, self.max_num_frames * 2, 2))[:self.max_num_frames]
            else:
                indices = list(range(len(reader)))
            
            frames = reader.get_batch(indices).asnumpy()
            frames = [torch.from_numpy(frame).float() for frame in frames]
            
            # Pad if necessary
            if len(frames) < self.max_num_frames:
                last_frame = frames[-1]
                frames.extend([last_frame.clone() for _ in range(self.max_num_frames - len(frames))])
                
            return frames[:self.max_num_frames]

    def get_batch(self, idx: int) -> Tuple:
        """Get a batch of data."""
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'image') == 'video':
            return self._process_video_sample(data_info)
        else:
            return self._process_image_sample(data_info)

    def _process_video_sample(self, data_info: Dict) -> Tuple:
        """Process video sample."""
        video_id, text = data_info['file_path'], data_info['text']
        video_dir = os.path.join(self.data_root, video_id) if self.data_root else video_id
        
        # Load 3DGS or flow data
        if self.use_3dgs:
            projected_images_tensor, masks_tensor = self.load_3dgs_render(video_dir)
            if projected_images_tensor is None or masks_tensor is None:
                raise ValueError(f"Failed to load 3DGS render for {video_dir}")
        else:
            projected_images_tensor, masks_tensor = self._process_flow_data(video_dir)
        
        # Load original video frames
        frames_tensor = self._load_original_video(video_dir)
        
        # Apply text dropout
        if random.random() < self.text_drop_ratio:
            text = ''
            
        return frames_tensor, text, 'video', video_dir, masks_tensor, projected_images_tensor

    def _process_flow_data(self, video_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process flow data for point cloud projection."""
        flow_data = self.load_flow_data(video_dir)
        if flow_data is None:
            raise ValueError(f"Failed to load flow data for {video_dir}")
            
        coords = flow_data["coords"].float()
        colors = flow_data["colors"]
        
        # Handle dimensions
        if len(coords.shape) == 3:
            coords = coords.unsqueeze(0)
        if len(colors.shape) == 2:
            colors = colors.unsqueeze(0)
            
        n_frames = coords.shape[1]
        
        # Project each frame
        masks, projected_images = [], []
        for i in range(min(n_frames, self.max_num_frames)):
            projected_image, mask = self.project_point_cloud(coords[0][i], colors[0])
            masks.append(mask)
            projected_images.append(projected_image)
        
        # Pad if necessary
        if len(masks) < self.max_num_frames:
            last_mask = masks[-1] if masks else torch.zeros((self.H, self.W, 3))
            last_image = projected_images[-1] if projected_images else torch.zeros((self.H, self.W, 3))
            
            for _ in range(self.max_num_frames - len(masks)):
                masks.append(last_mask.clone())
                projected_images.append(last_image.clone())
        
        return torch.stack(projected_images[:self.max_num_frames]), torch.stack(masks[:self.max_num_frames])

    def _load_original_video(self, video_dir: str) -> torch.Tensor:
        """Load original video frames."""
        try:
            frames = self._load_video_frames(video_dir)
            frames_tensor = torch.stack(frames)
            
            if not self.enable_bucket:
                frames_tensor = frames_tensor / 255.0
                frames_tensor = frames_tensor.permute(0, 3, 1, 2).contiguous()
                frames_tensor = self.video_transforms(frames_tensor)
                
            return frames_tensor
            
        except Exception as e:
            print(f"Failed to extract frames from video: {e}")
            raise ValueError(f"Failed to process video: {video_dir}")

    def _process_image_sample(self, data_info: Dict) -> Tuple:
        """Process image sample."""
        image_path, text = data_info['file_path'], data_info['text']
        if self.data_root is not None:
            image_path = os.path.join(self.data_root, image_path)
            
        image = Image.open(image_path).convert('RGB')
        
        if not self.enable_bucket:
            image = self.image_transforms(image).unsqueeze(0)
            mask = torch.zeros_like(image)
        else:
            image = np.expand_dims(np.array(image), 0)
            mask = np.zeros_like(image)
            
        if random.random() < self.text_drop_ratio:
            text = ''
            
        return image, text, 'image', image_path, mask, None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item with error handling."""
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        
        while True:
            try:
                pixel_values, name, data_type, file_path, mask, projected_images_tensor = self.get_batch(idx)
                
                sample = {
                    "pixel_values": pixel_values,
                    "text": name,
                    "data_type": data_type,
                    "idx": idx,
                    "mask": mask,
                    "projected_images": projected_images_tensor
                }
                
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                # Add inpainting masks
                if self.enable_inpaint and not self.enable_bucket:
                    if "mask" not in sample:
                        mask = get_random_mask(pixel_values.size())
                    else:
                        mask = sample["mask"]
                        
                    mask_pixel_values = projected_images_tensor * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
                    sample["mask_pixel_values"] = mask_pixel_values

                    clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
                    clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                    sample["clip_pixel_values"] = clip_pixel_values

                return sample
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing sample {idx}: {e}")
                idx = random.randint(0, self.length - 1)


class STraGDataset(Dataset):
    """Dataset for stage 1 training with flow control."""

    def __init__(self, ann_path: str, data_root: Optional[str] = None,
                 height: int = 384, width: int = 512, max_num_frames: int = 49,
                 device: torch.device = torch.device("cpu"), normalize_flow: bool = True,
                 text_drop_ratio: float = 0.1, skip_large_depth: bool = True,
                 max_sample_dataset: Optional[int] = None, normalize_track_z: bool = False):
        
        self.skip_large_depth = skip_large_depth
        self.normalize_flow = normalize_flow
        self.normalize_track_z = normalize_track_z
        self.max_sample_dataset = max_sample_dataset
        self.device = device
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width
        self.text_drop_ratio = text_drop_ratio
        self.H, self.W = 384, 512
        
        # Load dataset
        self.dataset = self._load_dataset(ann_path)
        self.data_root = data_root or "/mnt/pfs/users/ziyi.wang/workspace/CogVideo/data/webvid"
        self.length = len(self.dataset)
        
        # Initialize transforms
        self.video_transforms = transforms.Compose([transforms.Resize((self.H, self.W))])
        self.image_transforms = transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        
        self._print_config()

    def _load_dataset(self, ann_path: str) -> List[Dict]:
        """Load dataset from annotation file."""
        print(f"Loading annotations from {ann_path}...")
        
        if ann_path.endswith('.txt'):
            with open(ann_path, 'r') as f:
                prompts = [p.strip() for p in f.readlines()]
            
            if self.max_sample_dataset is not None:
                prompts = prompts[:self.max_sample_dataset]
            
            video_path = os.path.join(os.path.dirname(ann_path), 'videos_clean_normalized.txt')
            with open(video_path, 'r') as f:
                videos = [v.strip().replace(
                    "/mnt/pfs/users/ziyi.wang/workspace/CogVideo/data/webvid/",
                    "/mnt/data/yanran.zhang/data/4D.yanran/web_dataset/"
                ) for v in f.readlines()]
            
            return [{'file_path': videos[i], 'control_file_path': videos[i], 'text': prompts[i], 'type': 'video'} 
                   for i in range(len(prompts))]
                   
        elif ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                return list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            return json.load(open(ann_path))
        else:
            raise ValueError(f"Unsupported file format: {ann_path}")

    def _print_config(self):
        """Print dataset configuration."""
        print(f"Dataset size: {self.length}")
        print(f"Device: {self.device}")
        print(f"Max frames: {self.max_num_frames}")
        print(f"Resolution: {self.height}x{self.width}")
        print(f"Normalize flow: {self.normalize_flow}")
        print(f"Text drop ratio: {self.text_drop_ratio}")

    def load_flow_data(self, video_path: str) -> Optional[Dict]:
        """Load point cloud flow data."""
        import pickle
        
        flow_path = video_path.replace("videos", "dt3d_render").replace(".mp4", "_dt3d_pred.pkl")
        if not os.path.exists(flow_path) and "webvid_tmp_13" in flow_path:
            flow_path = flow_path.replace("webvid_tmp_13", "webvid_tmp_13_vepfs2")
        
        try:
            with open(flow_path, "rb") as f:
                data = pickle.load(f)
                
            if "coords" not in data or "colors" not in data:
                print(f"Invalid flow data in {flow_path}")
                return None
                
            return data
        except Exception as e:
            print(f"Failed to load flow from {flow_path}: {e}")
            return None

    def normalize_flow_data(self, flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize flow data based on first frame."""
        for b in range(flow.size(0)):
            frame0 = flow[b, :, 0, :, :]  # [3, H, W]
            max_vals = frame0.view(3, -1).max(dim=1)[0]  # [3]
            min_vals = frame0.view(3, -1).min(dim=1)[0]  # [3]
            diff = (max_vals - min_vals).max().repeat(3)  # [3]
            
            diff[diff == 0] = 1.0  # Avoid division by zero
            flow[b] = flow[b] / diff.view(3, 1, 1, 1)
            
        return flow, diff

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item with comprehensive error handling."""
        data_info = self.dataset[idx % len(self.dataset)]
        
        while True:
            try:
                sample = self._process_sample(data_info, idx)
                if len(sample) > 0:
                    return sample
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing sample {idx}: {e}")
                idx = random.randint(0, len(self.dataset) - 1)
                data_info = self.dataset[idx % len(self.dataset)]

    def _process_sample(self, data_info: Dict, idx: int) -> Dict:
        """Process a single sample."""
        data_type = data_info.get('type', 'image')
        
        if data_type == 'video':
            return self._process_video_sample(data_info, idx)
        else:
            return self._process_image_sample(data_info, idx)

    def _process_video_sample(self, data_info: Dict, idx: int) -> Dict:
        """Process video sample with flow data."""
        prompt = data_info['text']
        video_path = data_info['file_path']
        
        if self.data_root is not None:
            video_path = os.path.join(self.data_root, video_path)
            
        # Apply text dropout
        if random.random() < self.text_drop_ratio:
            prompt = ''
            
        # Load and process flow data
        flow_data = self.load_flow_data(video_path)
        if flow_data is None:
            raise ValueError(f"Flow data not found: {video_path}")
        
        coords = flow_data["coords"].float()
        colors = flow_data["colors"].float()
        
        # Process dimensions
        if len(coords.shape) == 3:
            coords = coords.unsqueeze(0)
        if len(colors.shape) == 2:
            colors = colors.unsqueeze(0)
        
        # Reshape flow data
        B, T = coords.shape[:2]
        flow = coords.reshape(B, T, self.height, self.width, 3).permute(0, 4, 1, 2, 3)  # [B, 3, T, H, W]
        image = colors.reshape(B, self.height, self.width, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
        
        # Apply normalization and filtering
        flow = self._process_flow_normalization(flow, video_path, idx)
        
        # Limit frame count
        flow = self._limit_and_pad_frames(flow)
        
        # Process image
        image = self.image_transforms(image)
        
        # Load RGB video frames
        rgb_frames = self._load_rgb_frames(video_path)
        
        return {
            "pixel_values": flow.squeeze(0).permute((1, 2, 3, 0)),  # [F, H, W, 3]
            "control_pixel_values": image.repeat(T, 1, 1, 1),  # [T, 3, H, W]
            "text": prompt,
            "data_type": "video",
            "idx": idx,
            "video_metadata": {
                "num_frames": flow.shape[2],
                "height": self.height,
                "width": self.width,
                "path": video_path
            },
            "flow_first_frame": coords.reshape(B, T, self.height, self.width, 3)[0, 0],  # [H, W, 3]
            "rgb_pixel_values": rgb_frames
        }

    def _process_flow_normalization(self, flow: torch.Tensor, video_path: str, idx: int) -> torch.Tensor:
        """Process flow normalization and filtering."""
        if not self.normalize_track_z:
            if self.normalize_flow:
                flow, diff = self.normalize_flow_data(flow)
                if self.skip_large_depth and diff.max() > 500.0:
                    raise ValueError(f"Large depth sample: {video_path} with diff {diff}")
            flow = flow - flow[:, :, :1, :, :]  # Relative to first frame
        else:
            if self.skip_large_depth and flow.max() > 500.0:
                raise ValueError(f"Large depth sample: {video_path}")
                
            flow_delta = flow - flow[:, :, :1, :, :]  # B, 3, T, H, W
            targets = self._normalize_with_depth(flow, flow_delta)
            flow = targets
            
        return flow

    def _normalize_with_depth(self, flow: torch.Tensor, flow_delta: torch.Tensor) -> torch.Tensor:
        """Normalize flow with depth information."""
        targets = []
        
        for b in range(flow.size(0)):
            frame0 = flow[b, :, 0, :, :].clone()  # [3, H, W]
            
            # Handle invalid depth values
            frame0[2, :, :][torch.isnan(frame0[2, :, :])] = 1.0
            frame0[2, :, :][frame0[2, :, :] == 0] = 1
            frame0[2, :, :][torch.isinf(frame0[2, :, :])] = 1.0
            
            # Calculate normalization factors
            H_ori, W_ori = [720, 960]
            H, W = [368, 512]
            
            if W_ori / W > H_ori / H:
                fx = 1
                fy = W_ori / H_ori / (W / H)
            else:
                fy = 1
                fx = H_ori / W_ori / (H / W)
                
            current_x_norm = frame0[2, :, :] / fx
            current_y_norm = frame0[2, :, :] / fy
            
            temp = flow_delta[b, :, :, :, :].clone()
            temp[0:1, :, :, :] = temp[0:1, :, :, :] / current_x_norm
            temp[1:2, :, :, :] = temp[1:2, :, :, :] / current_y_norm
            temp[2:3, :, :, :] = temp[2:3, :, :, :] / frame0[2:3, :, :]
            targets.append(temp)
            
        return torch.stack(targets, dim=0)  # B, 3, T, H, W

    def _limit_and_pad_frames(self, flow: torch.Tensor) -> torch.Tensor:
        """Limit frame count and pad if necessary."""
        if flow.shape[2] > self.max_num_frames:
            flow = flow[:, :, :self.max_num_frames, :, :]
        elif flow.shape[2] < self.max_num_frames:
            last_frame = flow[:, :, -1:, :, :]
            num_repeats = self.max_num_frames - flow.shape[2]
            padding = last_frame.repeat(1, 1, num_repeats, 1, 1)
            flow = torch.cat([flow, padding], dim=2)
            
        return flow

    def _load_rgb_frames(self, video_path: str) -> torch.Tensor:
        """Load RGB video frames."""
        video_reader = VideoReader(video_path, width=self.W, height=self.H, num_threads=2)
        
        if len(video_reader) < 1:
            raise ValueError(f"No frames in video: {video_path}")
            
        # Sample frames
        if len(video_reader) > self.max_num_frames * 2:
            indices = list(range(0, self.max_num_frames * 2, 2))[:self.max_num_frames]
        else:
            indices = list(range(len(video_reader)))
            
        frames = video_reader.get_batch(indices).asnumpy()
        frames = [torch.from_numpy(frames[0]).float() for _ in range(len(frames))]  # Use first frame
        
        # Pad frames if necessary
        if len(frames) < self.max_num_frames:
            last_frame = frames[-1]
            frames.extend([last_frame.clone() for _ in range(self.max_num_frames - len(frames))])
        else:
            frames = frames[:self.max_num_frames]
        
        frames_tensor = torch.stack(frames)
        frames_tensor = frames_tensor / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).contiguous()
        frames_tensor = self.video_transforms(frames_tensor)
        
        return frames_tensor

    def _process_image_sample(self, data_info: Dict, idx: int) -> Dict:
        """Process image sample."""
        image_path = data_info['file_path']
        prompt = data_info['text']
        
        if self.data_root is not None:
            image_path = os.path.join(self.data_root, image_path)
        
        if random.random() < self.text_drop_ratio:
            prompt = ''
        
        image = Image.open(image_path).convert('RGB')
        image = np.expand_dims(np.array(image), 0)
        control_image = image.copy()
        
        return {
            "pixel_values": image,
            "control_pixel_values": control_image,
            "text": prompt,
            "data_type": "image",
            "idx": idx
        }