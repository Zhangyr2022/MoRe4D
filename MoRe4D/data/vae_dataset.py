import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_sceneflow(video_path, posfix):
    with open(video_path, "r", encoding="utf-8") as file:
        return [Path(str(video_path.parent)) / line.replace("videos", "dt3d"+posfix).replace(".mp4", "_dt3d_pred.pkl").strip() for line in file.readlines() if len(line.strip()) > 0]


class VAEDataset(Dataset):
    """
    Base dataset class for Scene Flow prompt training.

    This dataset loads prompts, videos and corresponding conditioning masked videos for V2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        video_column: str,
        posfix: str,
        max_frames: int = 17
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        self.H, self.W = 384, 512
        self.max_frames = max_frames
        if os.path.exists(video_column):
            self.scene_flows = load_sceneflow(Path(video_column), posfix=posfix)
        else:
            self.scene_flows = load_sceneflow(data_root / video_column, posfix=posfix)

    def __len__(self):
        return len(self.scene_flows)

    def __getitem__(self, index):
        scene_flow_path = str(self.scene_flows[index])
        # print(scene_flow_path,self.scene_flows[index])
        if not os.path.exists(str(scene_flow_path)):
            assert str(scene_flow_path).split("/")[-4] == "webvid_tmp_13", f"Scene flow path {scene_flow_path} does not exist"
            scene_flow_path = Path(str(scene_flow_path).replace("webvid_tmp_13", "webvid_tmp_13_vepfs2"))
        with open(scene_flow_path, "rb") as f:
            scene_flow = pickle.load(f)
        if len(scene_flow["coords"].shape) == 3:
            scene_flow["coords"] = scene_flow["coords"].unsqueeze(0)
        if len(scene_flow["colors"].shape) == 2:
            scene_flow["colors"] = scene_flow["colors"].unsqueeze(0)
        B, T = scene_flow["coords"].shape[:2]
        scene_flow["coords"] = scene_flow["coords"].reshape(B, T, self.H, self.W, 3).permute(0, 4, 1, 2, 3)
        # scene_flow["coords_delta"] = scene_flow["coords"][:, :, 1:, :, :] - scene_flow["coords"][:, :, :-1, :, :]
        scene_flow["coords_delta"] = scene_flow["coords"] - scene_flow["coords"][:, :, 0:1, :, :]
        scene_flow["coords_normalized"] = scene_flow["coords"][:, :, :self.max_frames, :, :] / torch.abs(scene_flow["coords"][:, :, :self.max_frames, :, :]).max()
        scene_flow["colors"] = scene_flow["colors"].reshape(B, self.H, self.W, 3).permute(0, 3, 1, 2)
        scene_flow["vis"] = scene_flow["vis"].reshape(B, 1, T, self.H, self.W)
        return scene_flow