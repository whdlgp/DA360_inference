from pathlib import Path

import cv2
import numpy as np
import torch

from .depth_creator import DepthCreator
from .DA360.imple_da360 import DA360


class DA360DepthCreator(DepthCreator):
    """DA360-based ERP depth creator."""

    def __init__(self, checkpoint_path: str | Path, front_distance_mm: float = 3000.0, device: str | torch.device | None = None) -> None:
        """Load the DA360 model.

        Args:
            checkpoint_path: path to the model checkpoint.
            front_distance_mm: target depth at the center region, in millimeters.
            device: inference device. Defaults to CUDA when available.
        """
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.front_distance_mm = float(front_distance_mm)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.metadata = self._load_model()

    def _load_model(self) -> tuple[DA360, dict[str, str | int]]:
        """Load the model and its metadata from the checkpoint.

        Returns:
            Loaded DA360 model and checkpoint metadata.
        """
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        metadata = {
            "net": checkpoint.get("net", "DA360"),
            "height": int(checkpoint.get("height", 518)),
            "width": int(checkpoint.get("width", 1036)),
            "dinov2_encoder": checkpoint.get("dinov2_encoder", "vits"),
        }
        if metadata["net"] != "DA360":
            raise ValueError(f"Unsupported model architecture: {metadata['net']}")

        model = DA360(metadata["height"], metadata["width"], dinov2_encoder=metadata["dinov2_encoder"])
        model_state = model.state_dict()
        state_dict = {}
        for key, target in model_state.items():
            value = checkpoint.get(key)
            if value is None or not torch.is_tensor(value):
                if key.endswith(".conv.weight"):
                    value = checkpoint.get(key[:-len(".conv.weight")] + ".weight")
                elif key.endswith(".conv.bias"):
                    value = checkpoint.get(key[:-len(".conv.bias")] + ".bias")
                elif key.endswith(".weight") and not key.endswith(".conv.weight"):
                    value = checkpoint.get(key[:-len(".weight")] + ".conv.weight")
                elif key.endswith(".bias") and not key.endswith(".conv.bias"):
                    value = checkpoint.get(key[:-len(".bias")] + ".conv.bias")
            if torch.is_tensor(value):
                state_dict[key] = value.to(dtype=target.dtype)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if not key.endswith("num_batches_tracked")]
        if missing:
            raise RuntimeError(f"Checkpoint is missing {len(missing)} model parameters; first missing keys: {missing[:10]}")
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")

        model.to(self.device).eval()
        return model, metadata

    def run(self, img: np.ndarray) -> np.ndarray:
        """Create an ERP depth map from a BGR image.

        Args:
            img: src ERP image, shape (H, W, 3), BGR uint8.

        Returns:
            ERP depth map (radius) in millimeters, shape (H, W), uint16.
        """
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("img must be a 3-channel BGR image")

        original_height, original_width = img.shape[:2]
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.metadata["width"], self.metadata["height"]), interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = ((image - mean) / std).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    disparity = self.model(image)["pred_disp"]
            else:
                disparity = self.model(image)["pred_disp"]

        disparity = disparity[0, 0].float().cpu().numpy()
        disparity = cv2.resize(disparity, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
        valid = np.isfinite(disparity) & (disparity > 1e-6)
        depth = np.zeros(disparity.shape, dtype=np.float32)
        depth[valid] = 1.0 / disparity[valid]

        height, width = depth.shape
        y0, y1 = height * 2 // 5, height * 3 // 5
        x0, x1 = width * 2 // 5, width * 3 // 5
        front_region = depth[y0:y1, x0:x1]
        front_valid = np.isfinite(front_region) & (front_region > 0)
        if not np.any(front_valid):
            raise ValueError("No valid depth values found in the center region")

        front_depth = float(np.median(front_region[front_valid]))
        depth[valid] *= self.front_distance_mm / front_depth
        return np.clip(np.round(depth), 0, 65535).astype(np.uint16)
