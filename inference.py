from pathlib import Path

import cv2
import numpy as np

from depth_creator import DA360DepthCreator


def read_image(image_path: str | Path) -> np.ndarray:
    """Read a BGR image from a file."""
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return img


def save_depth(depth: np.ndarray, output_path: str | Path) -> None:
    """Save a depth map as an image."""
    output_path = Path(output_path)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"depth must be a 1-channel uint16 image, got {depth.dtype} {depth.shape}")
    if not cv2.imwrite(str(output_path), depth):
        raise OSError(f"Could not save depth image: {output_path}")

    print(f"Saved depth map: {output_path}")


def save_depth_vis(depth: np.ndarray, output_path: str | Path) -> None:
    """Save a colorized depth map for visualization."""
    depth = depth.astype(np.float32)
    normalized_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_visualization = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_TURBO)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), depth_visualization):
        raise OSError(f"Could not save depth visualization: {output_path}")

    print(f"Saved depth visualization: {output_path}")


def main() -> None:
    # Read image
    input_dir = Path("input")
    image_file = Path("input.jpg")
    img = read_image(input_dir / image_file)

    # DA360 checkpoints
    checkpoint_paths = [
        Path("checkpoints/DA360_large.pth"),
        Path("checkpoints/DA360_base.pth"),
        Path("checkpoints/DA360_small.pth"),
    ]
    # Depth at the center region, mm 
    front_distance_mm = 3000.0

    for checkpoint_path in checkpoint_paths:
        # init
        depth_creator = DA360DepthCreator(checkpoint_path, front_distance_mm=front_distance_mm)
        
        # inference
        depth = depth_creator.run(img)

        # Save
        output_dir = Path("output")
        output_file = image_file.with_name(f"{image_file.stem}_{checkpoint_path.stem}_depth.png")
        save_depth(depth, output_dir / output_file)
        output_vis_file = image_file.with_name(f"{image_file.stem}_{checkpoint_path.stem}_depth_vis.png")
        save_depth_vis(depth, output_dir / output_vis_file)


if __name__ == "__main__":
    main()
