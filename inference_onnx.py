from pathlib import Path

import argparse
import cv2
import numpy as np
import onnxruntime as ort


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


def preprocess_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Preprocess a BGR image for ONNX inference."""
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    image = image.transpose(2, 0, 1)
    image = image[None].astype(np.float32)
    image = np.ascontiguousarray(image)

    return image


def run_onnx(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    """Run ONNX inference and return disparity."""
    disparity = session.run(["disparity"], {"image": image})[0][0, 0]

    return disparity


def disparity_to_depth(disparity: np.ndarray, front_distance_mm: float = 5000.0) -> np.ndarray:
    """Convert disparity to a uint16 depth map in millimeters."""
    valid = np.isfinite(disparity) & (disparity > 1e-6)
    depth = np.zeros(disparity.shape, dtype=np.float32)
    depth[valid] = 1.0 / disparity[valid]

    height, width = depth.shape
    y0, y1 = height * 2 // 5, height * 3 // 5
    x0, x1 = width * 2 // 5, width * 3 // 5
    center = depth[y0:y1, x0:x1]
    center_valid = np.isfinite(center) & (center > 0)

    if not np.any(center_valid):
        raise ValueError("No valid depth values found in the center region")

    front_depth = float(np.median(center[center_valid]))
    depth[valid] *= front_distance_mm / front_depth

    return np.clip(np.round(depth), 0, 65535).astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX model on an input image.")
    parser.add_argument("--onnx_path", default="checkpoints/DA360_small.onnx", help="path to ONNX model")
    args = parser.parse_args()

    # Read image
    input_dir = Path("input")
    image_file = Path("input.jpg")
    img = read_image(input_dir / image_file)

    # ONNX model
    onnx_path = Path(args.onnx_path)
    # Depth at the center region, mm
    front_distance_mm = 5000.0

    # Init
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _, _, height, width = session.get_inputs()[0].shape

    # Preprocess
    image = preprocess_image(img, width, height)

    # Inference
    disparity = run_onnx(session, image)
    depth = disparity_to_depth(disparity, front_distance_mm)

    # Save
    output_dir = Path("output")
    output_file = image_file.with_name(f"{image_file.stem}_{onnx_path.stem}_depth.png")
    save_depth(depth, output_dir / output_file)
    output_vis_file = image_file.with_name(f"{image_file.stem}_{onnx_path.stem}_depth_vis.png")
    save_depth_vis(depth, output_dir / output_vis_file)


if __name__ == "__main__":
    main()