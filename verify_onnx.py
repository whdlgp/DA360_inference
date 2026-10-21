import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from depth_creator import DA360DepthCreator


def preprocess_image(image_path, width, height):
    """Read and preprocess an image for inference."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    original_shape = img.shape[:2]
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    return image, original_shape


def run_torch(model, image):
    """Run PyTorch inference."""
    image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = ((image - mean) / std).unsqueeze(0)

    with torch.inference_mode():
        disparity = model(image)["pred_disp"][0, 0].numpy()

    return disparity


def run_onnx(session, image):
    """Run ONNX inference."""
    image = np.ascontiguousarray(image.transpose(2, 0, 1)[None]).astype(np.float32)
    disparity = session.run(["disparity"], {"image": image})[0][0, 0]

    return disparity


def visualize_disparity(disparity_torch, disparity_onnx):
    """Display PyTorch and ONNX disparity maps."""
    disparity_min = min(disparity_torch.min(), disparity_onnx.min())
    disparity_max = max(disparity_torch.max(), disparity_onnx.max())
    if disparity_max == disparity_min:
        disparity_max += 1e-6

    disparity_torch = np.clip((disparity_torch - disparity_min) / (disparity_max - disparity_min) * 255, 0, 255).astype(np.uint8)
    disparity_onnx = np.clip((disparity_onnx - disparity_min) / (disparity_max - disparity_min) * 255, 0, 255).astype(np.uint8)

    disparity_torch = cv2.applyColorMap(disparity_torch, cv2.COLORMAP_TURBO)
    disparity_onnx = cv2.applyColorMap(disparity_onnx, cv2.COLORMAP_TURBO)

    cv2.putText(disparity_torch, "PyTorch", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(disparity_onnx, "ONNX", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    result = np.vstack((disparity_torch, disparity_onnx))
    cv2.imshow("Disparity Comparison", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def verify(checkpoint_path, onnx_path, image_path, visualize=False):
    """Verify ONNX inference against the original PyTorch model."""
    checkpoint_path = Path(checkpoint_path)
    onnx_path = Path(onnx_path) if onnx_path else checkpoint_path.with_suffix(".onnx")

    # Load models
    ## ONNX model. The input size is fixed in the graph.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _, _, height, width = session.get_inputs()[0].shape

    ## Original model
    model = DA360DepthCreator(checkpoint_path, device="cpu").model

    # Read input image
    image, original_shape = preprocess_image(image_path, width, height)

    # Run models
    out_torch = run_torch(model, image)
    out_onnx = run_onnx(session, image)

    # Compare disparity
    a = out_torch.astype(np.float64)
    b = out_onnx.astype(np.float64)
    diff = np.abs(a - b)
    tolerance = 1e-4 + 1e-3 * np.abs(b)
    failed = diff > tolerance
    print("max abs diff   :", diff.max())
    print("mean abs diff  :", diff.mean())
    print("failed pixels  :", np.count_nonzero(failed))
    print("total pixels   :", diff.size)
    print("failed ratio   :", np.mean(failed))
    y, x = np.where(failed)
    for iy, ix in zip(y, x):
        print(f"({iy}, {ix}): torch={a[iy, ix]:.8f}, onnx={b[iy, ix]:.8f}, diff={diff[iy, ix]:.8f}, tolerance={tolerance[iy, ix]:.8f}")
    print("allclose(1e-4):", np.allclose(a, b, atol=1e-4, rtol=1e-3))

    # Display disparity maps
    if visualize:
        visualize_disparity(a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify ONNX export against original PyTorch model.")
    parser.add_argument("--checkpoint", default="checkpoints/DA360_small.pth", help="path to .pth checkpoint file")
    parser.add_argument("--onnx_path", default=None, help="path to exported ONNX file (default: checkpoint path with .onnx)")
    parser.add_argument("--image", default="input/input.jpg", help="path to the input image")
    parser.add_argument("--visualize", action="store_true", help="display disparity maps")
    args = parser.parse_args()
    verify(checkpoint_path=args.checkpoint, onnx_path=args.onnx_path, image_path=args.image, visualize=args.visualize)