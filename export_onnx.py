import argparse
from pathlib import Path

import torch
import torch.nn as nn

from depth_creator import DA360DepthCreator
from depth_creator.DA360 import imple_da360


class Wrapper(nn.Module):
    """Bakes in the input normalization: RGB image in 0-255 -> disparity."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model((image / 255.0 - self.mean) / self.std)["pred_disp"]


def export(checkpoint_path: str | Path, output_path: str | Path | None, height: int | None, width: int | None, simplify: bool = False) -> None:
    """Export a DA360 checkpoint to an ONNX file with a fixed input shape.

    Args:
        checkpoint_path: path to the model checkpoint.
        output_path: output ONNX file path. Defaults to the checkpoint path with the .onnx suffix.
        height: input height. Defaults to the height stored in the checkpoint.
        width: input width. Defaults to the width stored in the checkpoint.
        simplify: simplify the exported graph with onnxsim.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path) if output_path else checkpoint_path.with_suffix(".onnx")

    # Load model
    depth_creator = DA360DepthCreator(checkpoint_path, device="cpu")
    height = height or depth_creator.metadata["height"]
    width = width or depth_creator.metadata["width"]
    model = Wrapper(depth_creator.model).eval()

    image = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (image,), str(output_path),
        input_names=["image"], output_names=["disparity"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    print("Saved", output_path)

    if simplify:
        try:
            import onnx
            import onnxsim

            model_onnx = onnx.load(str(output_path))
            model_opt, check = onnxsim.simplify(model_onnx)
            if not check:
                print("Simplifier: validation check failed, keeping original model.")
            else:
                onnx.save(model_opt, str(output_path))
                print("Simplification done.")
        except Exception as e:
            print(f"Simplifier failure: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DA360 checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="checkpoints/DA360_small.pth", help="path to .pth checkpoint file")
    parser.add_argument("--output", default=None, help="output ONNX file path (default: checkpoint path with .onnx)")
    parser.add_argument("--height", type=int, default=None, help="export input height (default: from checkpoint)")
    parser.add_argument("--width", type=int, default=None, help="export input width (default: from checkpoint)")
    parser.add_argument("--simplify", action="store_true", help="simplify exported ONNX graph")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(checkpoint_path=args.checkpoint,
           output_path=args.output,
           height=args.height,
           width=args.width,
           simplify=args.simplify)
