# DA360_inference
An unofficial inference-only implementation of DA360 for 360° equirectangular depth estimation

![da360_depth_comparison](demo_output/da360_depth_comparison.png)

## Checkpoints

Download checkpoints [here](https://huggingface.co/hljiang/DA360).  
Place the downloaded checkpoints in the `checkpoints/` directory.

```text
checkpoints/
├── DA360_base.pth
├── DA360_large.pth
├── DA360_small.pth
└── checkpoints_here.txt
```

## input data

Place the input file in the `input/` directory.

```text
input/
├── input.jpg
└── input_here.txt
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
uv run inference.py
```

## ONNX

Export, verify, and run inference for each model.

```bash
# DA360 Large
uv run export_onnx.py --checkpoint checkpoints/DA360_large.pth --simplify
uv run verify_onnx.py --checkpoint checkpoints/DA360_large.pth --visualize
uv run inference_onnx.py --onnx_path checkpoints/DA360_large.onnx

# DA360 Base
uv run export_onnx.py --checkpoint checkpoints/DA360_base.pth --simplify
uv run verify_onnx.py --checkpoint checkpoints/DA360_base.pth --visualize
uv run inference_onnx.py --onnx_path checkpoints/DA360_base.onnx

# DA360 Small
uv run export_onnx.py --checkpoint checkpoints/DA360_small.pth --simplify
uv run verify_onnx.py --checkpoint checkpoints/DA360_small.pth --visualize
uv run inference_onnx.py --onnx_path checkpoints/DA360_small.onnx
```

## Results

<details>
<summary>Input image</summary>

![input_image](input/input.jpg)

</details>

<details>
<summary>DA360 Large depth visualized</summary>

![DA360 Large](demo_output/input_DA360_large_depth_vis.png)

</details>

<details>
<summary>DA360 Base depth visualized</summary>

![DA360 Base](demo_output/input_DA360_base_depth_vis.png)

</details>

<details>
<summary>DA360 Small depth visualized</summary>

![DA360 Small](demo_output/input_DA360_small_depth_vis.png)

</details>
