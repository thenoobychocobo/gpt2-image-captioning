# Image Captioning with GPT-2

This repository contains code and instructions for generating image captions using a GPT-2 model fine-tuned on image captioning datasets.

## Getting Started

### Environment Setup

This project was developed and tested on Python 3.13. It may work on other versions, but compatibility is not guaranteed.

We recommend using [`uv`](https://docs.astral.sh/uv/), a fast Python package and project manager, written in Rust. You can install `uv` via `pip` or by following the installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/). 

```bash
pip install uv
```

Then, simply synchronize your virtual environment using:

```bash
uv sync
```

Do note that PyTorch installation may be dependent on your CUDA version. Check your CUDA version with:

```bash
nvidia-smi
```

The `pyproject.toml` file has already been configured to target CUDA 12.8:

```toml
...
# For PyTorch installation
[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

If you are targetting a different CUDA version (or CPU only), simply refer to the [uv documentation](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch) and modify the `pyproject.toml` file accordingly.

---

If you'd prefer to use `pip` and `venv`, you can create a virtual environment and install the required packages using:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Download
This project uses the [COCO dataset](https://cocodataset.org/#home), specifically the 2014 Train, Validation, and Test sets. Refer to this [page](https://cocodataset.org/#download) for the official download instructions.