# Image Captioning with GPT-2

This repository contains code and instructions for generating image captions using a GPT-2 model fine-tuned on image captioning datasets.

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/thenoobychocobo/gpt2-image-captioning.git
cd gpt2-image-captioning
```

### Environment Setup

#### Using `uv` (Recommended)

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

#### Using `pip` and `venv`

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
This project uses the [COCO dataset](https://cocodataset.org/#home); specifically the 2017 Train, Validation, and Test sets. 

While the [official download instructions](https://cocodataset.org/#download) recommends using `gsutil rsync` for efficient downloading of the datasets, this method does not work as the `gs://images.cocodataset.org` bucket does not exist (as of December 2025). Refer to this [raised issue](https://github.com/cocodataset/cocoapi/issues/368).

Instead, we provide a `download_coco_datasets.sh` shell script that uses `wget` to download the necessary files. You can run this script as follows:

```bash
./download_coco_datasets.sh
```

This will create a new `coco_data/` directory in your project root containing the COCO datasets.    