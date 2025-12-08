# Image Captioning with GPT-2

This repository contains code for generating image captions using the GPT-2 large language model. The goal of the project is to see how far we can push the capabilities of GPT-2 (a fairly small and weak model by today's standards) in the image captioning task, by implementing various techniques and supporting modules.

**Table of Contents**
- [Getting Started](#getting-started)
  - [Environment Setup](#environment-setup)
    - [Using `uv` (Recommended)](#using-uv-recommended)
    - [Using `pip` and `venv`](#using-pip-and-venv)
  - [Dataset Download](#dataset-download)
  - [Extracting Image Embeddings](#extracting-image-embeddings)
- [Methodology](#methodology)

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/thenoobychocobo/gpt2-image-captioning.git
cd gpt2-image-captioning
```

### Environment Setup

This project was developed and tested on Python 3.13. It may work on other versions, but compatibility is not guaranteed.

#### Using `uv` (Recommended)

We recommend using [`uv`](https://docs.astral.sh/uv/), a fast Python package and project manager, written in Rust. You can install `uv` by following the installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/). Alternatively, you can install via `pip`:

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

#### Using `pip` and `venv` 

Alternatively, you can use `pip` and `venv`. You can create a virtual environment and install the required packages using:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Download

This project uses the [COCO dataset](https://cocodataset.org/#home): specifically the 2017 Train, Validation, and Test sets. 

While the [official download instructions](https://cocodataset.org/#download) recommends using `gsutil rsync` for efficient downloading of the datasets, this method does not currently work (as of December 2025) as the `gs://images.cocodataset.org` bucket cannot be found (does not exist). Refer to this [raised issue](https://github.com/cocodataset/cocoapi/issues/368).

Instead, we provide a [`download_coco_datasets.sh`](download_coco_datasets.sh) shell script that uses `wget` or `curl` to download the necessary files. You can run this script as follows:

```bash
./download_coco_datasets.sh
```

This will create a new `coco_data/` directory in your project root containing the COCO datasets. The directory structure should look like this:

``` 
coco_data/
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   └── image_info_test2017.json
├── train2017
├── val2017
└── test2017
```

## Usage

### Vision Encoder: Extracting Image Embeddings

We currently have code for extracting image embeddings using three different Vision Encoders: (1) [CLIP](#clip); (2) [ViT](#vit); and (3) [DINOv3](#dinov3) (with `dino.txt` adapters).

#### CLIP

[CLIP](https://arxiv.org/abs/2103.00020) is a multimodal vision and language model that unifies image and text representations into a shared embedding space. We specifically use the CLIP image encoder as the Vision Encoder to extract image embeddings.

To extract image embeddings using CLIP, simply run the following notebook: [`/notebooks/extract_clip_embeddings.ipynb`](./notebooks/extract_clip_embeddings.ipynb)

#### ViT

You can also use any Vision Transformer (ViT) model from HuggingFace as the Vision Encoder. To extract image embeddings using ViT, simply run the following notebook: [`/notebooks/extract_vit_embeddings.ipynb`](./notebooks/extract_vit_embeddings.ipynb)

#### DINOv3

[DINOv3](https://github.com/facebookresearch/dinov3) is a state-of-the-art vision transformer model developed by Meta AI. It is designed to learn visual representations in a self-supervised manner, enabling it to perform well on various computer vision tasks without requiring large amounts of labeled data. 

To use DINOv3 as the vision encoder, we utilize the `dino.txt` adapters, which are lightweight modules that can be added on top of the pre-trained DINOv3 model to for text alignment.

First, head to this [link](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) (from the official repository) and request access to the DINOv3 models and adapters. Once you have access, you will be sent an email containing the download links to all pretrained DINOv3 models. Download the `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` model (backbone) and the `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth` (`dino.txt` text-alignment adapters) and place them in a directory of your choice.

Then update your virtual environment to include the additional dependencies required for DINOv3:
```bash
uv sync --group dinov3
```

To extract image embeddings using DINOv3 with `dino.txt` adapters, simply run the following notebook: [`/notebooks/extract_dino_embeddings.ipynb`](./notebooks/extract_dino_embeddings.ipynb). Do remember to update the paths to the downloaded model and adapter weights in the notebook before running it.

## Training

To train the GPT-2 image captioning model, run the following notebook: [`/notebooks/train.ipynb`](./notebooks/train.ipynb). For the mapping network architecture, there is a choice between a simple 2-layer MLP or a Transformer-based mapping network. Be sure to adjust the various hyperparameters in the notebook as needed.

## Evaluation

To evaluate your trained model on the test set, run the following notebook: [`/notebooks/eval.ipynb`](./notebooks/eval.ipynb). Be sure to update the path to your saved model checkpoint before running the notebook.

## Visualization

To visualize some sample captions generated by your trained model, add the required dependencies:
```bash
uv sync --group visualization
```
Then run the visualization script as follows:
```bash
uv run visualize_my_results.py
```

## Methodology

**Section is WIP**