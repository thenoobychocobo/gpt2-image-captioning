# Imports

import torch
from transformers import set_seed

from src.dataset import CocoDataset
from src.models import ImageCaptioningModel, TransformerMappingNetwork
from src.train import train
from src.utils import load_gpt2_tokenizer, load_config

if __name__ == "__main__":
    config = load_config("config.yml")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBEDDINGS_PATH = config['paths']['embeddings_path']
    ANNOTATIONS_PATH = config['paths']['annotations_path']
    CHECKPOINTS_PATH = config['paths']['checkpoints_path']
    MAX_CAPTION_LENGTH = config['max_caption_length']

    # Transformer Mapping Network Params
    EMBED_DIM = config['mapping_network']['embed_dim']  # Embedding dimension
    GPT_DIM = config['mapping_network']['gpt_dim']  # GPT-2 embedding dimension
    PREFIX_LENGTH = config['mapping_network']['prefix_length']  # Length of the prefix
    HIDDEN_LENGTH = config['mapping_network']['hidden_length']

    # Image Captioning Model Params
    FREEZE_GPT_WEIGHTS = config['image_captioning']['freeze_gpt_weights']  # Whether to fine-tune GPT-2 alongside the mapping network
    PREFIX_TASK_PROMPT: str | None = config['image_captioning']['prefix_task_prompt']

    # Training Params
    TRAIN_BATCH_SIZE = config['training']['batch_size']
    NUM_EPOCHS = config['training']['num_epochs']
    NUM_WORKERS = config['training']['num_workers']
    LEARNING_RATE = config['training']['learning_rate']
    NUM_WARMUP_STEPS = config['training']['num_warmup_steps']
    SAVE_EVERY_EPOCH = config['training']['save_every_epoch']

    # Validation Params
    EVAL_EVERY_EPOCH = config['validation']['eval_every_epoch']
    EVAL_BATCH_SIZE = config['validation']['batch_size']
    EVAL_MAX_CAPTION_LENGTH = MAX_CAPTION_LENGTH
    EVAL_TEMPERATURE = config['validation']['temperature']  # Greedy
    EVAL_TOP_P = config['validation']['top_p']

    # Tokenizer
    gpt2_tokenizer = load_gpt2_tokenizer()

    # Training Dataset
    train_dataset = CocoDataset(
        embeddings_path=EMBEDDINGS_PATH + "train_clip_embeddings.pt",
        annotations_path=ANNOTATIONS_PATH + "captions_train2017.json",
        tokenizer=gpt2_tokenizer,
        max_length=MAX_CAPTION_LENGTH,
        normalize_embeddings=False,  # `.pt` files already contain normalized embeddings
    )

    # Validation Dataset
    val_dataset = CocoDataset(
        embeddings_path=EMBEDDINGS_PATH + "val_clip_embeddings.pt",
        annotations_path=ANNOTATIONS_PATH + "captions_val2017.json",
        tokenizer=gpt2_tokenizer,
        max_length=MAX_CAPTION_LENGTH,
        normalize_embeddings=False,
    )

    # Note: No test dataset as test annotations are not publicly available

    # Models
    mapping_network = TransformerMappingNetwork(
        embed_dim=EMBED_DIM,
        gpt_dim=GPT_DIM,
        prefix_length=PREFIX_LENGTH,
        hidden_length=HIDDEN_LENGTH,
    )

    model = ImageCaptioningModel(
        mapping_network=mapping_network,
        prefix_task_prompt=PREFIX_TASK_PROMPT,
        tokenizer=gpt2_tokenizer,
        freeze_gpt_weights=FREEZE_GPT_WEIGHTS,
    ).to(DEVICE)

    print(model)

    # Train image captioning model with validation evaluation
    train_history = train(
        # Training params
        train_dataset=train_dataset,
        model=model,
        batch_size=TRAIN_BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        num_workers=NUM_WORKERS,
        learning_rate=LEARNING_RATE,
        num_warmup_steps=NUM_WARMUP_STEPS,
        save_every_epoch=SAVE_EVERY_EPOCH,
        device=DEVICE,
        outputs_dir=CHECKPOINTS_PATH,
        # Eval params
        val_dataset=val_dataset,
        val_annotations_path=ANNOTATIONS_PATH + "captions_val2017.json",
        eval_every_epoch=EVAL_EVERY_EPOCH,
        eval_batch_size=EVAL_BATCH_SIZE,
        eval_max_length=EVAL_MAX_CAPTION_LENGTH,
        eval_temperature=EVAL_TEMPERATURE,
        eval_top_p=EVAL_TOP_P,
    )

    print(
        f"\nBest validation CIDEr: {train_history['best_val_cider']:.4f} at epoch {train_history['best_epoch']}"
    )

    print(train_history)