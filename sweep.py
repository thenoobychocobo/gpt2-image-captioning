import os
import time
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import set_seed
import logging

from src.dataset import CocoDataset
from src.models import ImageCaptioningModel, TransformerMappingNetwork, MLPMappingNetwork
from src.train import train
from src.utils import load_gpt2_tokenizer, load_config, count_model_parameters

CHECKPOINTS_DIR = "checkpoints/"

# Load base configuration and automation updates configurations
updates = load_config("automation_config.yml")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def update_cfg(base_cfg: DictConfig, updates: dict):
    """
    Update the main cfg with values from updates dict
    """
    for k, v in updates.items():
        if k in base_cfg:
            if isinstance(v, dict):
                # Recurse into nested dict
                update_cfg(base_cfg[k], v)
            else:
                base_cfg[k] = v
        else:
            base_cfg[k] = v
    

def training_pipeline(cfg: DictConfig, checkpoints_dir: str):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed) 
    
    # Tokenizer
    gpt2_tokenizer = load_gpt2_tokenizer()

    # Training Dataset
    train_dataset = CocoDataset(
        embeddings_path=cfg.paths.embeddings_path + "train_clip_embeddings.pt",
        annotations_path=cfg.paths.annotations_path + "captions_train2017.json",
        tokenizer=gpt2_tokenizer,
        max_length=cfg.max_caption_length,
        normalize_embeddings=False,  # `.pt` files already contain normalized embeddings
    )

    # Validation Dataset
    val_dataset = CocoDataset(
        embeddings_path=cfg.paths.embeddings_path + "val_clip_embeddings.pt",
        annotations_path=cfg.paths.annotations_path + "captions_val2017.json",
        tokenizer=gpt2_tokenizer,
        max_length=cfg.max_caption_length,
        normalize_embeddings=False,
    )

    # Mapping Network
    params = {
        "embed_dim": cfg.mapping.embed_dim,
        "gpt_dim": cfg.mapping.gpt_dim,
        "prefix_length": cfg.mapping.prefix_length
    }
    
    # Note: No test dataset as test annotations are not publicly available
    if cfg.mapping.type == "transformer":
        params["hidden_length"] = cfg.mapping.hidden_length
        mapping_network = TransformerMappingNetwork(**params)
    elif cfg.mapping.type == "mlp":
        mapping_network = MLPMappingNetwork(**params)
    else:
        logger.error(f"Unknown mapping network type: {cfg.mapping.type}")
        raise ValueError(f"Unknown mapping network type: {cfg.mapping.type}")
        
    # Image Captioning Model
    model = ImageCaptioningModel(
        mapping_network=mapping_network,
        prefix_task_prompt=cfg.image_captioning.prefix_task_prompt,
        tokenizer=gpt2_tokenizer,
        freeze_gpt_weights=cfg.image_captioning.freeze_gpt_weights,
    ).to(DEVICE)

    # Train image captioning model with validation evaluation
    train_history = train(
        # Training params
        train_dataset=train_dataset,
        model=model,
        batch_size=cfg.training.batch_size,
        num_epochs=cfg.training.num_epochs,
        num_workers=cfg.training.num_workers,
        learning_rate=cfg.training.learning_rate,
        num_warmup_steps=cfg.training.num_warmup_steps,
        save_every_epoch=cfg.training.save_every_epoch,
        device=DEVICE,
        outputs_dir=checkpoints_dir,
        # Eval params
        val_dataset=val_dataset,
        val_annotations_path=cfg.paths.annotations_path + "captions_val2017.json",
        eval_every_epoch=cfg.validation.eval_every_epoch,
        eval_batch_size=cfg.validation.batch_size,
        eval_max_length=cfg.max_caption_length,
        eval_temperature=cfg.validation.temperature,
        eval_top_p=cfg.validation.top_p,
    )

    print(
        f"\nBest validation CIDEr: {train_history['best_val_cider']:.4f} at epoch {train_history['best_epoch']}"
    )

    print(train_history)
    return model

def set_up(name: str) -> str:
    # make new directory if not exists
    dir_path = os.path.join(CHECKPOINTS_DIR, name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

if __name__ == "__main__":
    logger.info("Starting automated training...")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    cfg = OmegaConf.load("config.yml")
    updates_cfg = updates['model']

    for i, cfg_idx in enumerate(updates_cfg.keys()):
        logger.info(f"Processing model {i+1}/{len(updates_cfg)}: {cfg_idx}")
        cfg = OmegaConf.load("config.yml")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        current_cfg_dict = updates_cfg[cfg_idx]
        save_dir = set_up(f"checkpoint_{cfg_idx}")

        logger.info(f"Updating configuration for model {cfg_idx}...")
        update_cfg(cfg, current_cfg_dict)

        logger.info(f"Configuration updated. Starting training for model {cfg_idx}...")
        # Save config file for this run
        with open(os.path.join(save_dir, f"config_{cfg_idx}.yml"), "w") as f:
            OmegaConf.save(cfg, f)

        # Track the model training time
        start_time = time.time()
        model = training_pipeline(cfg, save_dir)
        end_time = time.time()
        training_duration = end_time - start_time # currently in seconds
        training_duration_str = time.strftime("%H:%M:%S", time.gmtime(training_duration))
        logger.info(f"Training completed in {training_duration_str} for model {cfg_idx}")

        # Count number of model parameters
        trainable_params, total_params = count_model_parameters(model)
        logger.info(f"Model parameters for model {cfg_idx}: Trainable={trainable_params}, Total={total_params}")

        # Save training duration and parameter counts to a txt file
        with open(os.path.join(save_dir, f"training_info_{cfg_idx}.txt"), "w") as f:
            f.write(f"Training Duration (seconds): {training_duration:.2f}\n")
            f.write(f"Trainable Parameters: {trainable_params}\n")
            f.write(f"Total Parameters: {total_params}\n")

        # TODO: Evaluation