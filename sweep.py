import json
import logging
import os
import time

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from src.database.faiss_store import create_faiss_store
from src.dataset import CocoDataset
from src.eval import generate_and_evaluate, generate_and_evaluate_rat
from src.models import (
    ImageCaptioningModel,
    MLPMappingNetwork,
    RetrievalAugmentedTransformer,
    TransformerMappingNetwork,
)
from src.train import train, train_rat
from src.utils import count_model_parameters, load_config, load_gpt2_tokenizer

# Configure logging BEFORE creating loggers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = "checkpoints/"

# Load base configuration and automation updates configurations
updates = load_config("automation_config.yml")


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


def training_pipeline(cfg: DictConfig, save_dir: str):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # Tokenizer
    gpt2_tokenizer = load_gpt2_tokenizer()

    # Training Dataset
    train_dataset = CocoDataset(
        embeddings_path=cfg.paths.embeddings_path
        + f"train_{cfg.vision_encoder}_embeddings.pt",
        annotations_path=cfg.paths.annotations_path + "captions_train2017.json",
        tokenizer=gpt2_tokenizer,
        max_length=cfg.max_caption_length,
        normalize_embeddings=False,  # `.pt` files already contain normalized embeddings
    )

    # Validation Dataset
    val_dataset = CocoDataset(
        embeddings_path=cfg.paths.embeddings_path
        + f"val_{cfg.vision_encoder}_embeddings.pt",
        annotations_path=cfg.paths.annotations_path + "captions_val2017.json",
        tokenizer=gpt2_tokenizer,
        max_length=cfg.max_caption_length,
        normalize_embeddings=False,
    )

    # Mapping Network
    mapping_network_params = {
        "embed_dim": cfg.mapping.embed_dim,
        "gpt_dim": cfg.mapping.gpt_dim,
        "prefix_length": cfg.mapping.prefix_length,
    }

    # Note: No test dataset as test annotations are not publicly available
    if cfg.mapping.type == "transformer":
        mapping_network_params["hidden_length"] = cfg.mapping.hidden_length
        mapping_network = TransformerMappingNetwork(**mapping_network_params)
    elif cfg.mapping.type == "mlp":
        mapping_network = MLPMappingNetwork(**mapping_network_params)
    else:
        logger.error(f"Unknown mapping network type: {cfg.mapping.type}")
        raise ValueError(f"Unknown mapping network type: {cfg.mapping.type}")

    training_eval_params = {
        # Train params
        "train_dataset": train_dataset,
        "batch_size": cfg.training.batch_size,
        "num_epochs": cfg.training.num_epochs,
        "num_workers": cfg.training.num_workers,
        "learning_rate": cfg.training.learning_rate,
        "num_warmup_steps": cfg.training.num_warmup_steps,
        "save_every_epoch": cfg.training.save_every_epoch,
        "device": DEVICE,
        "outputs_dir": save_dir,
        # Eval params
        "val_dataset": val_dataset,
        "val_annotations_path": cfg.paths.annotations_path + "captions_val2017.json",
        "eval_every_epoch": cfg.validation.eval_every_epoch,
        "eval_batch_size": cfg.validation.batch_size,
        "eval_max_length": cfg.max_caption_length,
        "eval_temperature": cfg.validation.temperature,
        "eval_top_p": cfg.validation.top_p,
    }

    if cfg.retrieval_augmentation:  # Retrieval Augmented Training (RAT)
        # Load vector db (use FAISS for all experiments as it is faster)
        faiss_store = create_faiss_store(db_directory=cfg.paths.vector_db_path)

        # Retrieval-Augmented Transformer Model
        model = RetrievalAugmentedTransformer(
            embed_dim=cfg.retrieval.embed_dim,
            aggregation_type=cfg.retrieval.aggregation_type,
            mapping_network=mapping_network,
            prefix_task_prompt=cfg.image_captioning.prefix_task_prompt,
            tokenizer=gpt2_tokenizer,
            freeze_gpt_weights=cfg.image_captioning.freeze_gpt_weights,
        ).to(DEVICE)

        training_eval_params.update(
            {
                "model": model,
                "db_store": faiss_store,
                "top_k": cfg.retrieval.top_k,
                "top_i": cfg.retrieval.top_i,
            }
        )

        # Train RAT model with validation evaluation
        train_history = train_rat(**training_eval_params)
        db_store = faiss_store
    else:  # Standard
        # Image Captioning Model
        model = ImageCaptioningModel(
            mapping_network=mapping_network,
            prefix_task_prompt=cfg.image_captioning.prefix_task_prompt,
            tokenizer=gpt2_tokenizer,
            freeze_gpt_weights=cfg.image_captioning.freeze_gpt_weights,
        ).to(DEVICE)

        training_eval_params.update({"model": model})

        # Train image captioning model with validation evaluation
        train_history = train(**training_eval_params)
        db_store = None  # No db_store for standard training

    print(
        f"\nBest validation CIDEr: {train_history['best_val_cider']:.4f} at epoch {train_history['best_epoch']}"
    )

    print(train_history)
    return model, gpt2_tokenizer, db_store


def set_up(name: str) -> str:
    # make new directory if not exists
    dir_path = os.path.join(CHECKPOINTS_DIR, name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


if __name__ == "__main__":
    logger.info("Starting automated training...")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    cfg = OmegaConf.load("config.yml")
    updates_cfg = updates["model"]

    for i, cfg_idx in enumerate(updates_cfg.keys()):
        logger.info(f"Processing model {i + 1}/{len(updates_cfg)}: {cfg_idx}")
        cfg = OmegaConf.load("config.yml")
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
        model, tokenizer, db_store = training_pipeline(cfg, save_dir)
        end_time = time.time()
        train_val_duration = end_time - start_time  # currently in seconds
        train_val_duration_str = time.strftime(
            "%H:%M:%S", time.gmtime(train_val_duration)
        )
        logger.info(
            f"Training and validation completed in {train_val_duration_str} for model {cfg_idx}"
        )

        # Count number of model parameters
        trainable_params, total_params = count_model_parameters(model)
        logger.info(
            f"Model parameters for model {cfg_idx}: Trainable={trainable_params}, Total={total_params}"
        )

        # Save training duration and parameter counts to a txt file
        with open(os.path.join(save_dir, f"training_info_{cfg_idx}.txt"), "w") as f:
            f.write(
                f"Training and Validation Duration (seconds): {train_val_duration_str}\n"
            )
            f.write(f"Trainable Parameters: {trainable_params}\n")
            f.write(f"Total Parameters: {total_params}\n")

        # Evaluate model on test set
        logger.info(f"Starting evaluation on test set for model {cfg_idx}...")

        # Test Dataset (Using validation 2014 as test set)
        test_dataset = CocoDataset(
            embeddings_path=cfg.paths.embeddings_path
            + f"test_{cfg.vision_encoder}_embeddings.pt",
            annotations_path=cfg.paths.annotations_path + "captions_val2014.json",
            tokenizer=tokenizer,
            max_length=cfg.max_caption_length,
            normalize_embeddings=False,
        )

        testing_params = {
            "model": model,
            "dataset": test_dataset,
            "annotations_path": cfg.paths.annotations_path + "captions_val2014.json",
            "batch_size": cfg.validation.batch_size,
            "max_length": cfg.max_caption_length,
            "temperature": cfg.validation.temperature,
            "top_p": cfg.validation.top_p,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        if cfg.retrieval_augmentation:  # Retrieval Augmented
            testing_params.update(
                {
                    "db_store": db_store,
                    "top_k": cfg.retrieval.top_k,
                    "top_i": cfg.retrieval.top_i,
                }
            )
            predictions, metrics = generate_and_evaluate_rat(**testing_params)
        else:  # Standard
            predictions, metrics = generate_and_evaluate(**testing_params)

        # Save predictions and metrics
        with open(os.path.join(save_dir, f"test_predictions_{cfg_idx}.txt"), "w") as f:
            json.dump(predictions, f)
        with open(os.path.join(save_dir, f"test_metrics_{cfg_idx}.txt"), "w") as f:
            f.write(str(metrics))
        logger.info(f"Evaluation completed for model {cfg_idx}. Metrics saved.")

    logger.info("All model trainings and evaluations completed.")
    logger.info("Exiting...")
