import json
import logging
import os

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Tokenizer

from src.dataset import CocoDataset
from src.eval import generate_and_evaluate
from src.models import (
    ImageCaptioningModel,
    MLPMappingNetwork,
    TransformerMappingNetwork,
)
from src.utils import load_gpt2_tokenizer

# Configure logging BEFORE creating loggers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = "checkpoints/checkpoints/"


def testing_pipeline(
    cfg: DictConfig,
    model: ImageCaptioningModel,
    tokenizer: GPT2Tokenizer,
    save_dir: str,
):
    # Test Dataset (Using validation 2014 as test set)
    test_dataset = CocoDataset(
        embeddings_path=cfg.paths.embeddings_path
        + f"test_{cfg.vision_encoder}_embeddings.pt",
        annotations_path=cfg.paths.annotations_path + "captions_val2014.json",
        tokenizer=tokenizer,
        max_length=cfg.max_caption_length,
        normalize_embeddings=False,
    )

    predictions, metrics = generate_and_evaluate(
        model=model,
        dataset=test_dataset,
        annotations_path=cfg.paths.annotations_path + "captions_val2014.json",
        batch_size=cfg.validation.batch_size,
        max_length=cfg.max_caption_length,
        temperature=cfg.validation.temperature,
        top_p=cfg.validation.top_p,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir=save_dir,
    )
    return predictions, metrics


if __name__ == "__main__":
    # look through all checkpoints_i directories in CHECKPOINTS_DIR
    checkpoint_dirs = [d for d in os.listdir(CHECKPOINTS_DIR)]

    # Find the best model based on best_model_i.pt where i is the higherst number
    tokenizer = load_gpt2_tokenizer()

    # get all files in checkpoint_dir
    for _, i in enumerate(checkpoint_dirs):
        checkpoint_dir = os.path.join(CHECKPOINTS_DIR, i)
        all_files = os.listdir(checkpoint_dir)
        best_model_files = [
            f for f in all_files if f.startswith("best_model_") and f.endswith(".pt")
        ]
        # get index from best_model_i.pt
        best_model_indices = [
            int(f.split("_")[2].split(".")[0]) for f in best_model_files
        ]
        best_model_index = max(best_model_indices)

        # Load config
        cfg_idx = _ + 1
        cfg = OmegaConf.load(os.path.join(checkpoint_dir, f"config_{cfg_idx}.yml"))

        # Load mapping network from config
        params = {
            "embed_dim": cfg.mapping.embed_dim,
            "gpt_dim": cfg.mapping.gpt_dim,
            "prefix_length": cfg.mapping.prefix_length,
        }
        if cfg.mapping.type == "transformer":
            params["hidden_length"] = cfg.mapping.hidden_length
            mapping_network = TransformerMappingNetwork(**params)
        elif cfg.mapping.type == "mlp":
            mapping_network = MLPMappingNetwork(**params)
        else:
            raise ValueError(f"Unknown mapping network type: {cfg.mapping.type}")

        # Load model
        model = ImageCaptioningModel(
            mapping_network=mapping_network,
            prefix_task_prompt=cfg.image_captioning.prefix_task_prompt,
            tokenizer=tokenizer,
            freeze_gpt_weights=cfg.image_captioning.freeze_gpt_weights,
        )
        model.load_saved_parameters(
            os.path.join(checkpoint_dir, f"best_model_{best_model_index}.pt")
        )

        # Evaluate on test set
        logger.info(f"Starting evaluation for model {cfg_idx}...")
        if cfg.retrieval_augmentation:
            # TODO: Implement RAT evaluation pipeline
            pass
        else:
            predictions, metrics = testing_pipeline(
                cfg, model, tokenizer, checkpoint_dir
            )

        # Save predictions and metrics
        with open(
            os.path.join(checkpoint_dir, f"test_predictions_{cfg_idx}.txt"), "w"
        ) as f:
            json.dump(predictions, f)
        with open(
            os.path.join(checkpoint_dir, f"test_metrics_{cfg_idx}.txt"), "w"
        ) as f:
            f.write(str(metrics))
        logger.info(f"Evaluation completed for model {cfg_idx}. Metrics saved.")

        logger.info("All model evaluation completed.")
        logger.info("Exiting...")
