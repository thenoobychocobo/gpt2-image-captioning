from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from src.database import image_store, faiss_store
from src.utils import load_gpt2_tokenizer


class MLPMappingNetwork(nn.Module):
    """
    Maps an image embedding vector to a sequence of prefix tokens (learned vector representations)
    using a simple Multi-Layer Perceptron (MLP).

    The prefix tokens capture image information from the image embedding vector and are used as
    virtual tokens to condition GPT-2 decoding for image-captioning.
    """

    def __init__(
        self,
        prefix_length: int = 10,
        embed_dim: int = 512,
        gpt_dim: int = 768,
        bias: bool = True,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        """
        Args:
            prefix_length (int): The number of prefix tokens to generate.
            embed_dim (int): The dimensionality of the image embedding vector. Defaults to 512.
            gpt_dim (int): The dimensionality of each GPT-2 embedding space. Since the prefix tokens are meant
                to be passed into GPT-2 as virtual tokens, this will be the dimensionality of each prefix token.
                Defaults to 768.
            bias (bool, optional): Whether to include a bias term in the linear layers. Defaults to True.
            activation (nn.Module, optional): The activation function to use between linear layers. Defaults to nn.Tanh().
        """
        super().__init__()
        self.prefix_length = prefix_length
        self.embed_dim = embed_dim
        self.gpt_dim = gpt_dim

        # Output is a flat vector that will later be split into prefix_length number of gpt_dim vectors (tokens)
        output_dim = prefix_length * gpt_dim

        # Let the bottleneck size (hidden layer) be half the output size
        hidden_dim = output_dim // 2

        self.model = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=bias),
            activation,
            nn.Linear(hidden_dim, output_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a sequence of prefix tokens derived from the input image embedding vectors.

        Args:
            x (torch.Tensor): Image embedding vectors of shape (batch_size, embed_dim)

        Returns:
            torch.Tensor: Sequence of prefix tokens of shape (batch_size, prefix_length, gpt_dim)
        """
        # x shape: (batch_size, embed_dim)

        # Project to (batch_size, prefix_length * gpt_dim)
        x = self.model(x)

        # Split the flat output vector into a sequence of prefix_length vectors
        return x.view(x.shape[0], self.prefix_length, self.gpt_dim)


class TransformerMappingNetwork(nn.Module):
    """
    Maps an image embedding vector to a sequence of prefix tokens (learned vector representations)
    using a Transformer Encoder (bidirectional self-attention).

    Process:
    1. The image embedding vector is projected to a sequence of 'hidden_length' vectors (which we will refer to as the sequence of image tokens).
    2. There is also a learnable prefix, composed of 'prefix_length' learned constant vectors.
    3. The image token sequence and the learnable prefix are concatenated together and then passed through a Transformer Encoder.
    4. After encoding, we extract only the learnable prefix part (the last 'prefix_length' tokens) to be used as input to GPT-2 (virtual tokens).

    Each vector in the prefix is a learned constant vector, that is optimized during training. We refer to each as a prefix token.
    The prefix tokens are akin to the `[CLS]` token in BERT, but here we have multiple such learned tokens.
    The idea is that when the prefix tokens are processed together with the image token sequence and attend to each other (bidirectional self-attention),
    they effectively learn to retrieve meaningful information from the image embedding.
    The processed prefix tokens (with image information captured post-attention) are then passed to GPT-2 as virtual tokens to condition
    text generation (decoding).
    """

    def __init__(
        self,
        embed_dim: int,
        gpt_dim: int,
        prefix_length: int,
        hidden_length: int,
        num_layers: int = 8,
    ) -> None:
        """
        Args:
            embed_dim (int): The dimensionality of the image embedding vector.
            gpt_dim (int): The dimensionality of the GPT token embeddings.
            prefix_length (int): The number of prefix tokens to generate.
            hidden_length (int): The number of tokens to project the image embedding vector into.
            num_layers (int, optional): The number of Transformer encoder layers. Defaults to 8.
        """
        super().__init__()
        self.embed_dim = embed_dim  # Image embedding dimension (e.g., 512)
        self.gpt_dim = gpt_dim  # GPT token embedding dimension (e.g., 768)
        self.hidden_length = hidden_length
        self.prefix_length = prefix_length

        # Linear layer projects image embedding vector to a sequence of 'hidden_length' vectors
        self.linear = nn.Linear(embed_dim, hidden_length * gpt_dim)

        # Learnable Prefix (learned constant vectors later used as virtual tokens for GPT-2)
        # These prefix tokens will attend to the image information in the image token sequence via self-attention
        # We want to learn an optimal intialization for these prefix tokens
        self.prefix_const = nn.Parameter(
            torch.randn(prefix_length, gpt_dim), requires_grad=True
        )

        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gpt_dim,
            nhead=8,  # TODO: make configurable
            dim_feedforward=int(gpt_dim * 4),
            batch_first=True,
            activation="relu",
            norm_first=True,
        )

        # Transformer Encoder composed of multiple Encoder Layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Image embedding vectors of shape (batch_size, embed_dim)

        Returns:
            torch.Tensor: Sequence of prefix tokens of shape (batch_size, prefix_length, gpt_dim)
        """
        # x shape: (batch_size, embed_dim)
        batch_size = x.shape[0]

        # Project image embedding vector
        # (batch_size, embed_dim) to (batch_size, hidden_length * gpt_dim)
        x = self.linear(x)

        # Split this long vector into a sequence of vectors ("image tokens")
        x = x.view(
            batch_size, self.hidden_length, self.gpt_dim
        )  # (batch_size, hidden_length, gpt_dim)

        # Duplicate the learnable prefix for each input image (image embedding) in the batch
        # (prefix_length, gpt_dim) to (batch_size, prefix_length, gpt_dim)
        prefix = self.prefix_const.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate the projected image sequence with the learnable prefix
        inputs = torch.cat(
            (x, prefix), dim=1
        )  # (batch_size, hidden_length + prefix_length, gpt_dim)

        # Pass the entire sequence through the Transformer Encoder
        out = self.transformer(inputs)

        # Extract and return only the prefix tokens (the last 'prefix_length' tokens)
        return out[:, self.hidden_length :, :]  # (batch_size, prefix_length, gpt_dim)


class ImageCaptioningModel(nn.Module):
    """
    End-to-end Image Captioning model that takes in an image embedding and generates an image-caption using GPT-2.
    Combines a Mapping Network (that maps image embeddings to GPT prefix tokens) and a pretrained GPT-2 model.
    """

    def __init__(
        self,
        mapping_network: nn.Module,
        image_prefix_length: int | None = None,
        prefix_task_prompt: str | None = None,
        tokenizer: GPT2Tokenizer | None = None,
        gpt: GPT2LMHeadModel | None = None,
        freeze_gpt_weights: bool = True,
    ) -> None:
        """
        Args:
            mapping_network (nn.Module): Initialized mapping network that maps image embeddings to GPT prefix tokens.
            image_prefix_length (int, optional): The number of image prefix tokens from the mapping network.
                Defaults to None, in which case it is inferred from the mapping network's `prefix_length` attribute.
            prefix_task_prompt (str | None, optional): If provided, a task-specific text prompt that is used to initialize
                learnable prefix tokens (which we refer to as task prefix tokens) that are prepended to the image prefix tokens.
                This allows the model to adapt to specific tasks based on the prompt. Defaults to None.
            gpt (GPT2LMHeadModel | None, optional): Pretrained GPT-2 model. Defaults to None, in which case the standard GPT-2 model is loaded.
            freeze_gpt_weights (bool, optional): Explicitly freezes GPT-2 weights if True. Otherwise, will explicitly unfreeze them if False.
                Defaults to True.
        """
        # TODO: Have Mapping Network be an identifiable subclass

        super().__init__()
        self.image_prefix_length = image_prefix_length or mapping_network.prefix_length
        self.mapping_network = mapping_network

        # Load GPT-2 model and tokenizer
        self.gpt = gpt or GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.tokenizer = tokenizer or load_gpt2_tokenizer()

        # Explicitly freeze or unfreeze weights for GPT-2
        for param in self.gpt.parameters():
            param.requires_grad = not freeze_gpt_weights

        # Task-specific prompt prefix
        self.task_prefix_embeds: nn.Parameter | None = None
        if prefix_task_prompt:
            with torch.no_grad():
                # Tokenize the text prompt to get vector embeddings
                task_token_ids = self.tokenizer.encode(
                    prefix_task_prompt,
                    return_tensors="pt",
                )
                task_token_embeds = self.gpt.transformer.wte(task_token_ids)

            # Learnable task-specific prefix tokens
            # We use the token embeddings of the task prompt as the initial values
            self.task_prefix_embeds = nn.Parameter(
                task_token_embeds.squeeze(0),
                requires_grad=True,  # Trainable!
            )  # (num_task_prompt_tokens, gpt_dim)

    def forward(
        self,
        caption_token_ids: torch.Tensor,
        image_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple | CausalLMOutputWithCrossAttentions:
        """
        Forward pass for training with teacher forcing.
        Calculates the loss between generated caption tokens and ground truth caption tokens.

        For actual caption generation, use the `generate` method instead.

        Args:
            caption_token_ids (torch.Tensor): Ground truth caption token IDs of shape (batch_size, caption_length)
            image_embeddings (torch.Tensor): Image embeddings of shape (batch_size, embed_dim)
            attention_mask (torch.Tensor | None, optional): Attention mask for the caption tokens of shape (batch_size, caption_length). Defaults to None.
            labels (torch.Tensor | None, optional): Labels for computing loss of shape (batch_size, caption_length). Defaults to None.

        Returns:
            tuple | CausalLMOutputWithCrossAttentions: The output from the GPT-2 model, which includes the loss (if labels are provided) and the generated logits.
        """

        # Obtain GPT token embeddings for the ground-truth captions
        caption_tokens = self.gpt.transformer.wte(caption_token_ids)

        # Obtain the image prefix tokens from the image embeddings via the mapping network
        prefix_tokens = self.mapping_network(
            image_embeddings
        )  # (batch_size, prefix_length, gpt_dim)

        # Obtain the task-specific prompt prefix tokens (if any)
        if self.task_prefix_embeds is not None:
            # Duplicate the task-specific prefix for each input in the batch
            # (task_prefix_length, gpt_dim) to (batch_size, task_prefix_length, gpt_dim)
            batch_size = image_embeddings.shape[0]
            task_prefix_tokens = self.task_prefix_embeds.unsqueeze(0).expand(
                batch_size, -1, -1
            )

            # Concatenate the task prefix tokens with the image prefix tokens
            prefix_tokens = torch.cat(
                (prefix_tokens, task_prefix_tokens), dim=1
            )  # TODO: Determine which order is better

        # Determine total prefix length
        total_prefix_length = prefix_tokens.shape[1]

        # Concatenate prefix tokens + ground-truth caption tokens
        input_tokens = torch.cat((prefix_tokens, caption_tokens), dim=1)

        # Handle Labels (labels simply denote which tokens to compute loss on)
        if labels is not None:
            # Labels should be provided to mark padding tokens that should be ignored in loss computation (set to -100)
            # Add dummy labels for the prefix tokens so that we don't train on them
            # We want to compute loss only on the caption tokens, GPT-2 should not be trained to predict the prefix tokens
            # NOTE: The mapping network is still trained to produce good prefix tokens via loss/error signal from the generated caption tokens

            dummy_labels = torch.full(
                (
                    labels.shape[0],
                    total_prefix_length,
                ),  # (batch size, number of prefix tokens)
                -100,  # -100 is the standard PyTorch ignore_index for CrossEntropyLoss
                dtype=torch.int64,
                device=caption_token_ids.device,
            )
            labels = torch.cat((dummy_labels, labels), dim=1)

        # Handle Attention Mask
        if attention_mask is not None:
            # Add attention mask of 1s for the prefix tokens, ensuring they are attended to
            dummy_attention_mask = torch.ones(
                (
                    attention_mask.shape[0],
                    total_prefix_length,
                ),  # (batch size, number of prefix tokens)
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((dummy_attention_mask, attention_mask), dim=1)

        # GPT forward pass
        # Clear implementation can be found here: https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_gpt2.html#GPT2LMHeadModel
        return self.gpt.forward(
            inputs_embeds=input_tokens,
            labels=labels,  # All labels set to -100 are ignored for loss computation
            attention_mask=attention_mask,
        )

    def generate(
        self,
        image_embeddings: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate captions autoregressively given image embeddings.

        Args:
            image_embeddings (torch.Tensor): Image embeddings of shape (batch_size, embed_dim)
            max_length (int, optional): Maximum length of the generated caption. Defaults to 50.
            temperature (float, optional): Sampling temperature for controlling randomness. Defaults to 1.0.
            top_p (float, optional): Nucleus sampling probability threshold. Defaults to 0.9.

        Returns:
            torch.Tensor: Generated caption tokens of shape (batch_size, generated_length)
        """

        # Get the stop token ID (EOS token for GPT-2)
        stop_token_id = self.tokenizer.eos_token_id

        # Set to evaluation mode
        self.eval()  # Will recursively set mapping network and GPT to eval mode

        # Get device and batch size
        device = image_embeddings.device
        batch_size = image_embeddings.shape[0]

        # Process image embeddings through the mapping network to get image prefix tokens
        with torch.no_grad():
            prefix_tokens = self.mapping_network.forward(
                image_embeddings
            )  # (batch_size, prefix_length, gpt_dim)

        # Obtain the task-specific prompt prefix tokens (if any)
        if self.task_prefix_embeds is not None:
            # Duplicate the task-specific prefix for each input in the batch
            # (task_prefix_length, gpt_dim) to (batch_size, task_prefix_length, gpt_dim)
            batch_size = image_embeddings.shape[0]
            task_prefix_tokens = self.task_prefix_embeds.unsqueeze(0).expand(
                batch_size, -1, -1
            )

            # Concatenate the task prefix tokens with the image prefix tokens
            prefix_tokens = torch.cat(
                (prefix_tokens, task_prefix_tokens), dim=1
            )  # TODO: Determine which order is better

        # Initialize input for GPT-2 decoding
        current_input_tokens = prefix_tokens

        # List to store the generated token IDs for each step (for each item in the batch)
        generated_tokens: list[torch.Tensor] = []  # Each item has shape (batch_size, 1)

        # Boolean tensor to track which sequences have hit the stop token
        is_finished = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )  # (batch_size,)

        # Loop for autoregressive decoding (generation)
        for _ in range(max_length):
            if is_finished.all():
                break  # Stop the entire loop if all sequences are complete

            with torch.no_grad():
                # Pass current input to GPT
                outputs = self.gpt.forward(inputs_embeds=current_input_tokens)

                # Get logits of the last token only (for next token prediction)
                next_token_logits = outputs.logits[:, -1, :]

                # Apply temperature scaling (controls randomness)
                next_token_logits = next_token_logits / (
                    temperature if temperature > 0 else 1.0
                )

                # Apply top-p (nucleus) sampling
                # Next token selection is restricted to a subset of tokens whose cumulative probability exceeds the threshold p
                if top_p < 1.0 and temperature > 0:
                    # Mask the finished sequences by setting their logits to 0
                    # This ensures that once a sequence has generated the stop token, it will not generate any further tokens
                    next_token_logits[is_finished, :] = 0.0

                    # Sort the logits in descending order and compute cumulative probabilities
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p

                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    # Scatter the mask back to original indicies
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )

                    # Set removed logits to negative infinity so that they are never sampled
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                # Sample from the predicted next-token distribution
                if temperature == 0:
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(
                        -1
                    )  # (batch_size, 1)
                else:
                    # For finished sequences, the logits were set to 0, so the probabibility distribution is uniform on the first token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(
                        probs, num_samples=1
                    )  # Sample from the distribution (batch_size, 1)

                # Update the is_finished tracker
                # Check which sequences just generated the stop token
                is_current_stop = next_token_id.squeeze(-1).eq(stop_token_id)
                # An item is finished if it was already finished OR if it just generated the stop token
                is_finished = torch.logical_or(is_finished, is_current_stop)

                # If a sequence is finished, force its next token to be EOS token
                next_token_id[is_finished] = (
                    stop_token_id  # This will be stripped away later when decoding to text (skip_special_tokens=True)
                )

                # Track tokens generated
                generated_tokens.append(next_token_id)

                # Prepare input for next iteration by appending the newly generated token
                next_token = self.gpt.transformer.wte(next_token_id)
                current_input_tokens = torch.cat(
                    (current_input_tokens, next_token), dim=1
                )

        if not generated_tokens:
            # No tokens were generated (e.g., max_length was 0), return an empty tensor
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)

        # Return generated tokens (to be passed to a tokenizer for decoding to text)
        # TODO: Use self.tokenizer to decode directly
        return torch.cat(generated_tokens, dim=1)  # (batch_size, generated_length)

    def generate_captions(self, image_embeddings: torch.Tensor, **kwargs) -> list[str]:
        """
        Convenience method to generate and decode strings directly.
        """
        generated_ids = self.generate(image_embeddings, **kwargs)
        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,  # Strips away special tokens like <eos>
        )

    def save_parameters(self, output_path: str) -> None:
        """
        Saves all model parameters and buffers except for the GPT-2 weights if they are frozen.
        If GPT-2 weights are not frozen, all parameters are saved.
        Use the `load_partial_state_dict` method to load these parameters back into the model.

        Args:
            output_path (str): The file path to save the trainable parameters.
        """
        # Get all trainable parameter names
        trainable_param_names = {
            name for name, param in self.named_parameters() if param.requires_grad
        }

        # Filter state dict to include only:
        # 1. Trainable parameters (already in trainable_param_names set)
        # 2. Buffers (e.g., running mean/var in BatchNorm layers) belonging to trainable modules
        # We are essentially saving everything except for GPT-2 weights if they are frozen.
        keys_to_save = {}

        for name, param in self.state_dict().items():
            if name in trainable_param_names:
                keys_to_save[name] = param
            # If we froze GPT, we will save everything NOT inside 'gpt'
            elif not name.startswith("gpt."):
                keys_to_save[name] = param

        print(
            f"Saving {len(keys_to_save)} trainable parameters and buffers to {output_path}."
        )
        torch.save(keys_to_save, output_path)

    def load_saved_parameters(
        self, checkpoint_path: str, device: torch.device | None = None
    ) -> None:
        """
        Loads a state dictionary (containing saved model parameters) from the specified checkpoint path.
        This is useful for loading only the trained parameters and buffers into the model (see `save_parameters`).

        Args:
            checkpoint_path (str): The file path of the checkpoint to load.
            device (torch.device | None, optional): The device to map the loaded parameters to. Defaults to None.
        """
        # Load the state dictionary from the checkpoint (may not include GPT-2 weights)
        state_dict = torch.load(checkpoint_path, map_location=device)
        keys = self.load_state_dict(state_dict, strict=False)

        # Validation: check for unexpected keys
        if keys.unexpected_keys:
            raise ValueError(
                f"Unexpected keys found in the checkpoint: {keys.unexpected_keys}"
            )

        # Validation: check that any missing keys are only from frozen GPT weights and not from anything else
        non_gpt_missing = [k for k in keys.missing_keys if not k.startswith("gpt.")]
        if non_gpt_missing:
            raise ValueError(
                f"Missing keys found in the checkpoint that are not from frozen GPT weights: {non_gpt_missing}"
            )


class RetrievalAggregator(nn.Module):
    """
    Aggregates retrieved caption embeddings using various pooling strategies.
    Based on the RAT paper's comparison of aggregation functions.
    """

    def __init__(
        self,
        embed_dim: int,
        aggregation_type: Literal["mean", "max", "sum_norm", "attention"] = "mean",
    ):
        """
        Args:
            embed_dim: Dimension of embeddings
            aggregation_type:
                - "mean": Standard average pooling (best per paper)
                - "max": Max pooling over retrieved embeddings
                - "sum_norm": Sum of L2-normalized features followed by L2-norm
                - "attention": Learnable attention weights (not in paper, but good alternative)
        """
        super().__init__()
        self.aggregation_type = aggregation_type
        self.embed_dim = embed_dim

        if aggregation_type == "attention":
            # Learnable attention for comparison
            self.attention_proj = nn.Linear(embed_dim, 1)

    def forward(
        self, query_embedding: torch.Tensor, retrieved_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_embedding: (batch_size, embed_dim) - original image embedding
            retrieved_embeddings: (batch_size, top_k, embed_dim) - retrieved caption embeddings

        Returns:
            (batch_size, embed_dim) - aggregated embedding combined with query
        """
        if self.aggregation_type == "mean":
            # Standard average pooling (best according to paper)
            aggregated = retrieved_embeddings.mean(dim=1)  # (batch_size, embed_dim)

        elif self.aggregation_type == "max":
            # Max pooling
            aggregated = retrieved_embeddings.max(dim=1)[0]  # (batch_size, embed_dim)

        elif self.aggregation_type == "sum_norm":
            # Sum of L2-normalized features followed by L2-norm
            # Normalize each retrieved embedding
            retrieved_normalized = F.normalize(
                retrieved_embeddings, p=2, dim=2
            )  # (batch_size, top_k, embed_dim)
            # Sum across top_k
            summed = retrieved_normalized.sum(dim=1)  # (batch_size, embed_dim)
            # L2-normalize the result
            aggregated = F.normalize(summed, p=2, dim=1)  # (batch_size, embed_dim)

        elif self.aggregation_type == "attention":
            # Learnable attention weights (for comparison)
            attn_scores = self.attention_proj(
                retrieved_embeddings
            )  # (batch_size, top_k, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, top_k, 1)
            aggregated = (retrieved_embeddings * attn_weights).sum(
                dim=1
            )  # (batch_size, embed_dim)

        else:
            raise ValueError(f"Unknown aggregation_type: {self.aggregation_type}")

        # Combine aggregated retrieval with query image embedding
        # Simple addition (you could also try concatenation + projection)
        augmented = query_embedding + aggregated

        return augmented


class RetrievalAugmentedTransformer(ImageCaptioningModel):
    """
    Retrieval-Augmented Transformer for image captioning.
    Retrieves similar captions and aggregates them to augment the input image embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        max_workers: int = 4,
        aggregation_type: Literal[
            "mean", "max", "sum_norm", "attention"
        ] = "mean",  # mean is the best method in the paper
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            embed_dim: Dimension of image embeddings (e.g., 512 for CLIP)
            aggregation_type: Method to aggregate retrieved embeddings.
                "mean" (default) performs best according to the paper.
            *args, **kwargs: Arguments passed to ImageCaptioningModel
        """
        super().__init__(*args, **kwargs)
        self.max_workers = max_workers
        self.aggregator = RetrievalAggregator(embed_dim, aggregation_type)
    
    def _retrieve_batch(
        self,
        db_store,
        image_embeddings: torch.Tensor,
        top_i: int,
        top_k: int,
    ) -> torch.Tensor:
        """
        Retrieve caption embeddings for a batch of images.
        Handles both FAISS (fast batch ops) and ObjectBox (threaded).
        
        Returns:
            torch.Tensor of shape (batch_size, top_k, embed_dim)
        """
        batch_size = image_embeddings.shape[0]
        device = image_embeddings.device
        
        # Check if it's a FAISS store (has image_index attribute)
        is_faiss = hasattr(db_store, 'image_index')

        if is_faiss:
            # FAISS: Native batch search (MUCH faster!)
            query_vectors = image_embeddings.cpu().numpy().astype('float32')
            
            # Batch similarity search
            batch_image_results = faiss_store.retrieve_images_by_vector_similarity(
                db_store, query_vectors, top_i
            )
            
            # Extract filenames for each query
            batch_filenames = [
                [filename for filename, _ in results] 
                for results in batch_image_results
            ]
            
            # Batch caption retrieval
            caption_embeds_np = faiss_store.get_caption_embeddings(
                db_store, top_k, batch_filenames, embed_dim=512
            )
            
            # Convert to tensor
            return torch.from_numpy(caption_embeds_np).to(device)
        
        else:
            # ObjectBox: Use threading for better performance
            all_retrieved_embeddings = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        image_store.retrieve_for_single_embedding,
                        image_embeddings[i : i + 1],
                        db_store,
                        top_i,
                        top_k,
                        device,
                    )
                    for i in range(batch_size)
                ]
                all_retrieved_embeddings = [f.result() for f in futures]
            
            return torch.stack(all_retrieved_embeddings, dim=0)
        
    def forward(
        self,
        db_store,
        top_i: int,  # number of similar images to retrieve, not included in paper
        top_k: int,  # number of captions to retrieve, according to the paper, k=10 or k=20 works well
        caption_token_ids: torch.Tensor,
        image_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple | CausalLMOutputWithCrossAttentions:
        """
        Forward pass with retrieval augmentation.
        Retrieves top-k similar captions and augments the image embeddings.
        """

        # Retrieve caption embeddings (handles both FAISS and ObjectBox)
        retrieved_embeddings = self._retrieve_batch(
            db_store, image_embeddings, top_i, top_k
        )

        # Aggregate retrieved embeddings and combine with query
        augmented_embeddings = self.aggregator(image_embeddings, retrieved_embeddings)

        # Pass augmented embeddings to parent class forward
        return super().forward(
            caption_token_ids=caption_token_ids,
            image_embeddings=augmented_embeddings,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(
        self,
        db_store,
        top_k: int,
        top_i: int,
        image_embeddings: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate captions with retrieval augmentation.
        """

        # Retrieve caption embeddings (handles both FAISS and ObjectBox)
        retrieved_embeddings = self._retrieve_batch(
            db_store, image_embeddings, top_i, top_k
        )

        # Aggregate retrieved embeddings and combine with query
        augmented_embeddings = self.aggregator(image_embeddings, retrieved_embeddings)

        # Generate using parent class method
        return super().generate(augmented_embeddings, max_length, temperature, top_p)

    
    def generate_captions(self, db_store, top_k: int, top_i: int, image_embeddings: torch.Tensor, **kwargs) -> list[str]:
        """
        Convenience method to generate and decode strings directly.
        """
        generated_ids = self.generate(image_embeddings, db_store, top_k, top_i, **kwargs)
        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,  # Strips away special tokens like <eos>
        )

    def save_parameters(self, output_path: str) -> None:
        """
        Saves all model parameters and buffers except for the GPT-2 weights if they are frozen.
        If GPT-2 weights are not frozen, all parameters are saved.
        Use the `load_partial_state_dict` method to load these parameters back into the model.

        Args:
            output_path (str): The file path to save the trainable parameters.
        """
        # Get all trainable parameter names
        trainable_param_names = {
            name for name, param in self.named_parameters() if param.requires_grad
        }

        # Filter state dict to include only:
        # 1. Trainable parameters (already in trainable_param_names set)
        # 2. Buffers (e.g., running mean/var in BatchNorm layers) belonging to trainable modules
        # We are essentially saving everything except for GPT-2 weights if they are frozen.
        keys_to_save = {}

        for name, param in self.state_dict().items():
            if name in trainable_param_names:
                keys_to_save[name] = param
            # If we froze GPT, we will save everything NOT inside 'gpt'
            elif not name.startswith("gpt."):
                keys_to_save[name] = param

        print(
            f"Saving {len(keys_to_save)} trainable parameters and buffers to {output_path}."
        )
        torch.save(keys_to_save, output_path)
