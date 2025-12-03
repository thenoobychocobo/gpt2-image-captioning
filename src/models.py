from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


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


class EncoderLayer(nn.Module):
    """
    Defines a Encoder Layer that we are using for the TransformerMappingNetwork, choose between:
    1. A pure Transformer Encoder Layer
    2. A CNN based Encoder Layer
    3. A Hybrid CNN --> Transformer Encoder Layer
    """

    def __init__(
        self,
        gpt_dim: int,
        layer_type: Literal["transformer", "cnn", "hybrid"],
        cnn_kernel_size: int = 3,
        num_heads: int = 8,
        dim_feedforward: int | None = None,
    ) -> None:
        """
        Args:
            gpt_dim (int): The dimensionality of the GPT token embeddings.
            use_cnn (bool, optional): Whether to use a CNN based Encoder Layer. Defaults to False.
            cnn_kernel_size (int, optional): Kernel size for the CNN Encoder Layer. Only relevant if `layer_type` is "cnn" or "hybrid".
                Defaults to 3.
            num_heads (int, optional): Number of attention heads for Transformer Encoder Layer. Defaults to 8.
            dim_feedforward (int | None, optional): Dimensionality of the feedforward layer in Transformer Encoder Layer.
                Defaults to None, in which case it is set to 4 times the `gpt_dim`.
        """
        super().__init__()
        self.layer_type = layer_type
        dim_feedforward = dim_feedforward or int(gpt_dim * 4)

        if self.layer_type == "transformer":
            # Transformer Encoder Layer
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=gpt_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                activation="relu",
                norm_first=True,
            )
        elif self.layer_type == "cnn":
            # CNN based Encoder Layer
            padding = (cnn_kernel_size - 1) // 2  # To maintain sequence length
            self.cnn = nn.Conv1d(
                in_channels=gpt_dim,
                out_channels=gpt_dim,
                kernel_size=cnn_kernel_size,
                padding=padding,
            )
            self.norm = nn.LayerNorm(gpt_dim)
            self.activation = nn.ReLU()
        elif layer_type == "hybrid":
            # Hybrid CNN --> Transformer Encoder Layer
            self.cnn_transformer_layer = nn.TransformerEncoderLayer(
                d_model=gpt_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                activation="relu",
                norm_first=True,
            )
            padding = (cnn_kernel_size - 1) // 2  # To maintain sequence length
            self.cnn = nn.Conv1d(
                in_channels=gpt_dim,
                out_channels=gpt_dim,
                kernel_size=cnn_kernel_size,
                padding=padding,
            )
            self.norm = nn.LayerNorm(gpt_dim)
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported block_type: {layer_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, gpt_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, gpt_dim)
        """
        if self.layer_type == "transformer":
            return self.transformer_layer(x)
        elif self.layer_type == "cnn":
            # CNN expects input of shape (batch_size, gpt_dim, seq_length)
            x_perm = x.permute(0, 2, 1)
            x_conv = self.cnn(x_perm)
            x_conv = x_conv.permute(
                0, 2, 1
            )  # Back to (batch_size, seq_length, gpt_dim)
            x_norm = self.norm(x + x_conv)  # Residual connection
            return self.activation(x_norm)
        elif self.layer_type == "hybrid":
            # First pass through Transformer Encoder Layer
            x_transformed = self.cnn_transformer_layer(x)
            # Then pass through CNN
            x_perm = x_transformed.permute(0, 2, 1)
            x_conv = self.cnn(x_perm)
            x_conv = x_conv.permute(
                0, 2, 1
            )  # Back to (batch_size, seq_length, gpt_dim)
            x_norm = self.norm(x_transformed + x_conv)  # Residual connection
            return self.activation(x_norm)


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
        layer_type: Literal["transformer", "cnn", "hybrid"] = "transformer",
        num_layers: int = 8,
    ) -> None:
        """
        Args:
            embed_dim (int): The dimensionality of the image embedding vector.
            gpt_dim (int): The dimensionality of the GPT token embeddings.
            prefix_length (int): The number of prefix tokens to generate.
            hidden_length (int): The number of tokens to project the image embedding vector into.
            layer_type (Literal["transformer", "cnn", "hybrid"], optional): The type of encoder layer to use in the Transformer Encoder.
                Can be "transformer" for pure Transformer Encoder layers, "cnn" for CNN based layers, or "hybrid" for CNN with Transformer layers.
                Defaults to "transformer".
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
        encoder_layer = EncoderLayer(gpt_dim=gpt_dim, layer_type=layer_type)

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
        prefix_length: int | None = None,
        gpt: GPT2LMHeadModel | None = None,
        freeze_gpt_weights: bool = True,
    ) -> None:
        """
        Args:
            mapping_network (nn.Module): Initialized mapping network that maps image embeddings to GPT prefix tokens.
            prefix_length (int, optional): The number of prefix tokens (should match the mapping network's prefix length).
                Defaults to None, in which case it is inferred from the mapping network's `prefix_length` attribute.
            gpt (GPT2LMHeadModel | None, optional): Pretrained GPT-2 model. Defaults to None, in which case the standard GPT-2 model is loaded.
            freeze_gpt_weights (bool, optional): Explicitly freezes GPT-2 weights if True. Otherwise, will explicitly unfreeze them if False.
                Defaults to True.
        """
        # TODO: Have Mapping Network be an identifiable subclass

        super().__init__()
        self.prefix_length = prefix_length or mapping_network.prefix_length
        self.mapping_network = mapping_network

        # Load GPT-2
        self.gpt = gpt or GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        # Explicitly freeze or unfreeze weights for GPT-2
        for param in self.gpt.parameters():
            param.requires_grad = not freeze_gpt_weights

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

        # Obtain the prefix tokens from the image embeddings via the mapping network
        prefix_tokens = self.mapping_network(image_embeddings)

        # Concatenate prefix tokens with the ground-truth caption tokens
        input_tokens = torch.cat((prefix_tokens, caption_tokens), dim=1)

        # Handle Labels (labels simply denote which tokens to compute loss on)
        if labels is not None:
            # Add dummy labels for the prefix tokens so that we don't train on them
            # We want to compute loss only on the caption tokens, GPT-2 should not be trained to predict the prefix tokens
            # NOTE: The mapping network is still trained to produce good prefix tokens via loss/error signal from the generated caption tokens
            dummy_labels = (
                torch.zeros(
                    labels.shape[0],  # batch size
                    self.prefix_length,  # number of prefix tokens
                    dtype=torch.int64,
                    device=caption_token_ids.device,
                )
                - 100
            )  # -100 is the standard PyTorch ignore_index for CrossEntropyLoss
            labels = torch.cat((dummy_labels, labels), dim=1)

        # Handle Attention Mask
        if attention_mask is not None:
            # Add attention mask of 1s for the prefix tokens, ensuring they are attended to
            dummy_attention_mask = torch.ones(
                attention_mask.shape[0],  # batch size
                self.prefix_length,  # number of prefix tokens
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
        stop_token_id: int = 50256,
    ) -> torch.Tensor:
        """
        Generate captions autoregressively given image embeddings.

        Args:
            image_embeddings (torch.Tensor): Image embeddings of shape (batch_size, embed_dim)
            max_length (int, optional): Maximum length of the generated caption. Defaults to 50.
            temperature (float, optional): Sampling temperature for controlling randomness. Defaults to 1.0.
            top_p (float, optional): Nucleus sampling probability threshold. Defaults to 0.9.
            stop_token_id (int, optional): Token ID at which to stop generation. Defaults to 50256 (GPT-2's <|endoftext|> token ID).

        Returns:
            torch.Tensor: Generated caption tokens of shape (batch_size, generated_length)
        """

        # Set to evaluation mode
        self.eval()  # Will recursively set mapping network and GPT to eval mode

        # Get device and batch size
        device = image_embeddings.device
        batch_size = image_embeddings.shape[0]

        # Process image embeddings through the mapping network to get prefix tokens
        with torch.no_grad():
            prefix_tokens = self.mapping_network.forward(
                image_embeddings
            )  # (batch_size, prefix_length, gpt_dim)

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

                # If a sequence is finished, force its next token to be 0 (padding) to avoid generating nonsense tokens while preserving batch structure
                next_token_id[is_finished] = 0

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
        return torch.cat(generated_tokens, dim=1)  # (batch_size, generated_length)

class RetrievalAugmentedTransformer(ImageCaptioningModel):
    """"""
