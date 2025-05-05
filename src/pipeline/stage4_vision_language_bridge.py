# ========== Stage 4: Bridging Vision and Language - Cross-Attention ==========
import torch
import torch.nn as nn
import warnings
import os

# --- Import configurations from previous stages ---
try:
    from .stage1_data_acquisition import BATCH_SIZE
    # Get the output dimension of the aggregator from Stage 3
    from .stage3_slice_aggregation import AGGREGATOR_OUTPUT_DIM as STAGE3_AGGREGATOR_OUTPUT_DIM
    AGGREGATED_FEATURE_DIM = STAGE3_AGGREGATOR_OUTPUT_DIM
    print(f"Successfully derived AGGREGATED_FEATURE_DIM from Stage 3: {AGGREGATED_FEATURE_DIM}")

    # We also need the expected language model dimension for the bridge output
    # This will be defined in Stage 5, so we might use a placeholder here or import it
    # Let's assume it will be defined in Stage 5 and use a placeholder for now,
    # or attempt a risky import if structure allows.
    try:
        from .stage5_language_decoder import LANGUAGE_MODEL_DIM as STAGE5_TARGET_DIM
        TARGET_LANGUAGE_MODEL_DIM = STAGE5_TARGET_DIM
        print(f"Successfully derived TARGET_LANGUAGE_MODEL_DIM from Stage 5: {TARGET_LANGUAGE_MODEL_DIM}")
    except (ImportError, NameError):
         warnings.warn("Could not import LANGUAGE_MODEL_DIM from stage5. Using placeholder 768 for Stage 4 bridge target.")
         TARGET_LANGUAGE_MODEL_DIM = 768 # Placeholder (e.g., T5-base, BioGPT-base)

except ImportError as e_imp:
    warnings.warn(f"Could not import configurations from previous stages ({e_imp}). Using placeholder values for Stage 4.")
    BATCH_SIZE = 4
    AGGREGATED_FEATURE_DIM = 768 # Placeholder, MUST align with actual Stage 3 output
    TARGET_LANGUAGE_MODEL_DIM = 768 # Placeholder, MUST align with actual Stage 5 input
except NameError as e_name:
     warnings.warn(f"AGGREGATOR_OUTPUT_DIM not found in stage3 ({e_name}). Check stage3 definition. Using placeholder 768.")
     BATCH_SIZE = 4
     AGGREGATED_FEATURE_DIM = 768
     TARGET_LANGUAGE_MODEL_DIM = 768
except Exception as e_other:
     warnings.warn(f"Error importing config from previous stages ({e_other}). Using placeholder values.")
     BATCH_SIZE = 4
     AGGREGATED_FEATURE_DIM = 768
     TARGET_LANGUAGE_MODEL_DIM = 768

# --- Configuration for Stage 4 ---
# Vision Dim (from Stage 3), Language Dim (for Stage 5) are derived above
BRIDGE_HIDDEN_DIM = 512 # Hidden dimension within the bridge layers
BRIDGE_NUM_HEADS = 8 # Number of attention heads for cross-attention
BRIDGE_DROPOUT = 0.1 # Dropout rate

# --- Bridge Class (Cross-Attention) ---
class CrossAttentionBridge(nn.Module):
    """Bridge between visual features and language model using cross-attention."""

    def __init__(self,
                 vision_dim: int = AGGREGATED_FEATURE_DIM,
                 language_dim: int = TARGET_LANGUAGE_MODEL_DIM,
                 bridge_dim: int = BRIDGE_HIDDEN_DIM,
                 num_heads: int = BRIDGE_NUM_HEADS,
                 dropout: float = BRIDGE_DROPOUT):
        """
        Initialize the cross-attention bridge.

        Args:
            vision_dim: Dimension of the vision features (output of Stage 3)
            language_dim: Dimension of the language model embeddings (input/output of Stage 5)
            bridge_dim: Hidden dimension for the bridge projection/attention layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        if vision_dim <= 0 or language_dim <= 0 or bridge_dim <= 0:
             raise ValueError("Dimensions (vision, language, bridge) must be positive.")
        if bridge_dim % num_heads != 0:
             # Adjust num_heads or raise error
             original_nhead = num_heads
             while bridge_dim % num_heads != 0 and num_heads > 1: num_heads -= 1
             if bridge_dim % num_heads == 0:
                 warnings.warn(f"Bridge Attention: Adjusted num_heads from {original_nhead} to {num_heads} to be divisible by bridge_dim {bridge_dim}.")
             else:
                 raise ValueError(f"bridge_dim ({bridge_dim}) must be divisible by num_heads ({original_nhead}).")

        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.bridge_dim = bridge_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        # Project vision features to bridge dimension
        self.vision_proj = nn.Linear(vision_dim, bridge_dim)

        # Project language features to bridge dimension
        self.language_proj = nn.Linear(language_dim, bridge_dim)

        # Cross-attention layer (Language Query attends to Vision Key/Value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=bridge_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True # Input shape (B, SeqLen, Dim)
        )

        # Output projection back to language dimension
        self.output_proj = nn.Linear(bridge_dim, language_dim)

        # Layer normalization
        self.norm1_vis = nn.LayerNorm(bridge_dim)
        self.norm1_lang = nn.LayerNorm(bridge_dim)
        self.norm2 = nn.LayerNorm(bridge_dim) # Norm after attention + residual
        self.norm3 = nn.LayerNorm(language_dim) # Final norm

        # Feed-forward network (applied after cross-attention)
        self.ffn = nn.Sequential(
            nn.Linear(bridge_dim, bridge_dim * 4), # Expansion layer
            nn.GELU(),                             # Activation
            nn.Dropout(dropout),                   # Dropout after activation
            nn.Linear(bridge_dim * 4, bridge_dim)   # Projection back
        )

        self.dropout = nn.Dropout(dropout)

        print(f"Initialized CrossAttentionBridge: VisionDim={vision_dim}, LangDim={language_dim}, BridgeDim={bridge_dim}, Heads={num_heads}")

    def forward(self, vision_features, language_features, language_attention_mask=None):
        """
        Forward pass through the cross-attention bridge.

        Args:
            vision_features: Aggregated vision features. Shape: [B, D_v]
            language_features: Language features (e.g., token embeddings or hidden states).
                               Shape: [B, L, D_l] (L = sequence length)
            language_attention_mask: Optional mask for language sequence padding.
                                     Shape: [B, L]. True indicates masked positions.

        Returns:
            torch.Tensor: Updated language features incorporating visual information.
                          Shape: [B, L, D_l]
        """
        batch_size, seq_len, _ = language_features.shape

        # 1. Project features to bridge dimension
        # Vision features need sequence dimension: [B, 1, D_v] -> [B, 1, D_b]
        vision_proj = self.vision_proj(vision_features.unsqueeze(1)) # Shape: [B, 1, D_b]

        # Language features: [B, L, D_l] -> [B, L, D_b]
        language_proj = self.language_proj(language_features) # Shape: [B, L, D_b]

        # 2. Apply Layer Normalization (before attention)
        vision_qkv = self.norm1_vis(vision_proj)    # Key/Value from vision [B, 1, D_b]
        language_q = self.norm1_lang(language_proj) # Query from language [B, L, D_b]

        # 3. Compute cross-attention
        # Query: language_q [B, L, D_b]
        # Key:   vision_qkv [B, 1, D_b]
        # Value: vision_qkv [B, 1, D_b]
        # key_padding_mask: We don't typically mask the single visual feature.
        # attn_mask: Use language_attention_mask if provided to prevent attention to padded language tokens.
        attn_output, attn_weights = self.cross_attention(
            query=language_q,
            key=vision_qkv,
            value=vision_qkv,
            key_padding_mask=None, # No padding in the single visual feature key/value
            attn_mask=None # Standard attention mask usage if needed, but cross-attn might not need this specific mask type directly
        )
        # attn_output shape: [B, L, D_b]

        # 4. Residual connection and Layer Normalization after attention
        # Add attention output to the original *projected* language features
        attn_res = language_proj + self.dropout(attn_output)
        attn_norm = self.norm2(attn_res)

        # 5. Feed-forward network
        ffn_output = self.ffn(attn_norm)

        # 6. Residual connection and Layer Normalization after FFN
        ffn_res = attn_norm + self.dropout(ffn_output)
        # LayerNorm is often applied *before* the final projection in some architectures,
        # but applying after residual is also common. Let's stick to norm before projection.
        # ffn_norm = self.norm_ffn(ffn_res) # Add another norm if desired

        # 7. Project back to language dimension
        output = self.output_proj(ffn_res) # Shape: [B, L, D_l]

        # 8. Final Layer Normalization (optional, depends on decoder expectations)
        # This could interfere if the decoder expects raw outputs before its own norm.
        # Let's apply it here for consistency with the provided code structure.
        output = self.norm3(output)

        return output


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Stage 4 Example ---")
    # Assume scan_level_embedding (visual_features) from Stage 3 exists (B, D_v)
    # Assume language_features (e.g., token embeddings) exist (B, L, D_l)

    # Create dummy data using derived/placeholder dimensions
    dummy_visual_features = torch.randn(BATCH_SIZE, AGGREGATED_FEATURE_DIM)
    dummy_seq_len = 20 # Example sequence length for language part
    dummy_language_features = torch.randn(BATCH_SIZE, dummy_seq_len, TARGET_LANGUAGE_MODEL_DIM)
    # Optional dummy mask for language padding
    dummy_lang_mask = torch.zeros(BATCH_SIZE, dummy_seq_len, dtype=torch.bool)
    # dummy_lang_mask[:, -5:] = True # Example: Mask last 5 language tokens

    print(f"Input dummy visual features shape: {dummy_visual_features.shape}")
    print(f"Input dummy language features shape: {dummy_language_features.shape}")

    # Instantiate the bridge
    try:
        bridge = CrossAttentionBridge(
            vision_dim=AGGREGATED_FEATURE_DIM,
            language_dim=TARGET_LANGUAGE_MODEL_DIM,
            bridge_dim=BRIDGE_HIDDEN_DIM,
            num_heads=BRIDGE_NUM_HEADS,
            dropout=BRIDGE_DROPOUT
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bridge.to(device)
        dummy_visual_features = dummy_visual_features.to(device)
        dummy_language_features = dummy_language_features.to(device)
        # dummy_lang_mask = dummy_lang_mask.to(device) # Move mask to device if using
        print(f"Running example on device: {device}")

        # Perform cross-attention bridging
        bridge.eval()
        with torch.no_grad():
            # Pass mask only if needed by attention implementation
            updated_language_features = bridge(
                dummy_visual_features,
                dummy_language_features,
                language_attention_mask=None # Pass dummy_lang_mask if attention needs it
            )

        print(f"Bridge output shape (updated language features): {updated_language_features.shape}")
        # Expected: (BatchSize, SeqLen, TargetLanguageModelDimension)
        assert updated_language_features.shape == dummy_language_features.shape

    except ImportError as ie:
         print(f"\nImportError: {ie}. Make sure previous stages are accessible.")
    except Exception as e:
        print(f"\nError during cross-attention bridge example: {e}")
        import traceback
        traceback.print_exc()

    print("\nStage 4: Vision-Language bridge (Cross-Attention) setup complete.")
