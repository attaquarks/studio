# ========== Stage 4: Bridging Vision and Language ==========
import torch
import torch.nn as nn
import warnings
import os

# --- Import configurations from previous stages ---
try:
    # Use absolute import if stages are in the same package, relative if scripts
    # Assuming execution from a parent directory or package structure:
    # from stage1_data_acquisition import BATCH_SIZE # Needed for example usage
    # from stage3_slice_aggregation import AGGREGATOR_OUTPUT_DIM as STAGE3_AGGREGATOR_OUTPUT_DIM
    # If running as standalone scripts, relative import might work if in the same folder:
    from .stage1_data_acquisition import BATCH_SIZE # Needed for example usage
    from .stage3_slice_aggregation import AGGREGATOR_OUTPUT_DIM as STAGE3_AGGREGATOR_OUTPUT_DIM
    AGGREGATED_FEATURE_DIM = STAGE3_AGGREGATOR_OUTPUT_DIM
    print(f"Successfully derived AGGREGATED_FEATURE_DIM from Stage 3: {AGGREGATED_FEATURE_DIM}")

except ImportError as e_imp:
    warnings.warn(f"Could not import configurations from previous stages ({e_imp}). Using placeholder values for Stage 4.")
    BATCH_SIZE = 4 # Placeholder
    # Placeholder - MUST match the actual output dim of the chosen aggregator in Stage 3
    AGGREGATED_FEATURE_DIM = 768 # Example (e.g., if Stage 3 uses 'mean' or 'transformer' with ViT-Base features)
except NameError as e_name:
     warnings.warn(f"AGGREGATOR_OUTPUT_DIM not found in stage3 ({e_name}). Check stage3 definition. Using placeholder.")
     BATCH_SIZE = 4 # Placeholder
     AGGREGATED_FEATURE_DIM = 768
except Exception as e_other:
     warnings.warn(f"Error importing config from previous stages ({e_other}). Using placeholder values.")
     BATCH_SIZE = 4 # Placeholder
     AGGREGATED_FEATURE_DIM = 768


# --- Configuration for Stage 4 ---
# Dimension of the scan-level embedding from Stage 3 (derived above)
# AGGREGATED_FEATURE_DIM = STAGE3_AGGREGATOR_OUTPUT_DIM

# Target dimension required by the language model (e.g., its hidden size/embedding dim)
# This depends on the specific language model chosen in Stage 5
# IMPORTANT: UPDATE THIS VALUE BASED ON THE LM USED IN STAGE 5
LANGUAGE_MODEL_DIM = 768 # Example for T5-base or BioGPT-base (Adjust as needed)

# --- Bridge Class (Conceptual Projection) ---
class VisionLanguageBridge(nn.Module):
    """
    A conceptual bridge, often a simple linear layer, to project
    the aggregated visual features to the dimension expected by the language model.
    The actual cross-attention happens *within* the language decoder (Stage 5).
    """
    def __init__(self, visual_dim=AGGREGATED_FEATURE_DIM, language_dim=LANGUAGE_MODEL_DIM):
        super().__init__()
        self.visual_dim = visual_dim
        self.language_dim = language_dim

        if self.visual_dim <= 0:
             raise ValueError(f"Visual dimension must be positive, got {self.visual_dim}. Check Stage 3 output.")
        if self.language_dim <= 0:
             raise ValueError(f"Language dimension must be positive, got {self.language_dim}. Check Stage 5 model config.")

        # Simple linear projection layer
        self.projection = nn.Linear(self.visual_dim, self.language_dim)
        # Optional: Add Layer Normalization or activation functions if needed
        # self.layer_norm = nn.LayerNorm(self.language_dim)
        # self.activation = nn.ReLU()
        print(f"Initialized VisionLanguageBridge: Projecting {visual_dim} -> {language_dim}")

    def forward(self, scan_embedding):
        """
        Args:
            scan_embedding (torch.Tensor): Scan-level visual embedding
                                          Shape: (BatchSize, AggregatedFeatureDimension)
        Returns:
            torch.Tensor: Conditioned visual embedding ready for the language model.
                          Shape: (BatchSize, TargetLanguageModelDimension)
                          This output is often used as `encoder_hidden_states` in seq2seq models
                          or needs further processing for causal LMs.
        """
        # Input validation
        if scan_embedding.shape[-1] != self.visual_dim:
            raise ValueError(f"Input embedding dim ({scan_embedding.shape[-1]}) does not match bridge visual_dim ({self.visual_dim})")

        projected_embedding = self.projection(scan_embedding)
        # projected_embedding = self.layer_norm(projected_embedding) # Optional
        # projected_embedding = self.activation(projected_embedding) # Optional

        # Note: For many seq2seq models (like T5, BART), the output of this bridge
        # might need to be unsqueezed to add a sequence length dimension of 1
        # when passed as `encoder_outputs` or `encoder_hidden_states` to the `generate` method.
        # Example: return projected_embedding.unsqueeze(1) # -> (BatchSize, 1, TargetLanguageModelDimension)
        # However, the `generate` function in Stage 5 will handle this unsqueezing if needed.
        # We return the base projection here.
        return projected_embedding

# --- Example Usage ---
if __name__ == "__main__": # Ensures this runs only when script is executed directly
    print("--- Stage 4 Example ---")
    # Assume scan_level_embedding from Stage 3 exists (B, AggregatedFeatureDim)
    # Or create dummy data:
    # Use the AGGREGATED_FEATURE_DIM derived or set as placeholder
    dummy_scan_level_embedding = torch.randn(BATCH_SIZE, AGGREGATED_FEATURE_DIM)
    print(f"Input dummy scan embedding shape: {dummy_scan_level_embedding.shape}")

    # Instantiate the bridge
    try:
        bridge = VisionLanguageBridge(visual_dim=AGGREGATED_FEATURE_DIM, language_dim=LANGUAGE_MODEL_DIM)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bridge.to(device)
        dummy_scan_level_embedding = dummy_scan_level_embedding.to(device)
        print(f"Running example on device: {device}")


        # Perform projection
        bridge.eval()
        with torch.no_grad():
            conditioned_visual_embedding = bridge(dummy_scan_level_embedding)

        print(f"Bridge output shape: {conditioned_visual_embedding.shape}")
        # Expected: (BatchSize, TargetLanguageModelDimension)
        assert conditioned_visual_embedding.shape[0] == BATCH_SIZE
        assert conditioned_visual_embedding.shape[1] == LANGUAGE_MODEL_DIM

    except ImportError as ie:
         print(f"\nImportError: {ie}. Make sure necessary libraries (torch) are installed and previous stages are accessible.")
    except Exception as e:
        print(f"\nError during bridge test: {e}")
        import traceback
        traceback.print_exc()

    print("\nStage 4: Vision-Language bridge (projection) setup complete.")
    print("Note: Actual cross-attention or conditioning occurs within the Stage 5 Language Model.")
