# ========== Stage 4: Bridging Vision and Language ==========
import torch
import torch.nn as nn
import warnings
import os

# --- Import configurations from previous stages ---
try:
    # Assuming these files exist and define the variables at the module level
    # Note: It's generally better to use a dedicated config system.
    from .stage1_data_acquisition import BATCH_SIZE # Needed for example usage
    from .stage3_slice_aggregation import AGGREGATOR_OUTPUT_DIM as STAGE3_AGGREGATOR_OUTPUT_DIM
    # If Stage 3 uses a different variable name for output dimension, adjust accordingly
    # For example, if it was just `slice_aggregator.output_dim` in the example,
    # we might need to instantiate it or derive the dim based on its config.
    # Let's assume Stage 3 defines a constant or the variable is accessible.
except ImportError:
    warnings.warn("Could not import configurations from previous stages. Using placeholder values for Stage 4.")
    BATCH_SIZE = 4 # Placeholder
    # Placeholder - MUST match the actual output dim of the chosen aggregator in Stage 3
    STAGE3_AGGREGATOR_OUTPUT_DIM = 768 # Example (e.g., if Stage 3 uses 'mean' or 'transformer' with ViT-Base features)

# --- Configuration for Stage 4 ---
# Dimension of the scan-level embedding from Stage 3
AGGREGATED_FEATURE_DIM = STAGE3_AGGREGATOR_OUTPUT_DIM
# Target dimension required by the language model (e.g., its hidden size/embedding dim)
# This depends on the specific language model chosen in Stage 5
LANGUAGE_MODEL_DIM = 768 # Example for T5-base or BioGPT-base (Adjust as needed)

# --- Bridge Class (Conceptual Projection) ---
class VisionLanguageBridge(nn.Module):
    """
    A conceptual bridge, often a simple linear layer, to project
    the aggregated visual features to the dimension expected by the language model.
    The actual cross-attention happens *within* the language decoder.
    """
    def __init__(self, visual_dim=AGGREGATED_FEATURE_DIM, language_dim=LANGUAGE_MODEL_DIM):
        super().__init__()
        self.visual_dim = visual_dim
        self.language_dim = language_dim
        # Simple linear projection layer
        self.projection = nn.Linear(self.visual_dim, self.language_dim)
        # Optional: Add Layer Normalization or activation functions if needed
        # self.layer_norm = nn.LayerNorm(self.language_dim)
        # self.activation = nn.ReLU()
        print(f"Initialized VisionLanguageBridge: Projecting {visual_dim} -> {language_dim}")

    def forward(self, scan_embedding):
        """
        Args:
            scan_embedding (torch.Tensor): Scan-level visual embedding (BatchSize, AggregatedFeatureDimension)
        Returns:
            torch.Tensor: Conditioned visual embedding ready for the language model (BatchSize, TargetLanguageModelDimension)
                         This output is often called encoder_hidden_states or similar for cross-attention models.
        """
        projected_embedding = self.projection(scan_embedding)
        # projected_embedding = self.layer_norm(projected_embedding) # Optional
        # projected_embedding = self.activation(projected_embedding) # Optional

        # Note: Depending on the LM, you might need to unsqueeze the time/sequence dimension
        # e.g., if the LM expects (BatchSize, SequenceLength, Dim) for encoder hidden states.
        # If the projected embedding is used as a single context vector, this might be it.
        # If it's meant to be attended to over its "sequence" (even if length 1), unsqueeze might be needed.
        # Example: return projected_embedding.unsqueeze(1) # -> (BatchSize, 1, TargetLanguageModelDimension)
        # For now, returning (BatchSize, TargetLanguageModelDimension) as the base projection.
        return projected_embedding

# --- Example Usage ---
if __name__ == "__main__": # Ensures this runs only when script is executed directly
    print("--- Stage 4 Example ---")
    # Assume scan_level_embedding from Stage 3 exists (B, AggregatedFeatureDim)
    # Or create dummy data:
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

    except Exception as e:
        print(f"\nError during bridge test: {e}")
        import traceback
        traceback.print_exc()

    print("\nStage 4: Vision-Language bridge (projection) setup complete.")
    print("Note: Actual cross-attention or conditioning occurs within the Stage 5 Language Model.")
