# ========== Stage 3: Slice Aggregation – Scan-Level Representation ==========
import torch
import torch.nn as nn
import warnings # Import warnings
import os # Import os for path manipulation in example

# --- Import configurations from previous stages (assuming they exist) ---
try:
    from .stage1_data_acquisition import BATCH_SIZE, NUM_SLICES_PER_SCAN
    # We need the feature dim from Stage 2 to initialize the aggregator correctly
    from .stage2_vision_encoder import VisionEncoder # Import class to get dim
    temp_vision_encoder = VisionEncoder() # Instantiate to get feature_dim
    STAGE2_VISION_FEATURE_DIM = temp_vision_encoder.feature_dim
    del temp_vision_encoder # Clean up temporary instance
except ImportError:
    warnings.warn("Could not import configurations from stage1/stage2. Using placeholder values for Stage 3.")
    BATCH_SIZE = 4 # Placeholder
    NUM_SLICES_PER_SCAN = 64 # Placeholder
    STAGE2_VISION_FEATURE_DIM = 768 # Placeholder (e.g., ViT-Base)
except Exception as e:
     warnings.warn(f"Error importing/deriving config from stage1/stage2: {e}. Using placeholder values.")
     BATCH_SIZE = 4 # Placeholder
     NUM_SLICES_PER_SCAN = 64 # Placeholder
     STAGE2_VISION_FEATURE_DIM = 768 # Placeholder (e.g., ViT-Base)


# --- Configuration for Stage 3 ---
VISION_FEATURE_DIM = STAGE2_VISION_FEATURE_DIM
AGGREGATION_METHOD = 'mean' # Options: 'mean', 'lstm', 'gru', 'transformer'
# Hidden dimension for RNNs or Transformer feedforward layer
AGGREGATOR_HIDDEN_DIM = 512 # Relevant only if method is 'lstm', 'gru', or 'transformer'
TRANSFORMER_NHEAD = 8 # Number of attention heads for Transformer method
TRANSFORMER_NLAYERS = 1 # Number of layers for Transformer method
TRANSFORMER_DROPOUT = 0.1 # Dropout for transformer layers

# --- Slice Aggregator Class ---
class SliceAggregator(nn.Module):
    """
    Aggregates slice-level features into a single scan-level embedding.
    Supports mean pooling, LSTM, GRU, or Transformer Encoder aggregation.
    """
    def __init__(self, input_dim=VISION_FEATURE_DIM, method=AGGREGATION_METHOD,
                 hidden_dim=AGGREGATOR_HIDDEN_DIM, nhead=TRANSFORMER_NHEAD,
                 nlayers=TRANSFORMER_NLAYERS, dropout=TRANSFORMER_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.method = method
        self.hidden_dim = hidden_dim

        if self.method == 'mean':
            self.aggregator = None # Mean pooling is parameter-free
            self.output_dim = self.input_dim
        elif self.method == 'lstm':
            # batch_first=True expects input shape (Batch, Seq, Feature)
            self.aggregator = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False) # Or bidirectional=True
            self.output_dim = hidden_dim * (2 if self.aggregator.bidirectional else 1)
        elif self.method == 'gru':
            self.aggregator = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=False) # Or bidirectional=True
            self.output_dim = hidden_dim * (2 if self.aggregator.bidirectional else 1)
        elif self.method == 'transformer':
            # A simple Transformer Encoder stack
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim, # Or use input_dim*4 as common practice
                batch_first=True,
                dropout=dropout # Use dropout parameter
            )
            self.aggregator = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
            # Output dim remains the same as input for Transformer Encoder layers
            self.output_dim = input_dim
            # Optional: Add a learnable [CLS] token equivalent for aggregation
            # self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim)) # Example CLS token
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")

        print(f"Initialized Slice Aggregator with method '{self.method}'. Output dimension: {self.output_dim}")


    def forward(self, slice_features):
        """
        Args:
            slice_features (torch.Tensor): Features for slices in a batch (BatchSize, NumSlices, FeatureDimension)
        Returns:
            torch.Tensor: Aggregated scan-level embedding (BatchSize, AggregatedFeatureDimension)
        """
        if self.aggregator is None and self.method == 'mean':
            # Compute mean across the slice dimension (dim=1)
            scan_embedding = slice_features.mean(dim=1) # Shape: (BatchSize, FeatureDimension)
        elif isinstance(self.aggregator, (nn.LSTM, nn.GRU)):
            # Pass features through RNN
            outputs, hidden = self.aggregator(slice_features)

            # Extract the final hidden state (last layer's hidden state)
            if isinstance(hidden, tuple): # LSTM
                h_n = hidden[0] # Shape: (NumLayers * NumDirections, Batch, HiddenSize)
            else: # GRU
                h_n = hidden # Shape: (NumLayers * NumDirections, Batch, HiddenSize)

            # Handle bidirectionality
            if self.aggregator.bidirectional:
                 # Concatenate final forward and backward hidden states from the last layer
                 # hidden has shape (num_layers * num_directions, batch, hidden_size)
                 # Last layer forward: index -2, Last layer backward: index -1
                 scan_embedding = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # Shape: (Batch, HiddenDim*2)
            else:
                 # Take the hidden state of the last layer [-1]
                 scan_embedding = h_n[-1] # Shape: (Batch, HiddenDim)

        elif isinstance(self.aggregator, nn.TransformerEncoder):
            # Optional: Prepend a CLS token if defined
            # if hasattr(self, 'cls_token'):
            #     batch_size = slice_features.shape[0]
            #     cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            #     slice_features = torch.cat((cls_tokens, slice_features), dim=1) # Shape (B, N+1, D)

            # Pass through Transformer Encoder
            transformer_output = self.aggregator(slice_features) # Shape: (Batch, NumSlices(+1 if CLS), FeatureDim)

            # Aggregate the output sequence
            # If using CLS token: take the first token's output
            # if hasattr(self, 'cls_token'):
            #     scan_embedding = transformer_output[:, 0] # Shape: (Batch, FeatureDim)
            # else: # Mean pool the output sequence (common alternative if no CLS token)
            scan_embedding = transformer_output.mean(dim=1) # Shape: (Batch, FeatureDim)

        else:
             # Should not happen due to init check, but good practice
             raise ValueError(f"Aggregation logic not implemented for method: {self.method}")

        return scan_embedding

# --- Determine Output Dimension Based on Config ---
# This makes the output dimension available for import by Stage 4
# We instantiate a temporary aggregator to get its configured output dim
try:
    temp_aggregator = SliceAggregator(
        input_dim=VISION_FEATURE_DIM,
        method=AGGREGATION_METHOD,
        hidden_dim=AGGREGATOR_HIDDEN_DIM,
        nhead=TRANSFORMER_NHEAD,
        nlayers=TRANSFORMER_NLAYERS,
        dropout=TRANSFORMER_DROPOUT
    )
    AGGREGATOR_OUTPUT_DIM = temp_aggregator.output_dim
    print(f"Stage 3 Aggregator Output Dimension: {AGGREGATOR_OUTPUT_DIM}")
    del temp_aggregator # Clean up
except Exception as e:
     warnings.warn(f"Could not determine AGGREGATOR_OUTPUT_DIM dynamically: {e}. Using placeholder.")
     AGGREGATOR_OUTPUT_DIM = AGGREGATOR_HIDDEN_DIM if AGGREGATION_METHOD in ['lstm', 'gru'] else VISION_FEATURE_DIM


# --- Example Usage ---
if __name__ == "__main__": # Ensures this runs only when script is executed directly
    print("--- Stage 3 Example ---")

    # Assume slice_embeddings from Stage 2 exists (B, N, FeatureDim)
    # Create dummy data using the potentially imported or placeholder values
    dummy_slice_embeddings = torch.randn(BATCH_SIZE, NUM_SLICES_PER_SCAN, VISION_FEATURE_DIM)

    # Instantiate the aggregator
    try:
        # Try different methods by changing AGGREGATION_METHOD: 'mean', 'lstm', 'gru', 'transformer'
        slice_aggregator = SliceAggregator(
            input_dim=VISION_FEATURE_DIM,
            method=AGGREGATION_METHOD,
            hidden_dim=AGGREGATOR_HIDDEN_DIM,
            nhead=TRANSFORMER_NHEAD,
            nlayers=TRANSFORMER_NLAYERS,
            dropout=TRANSFORMER_DROPOUT
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        slice_aggregator.to(device)
        dummy_slice_embeddings = dummy_slice_embeddings.to(device)
        print(f"Running example on device: {device}")
        print(f"Input dummy embeddings shape: {dummy_slice_embeddings.shape}")


        # Perform aggregation
        slice_aggregator.eval() # Set to evaluation mode for inference
        with torch.no_grad(): # Disable gradient calculation
            scan_level_embedding = slice_aggregator(dummy_slice_embeddings)

        print(f"Slice aggregator ({AGGREGATION_METHOD}) output shape: {scan_level_embedding.shape}")
        # Expected shape: (BatchSize, AggregatedFeatureDimension) where AggregatedFeatureDimension depends on method
        assert scan_level_embedding.shape[0] == BATCH_SIZE
        # Use the dynamically determined output dim for assertion
        assert scan_level_embedding.shape[1] == AGGREGATOR_OUTPUT_DIM

    except Exception as e:
        print(f"\nError during slice aggregator test: {e}")
        import traceback
        traceback.print_exc()

    print("\nStage 3: Slice aggregation setup complete.\n")
