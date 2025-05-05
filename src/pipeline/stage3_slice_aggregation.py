# ========== Stage 3: Slice Aggregation – Scan-Level Representation ==========
import torch
import torch.nn as nn
import warnings # Import warnings
import os # Import os for path manipulation in example

# --- Import configurations from previous stages (assuming they exist) ---
try:
    # Use absolute import if stages are in the same package, relative if scripts
    # Assuming execution from a parent directory or package structure:
    # from stage1_data_acquisition import BATCH_SIZE, NUM_SLICES_PER_SCAN
    # from stage2_vision_encoder import VisionEncoder # Import class to get dim
    # If running as standalone scripts, relative import might work if in the same folder:
     from .stage1_data_acquisition import BATCH_SIZE, NUM_SLICES_PER_SCAN
     # Dynamically get feature dim from Stage 2
     from .stage2_vision_encoder import VisionEncoder # Import class to get dim
     # Instantiate VisionEncoder to get its configured feature_dim
     # Use try-except block in case stage2 fails or has different structure
     try:
         temp_vision_encoder = VisionEncoder() # Uses default config from stage2
         STAGE2_VISION_FEATURE_DIM = temp_vision_encoder.feature_dim
         print(f"Successfully derived VISION_FEATURE_DIM from Stage 2: {STAGE2_VISION_FEATURE_DIM}")
         del temp_vision_encoder # Clean up temporary instance
     except Exception as e_vis:
         warnings.warn(f"Could not instantiate VisionEncoder from Stage 2 to get feature_dim: {e_vis}. Using placeholder value for Stage 3 VISION_FEATURE_DIM.")
         STAGE2_VISION_FEATURE_DIM = 768 # Placeholder (e.g., ViT-Base)

except ImportError as e_imp:
    warnings.warn(f"Could not import configurations from stage1/stage2 ({e_imp}). Using placeholder values for Stage 3.")
    BATCH_SIZE = 4 # Placeholder
    NUM_SLICES_PER_SCAN = 64 # Placeholder
    STAGE2_VISION_FEATURE_DIM = 768 # Placeholder (e.g., ViT-Base)
except Exception as e_other:
     warnings.warn(f"Error importing/deriving config from stage1/stage2: {e_other}. Using placeholder values.")
     BATCH_SIZE = 4 # Placeholder
     NUM_SLICES_PER_SCAN = 64 # Placeholder
     STAGE2_VISION_FEATURE_DIM = 768 # Placeholder (e.g., ViT-Base)


# --- Configuration for Stage 3 ---
# Use the dimension derived from Stage 2
VISION_FEATURE_DIM = STAGE2_VISION_FEATURE_DIM
AGGREGATION_METHOD = 'mean' # Options: 'mean', 'lstm', 'gru', 'transformer'
# Hidden dimension for RNNs or Transformer feedforward layer
AGGREGATOR_HIDDEN_DIM = 512 # Relevant only if method is 'lstm', 'gru', or 'transformer'
RNN_BIDIRECTIONAL = False # Set to True to use bidirectional RNNs
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
                 hidden_dim=AGGREGATOR_HIDDEN_DIM, rnn_bidirectional=RNN_BIDIRECTIONAL,
                 nhead=TRANSFORMER_NHEAD, nlayers=TRANSFORMER_NLAYERS, dropout=TRANSFORMER_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.method = method
        self.hidden_dim = hidden_dim
        self.rnn_bidirectional = rnn_bidirectional

        if self.method == 'mean':
            self.aggregator = None # Mean pooling is parameter-free
            self.output_dim = self.input_dim
        elif self.method == 'lstm':
            # batch_first=True expects input shape (Batch, Seq, Feature)
            self.aggregator = nn.LSTM(
                input_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=self.rnn_bidirectional
            )
            self.output_dim = hidden_dim * (2 if self.rnn_bidirectional else 1)
        elif self.method == 'gru':
            self.aggregator = nn.GRU(
                input_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=self.rnn_bidirectional
            )
            self.output_dim = hidden_dim * (2 if self.rnn_bidirectional else 1)
        elif self.method == 'transformer':
            # A simple Transformer Encoder stack
            # Ensure input_dim is divisible by nhead
            if input_dim % nhead != 0:
                # Find the nearest valid head count or adjust input_dim if possible
                # For now, we raise an error or warning
                warnings.warn(f"input_dim ({input_dim}) is not divisible by nhead ({nhead}). Transformer may fail or perform poorly. Adjust config.")
                # Simple fix: adjust nhead down if possible
                valid_nhead = nhead
                while input_dim % valid_nhead != 0 and valid_nhead > 1:
                    valid_nhead -= 1
                if input_dim % valid_nhead == 0:
                     warnings.warn(f"Adjusting nhead to {valid_nhead} for compatibility.")
                     nhead = valid_nhead
                else:
                     raise ValueError(f"Cannot find valid nhead for input_dim {input_dim}. Original nhead was {nhead}.")


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
            # outputs shape: (Batch, Seq, NumDirections * HiddenSize)
            # hidden shape: (NumLayers * NumDirections, Batch, HiddenSize) for GRU
            # hidden shape: tuple ((h_n, c_n)) for LSTM, each (NumLayers * NumDirections, Batch, HiddenSize)
            outputs, hidden = self.aggregator(slice_features)

            # Extract the final hidden state (last layer's hidden state)
            if isinstance(hidden, tuple): # LSTM
                h_n = hidden[0] # Shape: (NumLayers * NumDirections, Batch, HiddenSize)
            else: # GRU
                h_n = hidden # Shape: (NumLayers * NumDirections, Batch, HiddenSize)

            # Handle bidirectionality
            if self.rnn_bidirectional:
                 # Concatenate final forward and backward hidden states from the last layer
                 # hidden has shape (num_layers * 2, batch, hidden_size)
                 # Last layer forward: index -2, Last layer backward: index -1
                 scan_embedding = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # Shape: (Batch, HiddenDim*2)
            else:
                 # Take the hidden state of the last layer [-1]
                 # hidden has shape (num_layers, batch, hidden_size)
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
        rnn_bidirectional=RNN_BIDIRECTIONAL,
        nhead=TRANSFORMER_NHEAD,
        nlayers=TRANSFORMER_NLAYERS,
        dropout=TRANSFORMER_DROPOUT
    )
    AGGREGATOR_OUTPUT_DIM = temp_aggregator.output_dim
    print(f"Stage 3 Aggregator Output Dimension determined: {AGGREGATOR_OUTPUT_DIM}")
    del temp_aggregator # Clean up
except ValueError as e_val: # Catch specific errors like nhead incompatibility
     warnings.warn(f"Could not determine AGGREGATOR_OUTPUT_DIM due to configuration error: {e_val}. Using fallback logic.")
     # Fallback logic based on method
     if AGGREGATION_METHOD == 'mean':
         AGGREGATOR_OUTPUT_DIM = VISION_FEATURE_DIM
     elif AGGREGATION_METHOD in ['lstm', 'gru']:
         AGGREGATOR_OUTPUT_DIM = AGGREGATOR_HIDDEN_DIM * (2 if RNN_BIDIRECTIONAL else 1)
     elif AGGREGATION_METHOD == 'transformer':
         AGGREGATOR_OUTPUT_DIM = VISION_FEATURE_DIM
     else:
         AGGREGATOR_OUTPUT_DIM = VISION_FEATURE_DIM # Default fallback
         warnings.warn(f"Unknown aggregation method '{AGGREGATION_METHOD}'. Defaulting AGGREGATOR_OUTPUT_DIM to VISION_FEATURE_DIM ({VISION_FEATURE_DIM}).")
     print(f"Stage 3 Aggregator Output Dimension (fallback): {AGGREGATOR_OUTPUT_DIM}")

except Exception as e:
     warnings.warn(f"Could not determine AGGREGATOR_OUTPUT_DIM dynamically: {e}. Using fallback logic.")
     # Fallback logic based on method
     if AGGREGATION_METHOD == 'mean':
         AGGREGATOR_OUTPUT_DIM = VISION_FEATURE_DIM
     elif AGGREGATION_METHOD in ['lstm', 'gru']:
         AGGREGATOR_OUTPUT_DIM = AGGREGATOR_HIDDEN_DIM * (2 if RNN_BIDIRECTIONAL else 1)
     elif AGGREGATION_METHOD == 'transformer':
         AGGREGATOR_OUTPUT_DIM = VISION_FEATURE_DIM
     else:
         AGGREGATOR_OUTPUT_DIM = VISION_FEATURE_DIM # Default fallback
         warnings.warn(f"Unknown aggregation method '{AGGREGATION_METHOD}'. Defaulting AGGREGATOR_OUTPUT_DIM to VISION_FEATURE_DIM ({VISION_FEATURE_DIM}).")
     print(f"Stage 3 Aggregator Output Dimension (fallback): {AGGREGATOR_OUTPUT_DIM}")


# --- Example Usage ---
if __name__ == "__main__": # Ensures this runs only when script is executed directly
    print("--- Stage 3 Example ---")

    # Assume slice_embeddings from Stage 2 exists (B, N, FeatureDim)
    # Create dummy data using the potentially imported or placeholder values
    # Use VISION_FEATURE_DIM which was derived or set as placeholder
    dummy_slice_embeddings = torch.randn(BATCH_SIZE, NUM_SLICES_PER_SCAN, VISION_FEATURE_DIM)

    # Instantiate the aggregator
    try:
        # Try different methods by changing AGGREGATION_METHOD: 'mean', 'lstm', 'gru', 'transformer'
        # Pass all relevant config values
        slice_aggregator = SliceAggregator(
            input_dim=VISION_FEATURE_DIM,
            method=AGGREGATION_METHOD,
            hidden_dim=AGGREGATOR_HIDDEN_DIM,
            rnn_bidirectional=RNN_BIDIRECTIONAL,
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

    except ImportError as ie:
         print(f"\nImportError: {ie}. Make sure necessary libraries (torch) are installed and previous stages are accessible.")
    except Exception as e:
        print(f"\nError during slice aggregator test: {e}")
        import traceback
        traceback.print_exc()

    print("\nStage 3: Slice aggregation setup complete.\n")
