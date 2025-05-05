# ========== Stage 3: Slice Aggregation – Scan-Level Representation ==========
import torch
import torch.nn as nn
import warnings
import os

# --- Import configurations from previous stages ---
try:
    from .stage1_data_acquisition import BATCH_SIZE, NUM_SLICES_PER_SCAN
    from .stage2_vision_encoder import VISION_FEATURE_DIM as STAGE2_VISION_FEATURE_DIM
    VISION_FEATURE_DIM = STAGE2_VISION_FEATURE_DIM
    print(f"Successfully derived VISION_FEATURE_DIM from Stage 2: {VISION_FEATURE_DIM}")
except ImportError as e_imp:
    warnings.warn(f"Could not import configurations from stage1/stage2 ({e_imp}). Using placeholder values for Stage 3.")
    BATCH_SIZE = 4
    NUM_SLICES_PER_SCAN = 64
    VISION_FEATURE_DIM = 768 # Placeholder (e.g., ViT-Base)
except NameError as e_name:
    warnings.warn(f"VISION_FEATURE_DIM not found in stage2 ({e_name}). Check stage2 definition. Using placeholder 768.")
    BATCH_SIZE = 4
    NUM_SLICES_PER_SCAN = 64
    VISION_FEATURE_DIM = 768
except Exception as e_other:
    warnings.warn(f"Error importing/deriving config from stage1/stage2: {e_other}. Using placeholder values.")
    BATCH_SIZE = 4
    NUM_SLICES_PER_SCAN = 64
    VISION_FEATURE_DIM = 768


# --- Configuration for Stage 3 ---
AGGREGATION_TYPE = 'lstm' # Options: 'lstm', 'gru', 'transformer', 'mean'
AGGREGATOR_HIDDEN_DIM = 512 # Hidden dimension for LSTM/GRU/Transformer feedforward
AGGREGATOR_BIDIRECTIONAL = True # Whether to use bidirectional LSTM/GRU
AGGREGATOR_NUM_LAYERS = 2 # Number of layers for LSTM/GRU/Transformer
AGGREGATOR_DROPOUT = 0.1 # Dropout rate for LSTM/GRU/Transformer

# --- Slice Aggregator Class ---
class SliceAggregator(nn.Module):
    """Aggregates features from individual slices into a scan-level representation."""

    def __init__(self,
                 feature_dim: int = VISION_FEATURE_DIM,
                 hidden_dim: int = AGGREGATOR_HIDDEN_DIM,
                 aggr_type: str = AGGREGATION_TYPE,
                 bidirectional: bool = AGGREGATOR_BIDIRECTIONAL,
                 num_layers: int = AGGREGATOR_NUM_LAYERS,
                 dropout: float = AGGREGATOR_DROPOUT):
        """
        Initialize the slice aggregator.

        Args:
            feature_dim: Dimension of the input features (from Stage 2)
            hidden_dim: Hidden dimension for LSTM/GRU or Transformer FFN
            aggr_type: Aggregation type ('lstm', 'gru', 'transformer', 'mean')
            bidirectional: Whether to use bidirectional LSTM/GRU
            num_layers: Number of layers for LSTM/GRU/Transformer
            dropout: Dropout rate
        """
        super().__init__()

        self.aggr_type = aggr_type.lower() # Ensure lowercase for comparison
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional if self.aggr_type in ['lstm', 'gru'] else False # Bidirectional only for RNNs
        self.dropout = dropout if num_layers > 1 else 0 # Apply dropout only if multiple layers

        self.aggregator = None # Initialize aggregator
        aggregator_output_dim = feature_dim # Default output dim

        if self.aggr_type == "lstm":
            self.aggregator = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True, # Input shape (B, S, D)
                bidirectional=self.bidirectional,
                dropout=self.dropout
            )
            aggregator_output_dim = hidden_dim * (2 if self.bidirectional else 1)

        elif self.aggr_type == "gru":
            self.aggregator = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True, # Input shape (B, S, D)
                bidirectional=self.bidirectional,
                dropout=self.dropout
            )
            aggregator_output_dim = hidden_dim * (2 if self.bidirectional else 1)

        elif self.aggr_type == "transformer":
            # Transformer requires input_dim % nhead == 0
            nhead = 8 # Example, make configurable if needed
            if feature_dim % nhead != 0:
                 warnings.warn(f"Transformer input_dim ({feature_dim}) not divisible by nhead ({nhead}). Adjust config or model.")
                 # Adjust nhead if possible
                 original_nhead = nhead
                 while feature_dim % nhead != 0 and nhead > 1: nhead -= 1
                 if feature_dim % nhead == 0:
                     warnings.warn(f"Adjusted nhead to {nhead}.")
                 else:
                     raise ValueError(f"Cannot find compatible nhead for input_dim {feature_dim}. Original nhead was {original_nhead}.")

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim, # FFN dimension
                dropout=self.dropout, # Use the main dropout param
                batch_first=True # Input shape (B, S, D)
            )
            self.aggregator = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers
            )
            # Transformer output dim is same as input dim
            aggregator_output_dim = feature_dim

        elif self.aggr_type == "mean":
            # Mean pooling is parameter-free, handled in forward pass
            aggregator_output_dim = feature_dim
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggr_type}")

        # Final linear projection layer (optional but good practice)
        # Projects the output of the aggregator (which might have a different dim)
        # back to the original feature_dim or a desired output dimension.
        # Let's project back to feature_dim for consistency with 'mean' and 'transformer'.
        self.output_dim = feature_dim # Define final output dim
        self.output_proj = nn.Linear(aggregator_output_dim, self.output_dim)

        print(f"Initialized SliceAggregator: Type='{self.aggr_type}', InputDim={feature_dim}, OutputDim={self.output_dim}")
        if self.aggr_type in ['lstm', 'gru']:
             print(f"  RNN Config: HiddenDim={hidden_dim}, Layers={num_layers}, Bidirectional={self.bidirectional}, Dropout={self.dropout}")
        elif self.aggr_type == 'transformer':
             print(f"  Transformer Config: FFN_Dim={hidden_dim}, Layers={num_layers}, Heads={nhead}, Dropout={self.dropout}")


    def forward(self, x, mask=None):
        """
        Forward pass through the slice aggregator.

        Args:
            x: Input tensor of shape [B, S, D_in] (FeatureDimension from Stage 2)
            mask: Optional mask (e.g., for transformer padding). Shape [B, S].
                  True values indicate positions to be masked/ignored.

        Returns:
            Aggregated feature tensor of shape [B, D_out] (self.output_dim)
        """
        if x.shape[2] != self.feature_dim:
            raise ValueError(f"Input feature dimension ({x.shape[2]}) does not match aggregator input_dim ({self.feature_dim})")

        if self.aggr_type == "lstm" or self.aggr_type == "gru":
            # LSTM/GRU aggregation
            # packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False) # Use if sequences have variable lengths
            # output, hidden = self.aggregator(packed_input)
            # output_unpacked, _ = pad_packed_sequence(output, batch_first=True)

            # Assuming fixed length sequences (padded in Dataset)
            output, hidden = self.aggregator(x) # output shape [B, S, D_agg], hidden shape depends on RNN type/layers

            # Extract final hidden state(s) to represent the sequence
            if self.aggr_type == "lstm":
                 # hidden is tuple (h_n, c_n), each of shape (num_layers * num_directions, B, H)
                 h_n = hidden[0] # Take the hidden state h_n
            else: # GRU
                 # hidden is h_n directly, shape (num_layers * num_directions, B, H)
                 h_n = hidden

            # Get hidden state from the last layer
            if self.bidirectional:
                # Concatenate last forward and backward hidden states
                # h_n shape: (L*2, B, H) -> access last forward (-2) and last backward (-1)
                aggregated = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) # Shape [B, H*2]
            else:
                # h_n shape: (L, B, H) -> access last layer (-1)
                aggregated = h_n[-1, :, :] # Shape [B, H]

        elif self.aggr_type == "transformer":
            # Transformer aggregation
            # Transformer expects boolean mask where True indicates masking.
            # If mask is provided (e.g., from padding), use it.
            if mask is not None:
                 if mask.shape != x.shape[:2]:
                      raise ValueError(f"Mask shape {mask.shape} incompatible with input shape {x.shape[:2]}")
                 # Transformer layer expects mask where True means "ignore"
                 src_key_padding_mask = mask.bool()
                 output = self.aggregator(x, src_key_padding_mask=src_key_padding_mask) # Shape [B, S, D_in]
            else:
                 output = self.aggregator(x) # Shape [B, S, D_in]

            # Aggregate the output sequence - mean pooling is common
            # Use masked mean if mask is available to ignore padding
            if mask is not None:
                 # Invert mask for selecting valid tokens (False means valid)
                 valid_tokens_mask = ~src_key_padding_mask # Shape [B, S]
                 # Expand mask for broadcasting: [B, S, 1]
                 valid_tokens_mask = valid_tokens_mask.unsqueeze(-1)
                 # Sum valid tokens and divide by number of valid tokens
                 masked_output = output * valid_tokens_mask # Zero out masked tokens
                 summed_features = masked_output.sum(dim=1) # Shape [B, D_in]
                 num_valid_tokens = valid_tokens_mask.sum(dim=1) # Shape [B, 1]
                 num_valid_tokens = num_valid_tokens.clamp(min=1) # Avoid division by zero
                 aggregated = summed_features / num_valid_tokens # Shape [B, D_in]
            else:
                 # Simple mean pooling if no mask
                 aggregated = output.mean(dim=1) # Shape [B, D_in]

        elif self.aggr_type == "mean":
            # Simple mean pooling across the slice dimension (dim=1)
            aggregated = x.mean(dim=1) # Shape [B, D_in]

        else:
            # This case should be caught in __init__
             raise NotImplementedError(f"Forward pass not implemented for type '{self.aggr_type}'")

        # Project the aggregated features to the final output dimension
        output = self.output_proj(aggregated) # Shape [B, self.output_dim]

        return output


# --- Determine Output Dimension Based on Config ---
# Instantiate a temporary aggregator to get its configured output dim
try:
    _temp_aggregator = SliceAggregator(
        feature_dim=VISION_FEATURE_DIM, # Use derived/placeholder dim
        aggr_type=AGGREGATION_TYPE,
        hidden_dim=AGGREGATOR_HIDDEN_DIM,
        bidirectional=AGGREGATOR_BIDIRECTIONAL,
        num_layers=AGGREGATOR_NUM_LAYERS,
        dropout=AGGREGATOR_DROPOUT
    )
    AGGREGATOR_OUTPUT_DIM = _temp_aggregator.output_dim
    print(f"Stage 3 Aggregator Final Output Dimension determined: {AGGREGATOR_OUTPUT_DIM}")
    del _temp_aggregator # Clean up
except ValueError as e_val: # Catch config errors like nhead incompatibility
     warnings.warn(f"Could not determine AGGREGATOR_OUTPUT_DIM due to configuration error: {e_val}. Using fallback.")
     # Fallback logic based on the intended output dim (which is VISION_FEATURE_DIM in this setup)
     AGGREGATOR_OUTPUT_DIM = VISION_FEATURE_DIM
     print(f"Stage 3 Aggregator Output Dimension (fallback): {AGGREGATOR_OUTPUT_DIM}")
except Exception as e:
     warnings.warn(f"Could not determine AGGREGATOR_OUTPUT_DIM dynamically: {e}. Using fallback.")
     AGGREGATOR_OUTPUT_DIM = VISION_FEATURE_DIM # Fallback to input dim
     print(f"Stage 3 Aggregator Output Dimension (fallback): {AGGREGATOR_OUTPUT_DIM}")


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Stage 3 Example ---")

    # Assume slice_features from Stage 2 exists (B, S, D_in)
    # Create dummy data using the derived/placeholder VISION_FEATURE_DIM
    dummy_slice_features = torch.randn(BATCH_SIZE, NUM_SLICES_PER_SCAN, VISION_FEATURE_DIM)
    # Optional: Create a dummy mask (e.g., last few slices are padding)
    dummy_mask = torch.zeros(BATCH_SIZE, NUM_SLICES_PER_SCAN, dtype=torch.bool)
    # dummy_mask[:, -5:] = True # Example: Mask last 5 slices

    # Instantiate the aggregator using configured parameters
    try:
        slice_aggregator = SliceAggregator(
            feature_dim=VISION_FEATURE_DIM,
            aggr_type=AGGREGATION_TYPE,
            hidden_dim=AGGREGATOR_HIDDEN_DIM,
            bidirectional=AGGREGATOR_BIDIRECTIONAL,
            num_layers=AGGREGATOR_NUM_LAYERS,
            dropout=AGGREGATOR_DROPOUT
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        slice_aggregator.to(device)
        dummy_slice_features = dummy_slice_features.to(device)
        # dummy_mask = dummy_mask.to(device) # Move mask to device if using
        print(f"Running example on device: {device}")
        print(f"Input dummy features shape: {dummy_slice_features.shape}")

        # Perform aggregation
        slice_aggregator.eval()
        with torch.no_grad():
            # Pass mask only if using transformer
            # mask_input = dummy_mask if AGGREGATION_TYPE == 'transformer' else None
            mask_input = None # Example without mask for now
            scan_level_embedding = slice_aggregator(dummy_slice_features, mask=mask_input)

        print(f"Slice aggregator ({AGGREGATION_TYPE}) output shape: {scan_level_embedding.shape}")
        # Expected shape: (BatchSize, AGGREGATOR_OUTPUT_DIM)
        assert scan_level_embedding.shape[0] == BATCH_SIZE
        assert scan_level_embedding.shape[1] == AGGREGATOR_OUTPUT_DIM # Check against derived dim

    except ImportError as ie:
         print(f"\nImportError: {ie}. Make sure previous stages are accessible.")
    except Exception as e:
        print(f"\nError during slice aggregator example: {e}")
        import traceback
        traceback.print_exc()

    print("\nStage 3: Slice aggregation setup complete.\n")
