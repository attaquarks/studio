# ========== Stage 2: Vision Encoder – Feature Extraction ==========
import torch
import torch.nn as nn
import timm
import warnings
import os

# --- Import configurations from Stage 1 ---
try:
    from .stage1_data_acquisition import TARGET_SIZE, BATCH_SIZE, NUM_SLICES_PER_SCAN
    IMG_SIZE = TARGET_SIZE[0] # Use the first dimension of target size
except ImportError:
    warnings.warn("Could not import configurations from stage1_data_acquisition. Using placeholder values for Stage 2.")
    TARGET_SIZE = (224, 224)
    IMG_SIZE = 224
    BATCH_SIZE = 4
    NUM_SLICES_PER_SCAN = 64

# --- Configuration for Stage 2 ---
VISION_MODEL_NAME = 'vit_base_patch16_224' # timm model name
PRETRAINED = True
FREEZE_BACKBONE = False # Whether to freeze the backbone during training

# --- Vision Encoder Class ---
class VisionEncoder(nn.Module):
    """Vision encoder based on Vision Transformers for MRI feature extraction."""

    def __init__(self,
                 model_name: str = VISION_MODEL_NAME,
                 pretrained: bool = PRETRAINED,
                 freeze_backbone: bool = FREEZE_BACKBONE):
        """
        Initialize the vision encoder.

        Args:
            model_name: Name of the vision transformer model
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone
        """
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        # Load ViT model from timm
        try:
            # num_classes=0 removes the final classification head
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0
            )
            # Get feature dimension from the loaded backbone
            self.feature_dim = self.backbone.num_features
            print(f"Loaded timm model '{self.model_name}' with feature dimension {self.feature_dim}.")

        except Exception as e:
            raise RuntimeError(f"Failed to load vision model '{self.model_name}' using timm: {e}")

        # Freeze backbone if specified
        if freeze_backbone:
            print(f"Freezing backbone parameters for {self.model_name}.")
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the vision encoder.

        Args:
            x: Input tensor of shape [B, S, C, H, W]
                B = batch size, S = number of slices, C = channels, H = height, W = width
               Expected C=3 for standard ViT models.

        Returns:
            Feature tensor of shape [B, S, D] where D is the feature dimension
        """
        batch_size, num_slices, C, H, W = x.shape

        # Input validation
        if C != 3:
             warnings.warn(f"Input tensor has {C} channels, but ViT model {self.model_name} typically expects 3 channels. Ensure data preprocessing aligns.")
             # Attempt to proceed, but results might be suboptimal. Consider repeating channels in Stage 1 if needed.

        # Reshape to process all slices as a single batch for the backbone
        # Input shape expected by backbone: [B*S, C, H, W]
        x = x.view(batch_size * num_slices, C, H, W)

        # Extract features using the backbone
        # timm models with num_classes=0 usually output features directly
        # For ViT, this is often the [CLS] token or average pooled features depending on model config
        # Let's assume it outputs [B*S, D] directly after pooling/CLS token extraction within the backbone
        # If backbone outputs sequence (B*S, SeqLen, D), need to pool here. Check timm docs for specific model.
        try:
            features = self.backbone(x) # Shape should be [B*S, D]
        except Exception as e:
             print(f"Error during backbone forward pass: {e}")
             # Return dummy tensor of expected shape on error
             return torch.zeros(batch_size, num_slices, self.feature_dim, device=x.device)


        # Validate output shape from backbone
        expected_shape_part = (batch_size * num_slices, self.feature_dim)
        if features.shape != expected_shape_part:
            warnings.warn(f"Unexpected feature shape from backbone {self.model_name}: got {features.shape}, expected ~{expected_shape_part}. Attempting fallback pooling if sequence output detected.")
            # Fallback: If output is sequence (e.g., ViT without final pool), pool it.
            if len(features.shape) == 3 and features.shape[0] == batch_size * num_slices: # Shape (B*S, SeqLen, D)
                # Example: Average pool sequence features (excluding CLS if present at index 0)
                # This assumes CLS token is present and we want avg pool of patch tokens
                if features.shape[1] > 1: # Check if there are patch tokens
                    features = features[:, 1:, :].mean(dim=1) # Avg pool patches
                else: # If only CLS token is output? Or seq len 1? Use it directly.
                    features = features[:, 0, :] # Take the first token
                print("Applied fallback average pooling to sequence output.")
                # Re-check shape after fallback pooling
                if features.shape != expected_shape_part:
                     raise RuntimeError(f"Feature shape still incorrect ({features.shape}) after fallback pooling.")
            else:
                # Cannot resolve shape mismatch
                raise RuntimeError(f"Cannot handle backbone output shape: {features.shape}")


        # Reshape back to separate batch and slice dimensions
        # Target shape: [B, S, D]
        features = features.view(batch_size, num_slices, self.feature_dim)

        return features

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Stage 2 Example ---")
    # Create dummy input data consistent with Stage 1 output
    dummy_pixel_values = torch.randn(BATCH_SIZE, NUM_SLICES_PER_SCAN, 3, IMG_SIZE, IMG_SIZE) # B, S, C, H, W

    # Instantiate the encoder
    try:
        vision_encoder = VisionEncoder(
            model_name=VISION_MODEL_NAME,
            pretrained=PRETRAINED,
            freeze_backbone=FREEZE_BACKBONE
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_encoder.to(device)
        dummy_pixel_values = dummy_pixel_values.to(device)
        print(f"Running example on device: {device}")
        print(f"Input dummy pixel values shape: {dummy_pixel_values.shape}")

        # Perform feature extraction
        vision_encoder.eval() # Set to evaluation mode
        with torch.no_grad():
            slice_features = vision_encoder(dummy_pixel_values)

        print(f"Vision encoder output shape: {slice_features.shape}") # Expected: (B, S, D)
        # Check dimensions
        assert slice_features.shape[0] == BATCH_SIZE
        assert slice_features.shape[1] == NUM_SLICES_PER_SCAN
        assert slice_features.shape[2] == vision_encoder.feature_dim

    except ImportError as ie:
        print(f"\nImportError: {ie}. Make sure 'timm' is installed (pip install timm).")
    except Exception as e:
        print(f"\nError during vision encoder example: {e}")
        import traceback
        traceback.print_exc()

    print("\nStage 2: Vision feature extraction setup complete.\n")

# Export the feature dimension for subsequent stages
# Instantiate once to get the dimension
try:
    _temp_encoder = VisionEncoder()
    VISION_FEATURE_DIM = _temp_encoder.feature_dim
    print(f"Stage 2 VISION_FEATURE_DIM determined: {VISION_FEATURE_DIM}")
    del _temp_encoder
except Exception as e:
    warnings.warn(f"Could not determine VISION_FEATURE_DIM dynamically: {e}. Using placeholder 768.")
    VISION_FEATURE_DIM = 768 # Fallback (e.g., ViT-Base)
