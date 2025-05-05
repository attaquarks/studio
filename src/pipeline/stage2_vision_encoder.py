# ========== Stage 2: Vision Encoder – Feature Extraction ==========
import torch
import torch.nn as nn
import timm # Or use transformers library
import warnings
import os
# from transformers import AutoModel

# --- Import configurations from Stage 1 ---
try:
    # Use absolute import if stages are in the same package, relative if scripts
    # Assuming execution from a parent directory or package structure:
    # from stage1_data_acquisition import IMG_SIZE, NUM_SLICES_PER_SCAN, BATCH_SIZE
    # If running as standalone scripts, relative import might work if in the same folder:
     from .stage1_data_acquisition import IMG_SIZE, NUM_SLICES_PER_SCAN, BATCH_SIZE
except ImportError:
    warnings.warn("Could not import configurations from stage1_data_acquisition. Using placeholder values for Stage 2.")
    IMG_SIZE = 224 # Placeholder
    NUM_SLICES_PER_SCAN = 64 # Placeholder
    BATCH_SIZE = 4 # Placeholder


# --- Configuration ---
VISION_MODEL_NAME = 'vit_base_patch16_224' # timm model name (e.g., ViT-Base)
# VISION_MODEL_NAME = 'google/vit-base-patch16-224-in21k' # HuggingFace model name
PRETRAINED = True
FEATURE_EXTRACTION_METHOD = 'cls' # 'cls' or 'avg' pooling

# --- Vision Encoder Class ---
class VisionEncoder(nn.Module):
    """
    Encodes batches of image slices using a pre-trained vision model (e.g., ViT).
    Exposes the feature dimension via `self.feature_dim`.
    """
    def __init__(self, model_name=VISION_MODEL_NAME, pretrained=PRETRAINED,
                 feature_extraction_method=FEATURE_EXTRACTION_METHOD):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.feature_extraction_method = feature_extraction_method

        # Load the pre-trained vision model (using timm here)
        try:
            self.model = timm.create_model(self.model_name, pretrained=self.pretrained)
            # Remove the final classification layer to get features
            # Check if model has 'head' or 'fc' or similar classifier attribute
            if hasattr(self.model, 'head'):
                self.feature_dim = self.model.head.in_features # Get input features dim of the classifier head
                self.model.head = nn.Identity() # Replace head with identity layer
            elif hasattr(self.model, 'fc'): # Common in ResNets etc.
                self.feature_dim = self.model.fc.in_features
                self.model.fc = nn.Identity()
            elif hasattr(self.model, 'classifier'): # Some models use 'classifier'
                 self.feature_dim = self.model.classifier.in_features
                 self.model.classifier = nn.Identity()
            else:
                 # Fallback for ViT if no head attribute - get from final block or norm layer output
                 # This might require inspecting the specific timm model structure
                 if hasattr(self.model, 'norm') and hasattr(self.model.norm, 'normalized_shape'):
                      # Check if normalized_shape is an iterable (like tuple/list) or int
                      norm_shape = self.model.norm.normalized_shape
                      if isinstance(norm_shape, (list, tuple)) and len(norm_shape) > 0:
                           self.feature_dim = norm_shape[0]
                      elif isinstance(norm_shape, int):
                           self.feature_dim = norm_shape
                      else:
                           # Attempt inference through a dummy forward pass if structure is unknown
                           warnings.warn(f"Could not determine feature dim from norm layer shape ({norm_shape}). Attempting dummy forward pass.")
                           try:
                                dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE) # Use IMG_SIZE from config
                                # Try forward_features first
                                if hasattr(self.model, 'forward_features'):
                                     dummy_output = self.model.forward_features(dummy_input)
                                else: # Fallback to full forward
                                     dummy_output = self.model(dummy_input)

                                if len(dummy_output.shape) == 3: # ViT-like (B, SeqLen, D)
                                     self.feature_dim = dummy_output.shape[-1]
                                elif len(dummy_output.shape) == 2: # Possibly already pooled (B, D)
                                     self.feature_dim = dummy_output.shape[-1]
                                else: # Likely CNN features (B, D, H', W') - not directly usable here
                                     raise ValueError("Dummy forward pass output shape not recognized for dim inference.")
                                print(f"Inferred feature_dim via dummy forward pass: {self.feature_dim}")
                           except Exception as e_dummy:
                                warnings.warn(f"Dummy forward pass failed ({e_dummy}). Assuming feature_dim=768. Check model structure.")
                                self.feature_dim = 768 # Default fallback
                 else:
                      # Last resort: try to infer from a known common dimension (less robust)
                      warnings.warn(f"Could not automatically determine feature dim for {self.model_name}. Assuming 768. Check model structure.")
                      self.feature_dim = 768 # Common ViT-Base dim, adjust if needed

            print(f"Loaded timm model '{self.model_name}' with feature dimension {self.feature_dim}.")

            # Example using HuggingFace Transformers (uncomment if preferred)
            # from transformers import AutoModel
            # self.model = AutoModel.from_pretrained(self.model_name)
            # self.feature_dim = self.model.config.hidden_size
            # # HF models usually don't need head removal for feature extraction with AutoModel
            # print(f"Loaded HuggingFace model '{self.model_name}' with feature dimension {self.feature_dim}.")

        except ImportError:
             raise RuntimeError("HuggingFace Transformers library not found. Install it with: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load vision model '{self.model_name}': {e}")


    def forward(self, slices_batch):
        """
        Args:
            slices_batch (torch.Tensor): Input tensor of shape (BatchSize, NumSlices, Channels, Height, Width)
        Returns:
            torch.Tensor: Slice features tensor of shape (BatchSize, NumSlices, FeatureDimension)
        """
        batch_size, num_slices, C, H, W = slices_batch.shape
        # Reshape for model input: Treat slices as batch items
        # Input shape expected by ViT: (BatchSize * NumSlices, Channels, Height, Width)
        slices_batch = slices_batch.view(batch_size * num_slices, C, H, W)

        # --- Feature Extraction ---
        # Pass the reshaped batch through the vision model

        # Example for timm models (common interface):
        # Using model directly often gives classification output. Use specific feature methods if available.
        # `forward_features` is common in many timm models including ViT.
        if hasattr(self.model, 'forward_features'):
             features = self.model.forward_features(slices_batch)
        # elif hasattr(self.model, 'extract_features'): # Some models might use this
        #     features = self.model.extract_features(slices_batch)
        else:
             # Fallback: Use the full forward pass and hope the Identity layer works
             # This might return different things depending on the model structure before the head
             warnings.warn(f"Model {self.model_name} has no 'forward_features'. Using full forward pass. Output interpretation might be incorrect.")
             features = self.model(slices_batch) # Output shape might vary

        # features shape for ViT is typically (BatchSize * NumSlices, NumTokens + 1, FeatureDimension)
        # For CNNs, it might be (BatchSize*NumSlices, FeatureDim, H', W') after avgpool layer removed

        # --- Select Feature Representation ---
        # This part depends heavily on the output shape of `features`

        if len(features.shape) == 4: # Likely CNN features (B*N, D, H', W')
             # Apply global average pooling
             slice_features = torch.mean(features, dim=[2, 3]) # Shape: (B*N, D)
             self.feature_extraction_method = 'avg_cnn' # Indicate how features were derived
             print("Detected CNN-like features, applied global average pooling.")

        elif len(features.shape) == 3: # Likely Transformer/ViT features (B*N, SeqLen, D)
            if self.feature_extraction_method == 'cls':
                 # The [CLS] token embedding is usually the first one
                 slice_features = features[:, 0] # Shape: (BatchSize * NumSlices, FeatureDimension)
            elif self.feature_extraction_method == 'avg':
                 # Average pooling over patch embeddings (excluding the [CLS] token)
                 # Assumes CLS token is present at index 0
                 slice_features = features[:, 1:, :].mean(dim=1) # Shape: (BatchSize * NumSlices, FeatureDimension)
            else:
                 raise ValueError(f"Unsupported feature_extraction_method for Transformer/ViT: {self.feature_extraction_method}")
        elif len(features.shape) == 2: # Features might already be pooled (B*N, D)
             slice_features = features
             print("Features seem to be already pooled.")
        else:
             raise ValueError(f"Unexpected feature shape from model: {features.shape}")


        # Reshape back to (BatchSize, NumSlices, FeatureDimension)
        slice_features = slice_features.view(batch_size, num_slices, self.feature_dim)

        return slice_features

# --- Example Usage ---
if __name__ == "__main__": # Ensures this runs only when script is executed directly
    print("--- Stage 2 Example ---")
    # Assume sample_batch from Stage 1 exists and contains 'pixel_values'
    # Or create dummy data:
    # Make sure dummy data uses the imported config values
    dummy_pixel_values = torch.randn(BATCH_SIZE, NUM_SLICES_PER_SCAN, 3, IMG_SIZE, IMG_SIZE) # B, N, C, H, W

    # Instantiate the encoder
    try:
        vision_encoder = VisionEncoder(
            model_name=VISION_MODEL_NAME,
            pretrained=PRETRAINED,
            feature_extraction_method=FEATURE_EXTRACTION_METHOD
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_encoder.to(device)
        dummy_pixel_values = dummy_pixel_values.to(device)
        print(f"Running example on device: {device}")
        print(f"Input dummy pixel values shape: {dummy_pixel_values.shape}")


        # Perform feature extraction
        vision_encoder.eval() # Set to evaluation mode for inference
        with torch.no_grad(): # Disable gradient calculation for inference
            slice_embeddings = vision_encoder(dummy_pixel_values)

        print(f"Vision encoder output shape: {slice_embeddings.shape}")
        # Expected: (BatchSize, NumSlices, FeatureDimension)
        assert slice_embeddings.shape[0] == BATCH_SIZE
        assert slice_embeddings.shape[1] == NUM_SLICES_PER_SCAN
        assert slice_embeddings.shape[2] == vision_encoder.feature_dim # Check against instance attribute

    except ImportError as ie:
         print(f"\nImportError: {ie}. Make sure necessary libraries (torch, timm) are installed.")
    except Exception as e:
        print(f"\nError during vision encoder test: {e}")
        import traceback
        traceback.print_exc()

    print("\nStage 2: Vision feature extraction setup complete.\n")
