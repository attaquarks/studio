# ========== Stage 2: Vision Encoder – Feature Extraction ==========
import torch
import torch.nn as nn
import timm # Or use transformers library
# from transformers import AutoModel

# --- Configuration ---
VISION_MODEL_NAME = 'vit_base_patch16_224' # timm model name (e.g., ViT-Base)
# VISION_MODEL_NAME = 'google/vit-base-patch16-224-in21k' # HuggingFace model name
PRETRAINED = True
FEATURE_EXTRACTION_METHOD = 'cls' # 'cls' or 'avg' pooling

# Placeholder values from Stage 1 - these might be imported or passed
IMG_SIZE = 224
NUM_SLICES_PER_SCAN = 64
BATCH_SIZE = 4

# --- Vision Encoder Class ---
class VisionEncoder(nn.Module):
    """
    Encodes batches of image slices using a pre-trained vision model (e.g., ViT).
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
            self.feature_dim = self.model.head.in_features # Get input features dim of the classifier head
            self.model.head = nn.Identity() # Replace head with identity layer
            print(f"Loaded timm model '{self.model_name}' with feature dimension {self.feature_dim}.")

            # Example using HuggingFace Transformers (uncomment if preferred)
            # self.model = AutoModel.from_pretrained(self.model_name)
            # self.feature_dim = self.model.config.hidden_size
            # print(f"Loaded HuggingFace model '{self.model_name}' with feature dimension {self.feature_dim}.")

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
        # Output depends on the model library (timm vs transformers)
        # Using timm ViT: model.forward_features often returns patch embeddings + CLS token
        # Using HF ViT: model(pixel_values).last_hidden_state returns sequence of embeddings

        # Example for timm ViT models:
        features = self.model.forward_features(slices_batch)
        # features shape is typically (BatchSize * NumSlices, NumTokens + 1, FeatureDimension) for ViT
        # where NumTokens = (H/patch_size) * (W/patch_size), and +1 is for the [CLS] token

        # Example for HuggingFace ViT models:
        # outputs = self.model(pixel_values=slices_batch)
        # features = outputs.last_hidden_state # Shape (BatchSize*NumSlices, SeqLen, FeatureDim)


        # --- Select Feature Representation ---
        if self.feature_extraction_method == 'cls':
             # The [CLS] token embedding is usually the first one in the sequence
             slice_features = features[:, 0] # Shape: (BatchSize * NumSlices, FeatureDimension)
        elif self.feature_extraction_method == 'avg':
             # Average pooling over patch embeddings (excluding the [CLS] token if present)
             # For ViT, index 0 is CLS, so pool from index 1 onwards
             slice_features = features[:, 1:, :].mean(dim=1) # Shape: (BatchSize * NumSlices, FeatureDimension)
        else:
             raise ValueError(f"Unsupported feature_extraction_method: {self.feature_extraction_method}")

        # Reshape back to (BatchSize, NumSlices, FeatureDimension)
        slice_features = slice_features.view(batch_size, num_slices, self.feature_dim)

        return slice_features

# --- Example Usage ---
if __name__ == "__main__": # Ensures this runs only when script is executed directly
    print("--- Stage 2 Example ---")
    # Assume sample_batch from Stage 1 exists and contains 'pixel_values'
    # Or create dummy data:
    dummy_pixel_values = torch.randn(BATCH_SIZE, NUM_SLICES_PER_SCAN, 3, IMG_SIZE, IMG_SIZE) # B, N, C, H, W

    # Instantiate the encoder
    try:
        vision_encoder = VisionEncoder(model_name=VISION_MODEL_NAME, feature_extraction_method=FEATURE_EXTRACTION_METHOD)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_encoder.to(device)
        dummy_pixel_values = dummy_pixel_values.to(device)

        # Perform feature extraction
        vision_encoder.eval() # Set to evaluation mode for inference
        with torch.no_grad(): # Disable gradient calculation for inference
            slice_embeddings = vision_encoder(dummy_pixel_values)

        print(f"Vision encoder output shape: {slice_embeddings.shape}") # Expected: (BatchSize, NumSlices, FeatureDimension)
    except Exception as e:
        print(f"Error initializing or running vision encoder: {e}")

    print("Stage 2: Vision feature extraction setup complete.\n")
