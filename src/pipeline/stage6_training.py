# ========== Stage 6: Training and Fine-Tuning ==========
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader # For dummy loader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # For dummy model/tokenizer
import warnings
import os

# --- Import Components from Previous Stages ---
# Attempt to import necessary classes and variables
# Use try-except blocks to handle cases where stages are run independently
pipeline_dir = os.path.dirname(__file__) # Get directory of the current file

try:
    from .stage1_data_acquisition import MRIDataset, BATCH_SIZE, NUM_SLICES_PER_SCAN # For dummy loader
except ImportError:
    warnings.warn("Could not import from stage1. Defining dummy MRIDataset class and constants for Stage 6.")
    BATCH_SIZE = 2
    NUM_SLICES_PER_SCAN = 8
    IMG_SIZE = 32 # Small dummy size
    # Define a minimal dummy Dataset class
    class MRIDataset(torch.utils.data.Dataset):
        def __init__(self, file_paths, labels, **kwargs):
            self.file_paths = file_paths
            self.labels = labels
            self.num_slices = NUM_SLICES_PER_SCAN
            self.img_size = IMG_SIZE
        def __len__(self): return len(self.file_paths)
        def __getitem__(self, idx):
            label = self.labels[idx]
            dummy_pixels = torch.randn(self.num_slices, 3, self.img_size, self.img_size)
            item = {'pixel_values': dummy_pixels}
            if isinstance(label, dict):
                item['question'] = label.get('question', 'Dummy Q')
                item['answer'] = label.get('answer', 'Dummy A')
            else:
                item['report'] = str(label)
            return item

try:
    from .stage2_vision_encoder import VisionEncoder
except ImportError:
    warnings.warn("Could not import VisionEncoder from stage2. Defining dummy VisionEncoder class.")
    class VisionEncoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.feature_dim = 768 # Dummy dimension
            self.dummy_layer = nn.Linear(10, self.feature_dim) # Placeholder layer
        def forward(self, x):
            b, n, c, h, w = x.shape
            # Simple projection, ignoring actual image data for dummy structure
            dummy_features = torch.randn(b * n, self.feature_dim, device=x.device)
            return dummy_features.view(b, n, self.feature_dim)

try:
    from .stage3_slice_aggregation import SliceAggregator
except ImportError:
    warnings.warn("Could not import SliceAggregator from stage3. Defining dummy SliceAggregator class.")
    class SliceAggregator(nn.Module):
        def __init__(self, input_dim=768, method='mean', **kwargs):
            super().__init__()
            self.method = method
            self.input_dim = input_dim
            self.output_dim = input_dim # Dummy output dim for mean
        def forward(self, x):
            return x.mean(dim=1) # Dummy mean aggregation

try:
    from .stage4_vision_language_bridge import VisionLanguageBridge, LANGUAGE_MODEL_DIM as STAGE4_TARGET_DIM
except ImportError:
    warnings.warn("Could not import from stage4. Defining dummy VisionLanguageBridge class and constant.")
    STAGE4_TARGET_DIM = 768 # Must match dummy LM dim below
    class VisionLanguageBridge(nn.Module):
        def __init__(self, visual_dim=768, language_dim=STAGE4_TARGET_DIM, **kwargs):
            super().__init__()
            self.projection = nn.Linear(visual_dim, language_dim)
            self.output_dim = language_dim
        def forward(self, x):
            return self.projection(x)

try:
    # Import model, tokenizer, is_encoder_decoder flag, and USE_QLORA flag from stage 5
    from .stage5_language_decoder import (
        language_model as loaded_language_model,
        tokenizer as loaded_tokenizer,
        is_encoder_decoder as loaded_is_encoder_decoder,
        USE_QLORA,
        LANGUAGE_MODEL_NAME # Re-use the model name if reloading
    )
    # Check if model and tokenizer were successfully loaded in stage 5
    if loaded_language_model is None or loaded_tokenizer is None:
        raise ImportError("Stage 5 failed to load model or tokenizer.")
    language_model = loaded_language_model
    tokenizer = loaded_tokenizer
    is_encoder_decoder = loaded_is_encoder_decoder
    print("Successfully imported components from Stage 5.")

except ImportError as e:
    warnings.warn(f"Could not import components from stage5 ({e}). Defining dummy language model/tokenizer for Stage 6 structure.")
    # Define dummy components if import fails
    LANGUAGE_MODEL_NAME = 'google/flan-t5-base' # Use a consistent model name
    try:
        tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e_tok:
        raise RuntimeError(f"Failed to load dummy tokenizer {LANGUAGE_MODEL_NAME}: {e_tok}")

    try:
        language_model = AutoModelForSeq2SeqLM.from_pretrained(LANGUAGE_MODEL_NAME) # Load a small dummy model
        is_encoder_decoder = True # T5 is encoder-decoder
    except Exception as e_mod:
         raise RuntimeError(f"Failed to load dummy model {LANGUAGE_MODEL_NAME}: {e_mod}")

    # Assume QLoRA is not used if we are defining dummy components
    USE_QLORA = False
    STAGE4_TARGET_DIM = language_model.config.d_model # Get dim from dummy model
    print(f"Using dummy T5-base model/tokenizer. Target LM Dim: {STAGE4_TARGET_DIM}")



# --- Configuration ---
LEARNING_RATE = 1e-4 # Adjust based on experiments (might need lower for full model tuning, higher for LoRA)
NUM_EPOCHS = 3
MODEL_SAVE_PATH = "./neuroreport_model_checkpoint" # Directory to save checkpoints
# Inherited from Stage 5 or defaults
MAX_LABEL_LENGTH = 128 # Max length for target labels (reports/answers)

# --- Create actual component instances ---
# Use try-except to handle potential errors during instantiation
try:
    # Instantiate components using configurations from previous stages
    # Note: language_model and tokenizer are already loaded/defined above
    vision_encoder_instance = VisionEncoder() # Uses defaults from stage2 or dummy
    aggregator_input_dim = vision_encoder_instance.feature_dim
    slice_aggregator_instance = SliceAggregator(input_dim=aggregator_input_dim) # Uses defaults from stage3 or dummy

    # Bridge: only needed if aggregator output dim doesn't match LM dim
    bridge_input_dim = slice_aggregator_instance.output_dim
    if bridge_input_dim != STAGE4_TARGET_DIM:
        bridge_instance = VisionLanguageBridge(visual_dim=bridge_input_dim, language_dim=STAGE4_TARGET_DIM)
        print(f"Instantiated Bridge: {bridge_input_dim} -> {STAGE4_TARGET_DIM}")
    else:
        bridge_instance = nn.Identity() # No projection needed
        print(f"Aggregator output dim ({bridge_input_dim}) matches LM dim ({STAGE4_TARGET_DIM}). Using Identity Bridge.")

except Exception as e_inst:
    print(f"Error instantiating pipeline components: {e_inst}")
    print("Check component definitions and configurations in previous stages.")
    exit() # Exit if components cannot be created


# --- Combined Model (PyTorch Lightning Module) ---
class NeuroReportModel(pl.LightningModule):
    """
    Combines all stages into a single model for end-to-end training using PyTorch Lightning.
    """
    def __init__(self, vision_encoder, slice_aggregator, bridge, language_model, tokenizer,
                 is_encoder_decoder_model, max_label_length=MAX_LABEL_LENGTH, learning_rate=LEARNING_RATE):
        super().__init__()
        # Use save_hyperparameters to automatically save important config values
        # Ignore large model components and tokenizer
        self.save_hyperparameters(ignore=['vision_encoder', 'slice_aggregator', 'bridge', 'language_model', 'tokenizer'])

        self.vision_encoder = vision_encoder
        self.slice_aggregator = slice_aggregator
        self.bridge = bridge # This could be nn.Identity() if dims already match
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.is_encoder_decoder_model = is_encoder_decoder_model
        self.max_label_length = max_label_length
        self.learning_rate = learning_rate

        # Freeze components if needed (e.g., only train LoRA adapters and bridge)
        # Example: Freeze vision encoder if not fine-tuning it
        # for param in self.vision_encoder.parameters():
        #     param.requires_grad = False

        # If using QLoRA/PEFT, only the adapters (and potentially the bridge) should have requires_grad=True
        # PEFT's get_peft_model usually handles setting requires_grad correctly for adapters.
        # Verify trainable parameters:
        print("\n--- Trainable Parameters in NeuroReportModel ---")
        total_trainable_params = 0
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                print(f"  - {name} ({param.numel()})")
                total_trainable_params += param.numel()
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {total_trainable_params:,}")
        print(f"Trainable Ratio: {total_trainable_params / total_params * 100:.4f}%")
        print("-" * 40 + "\n")

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        """
        The forward pass through the entire pipeline.
        Args:
            pixel_values (torch.Tensor): Batch of image slices (B, N, C, H, W)
            input_ids (torch.Tensor, optional): Tokenized input text (e.g., questions) (B, SeqLen)
            attention_mask (torch.Tensor, optional): Attention mask for input_ids (B, SeqLen)
            labels (torch.Tensor, optional): Tokenized target text (answers/reports) (B, TargetSeqLen)
                                             Used for loss calculation during training.
        Returns:
            outputs: Raw output from the language model (contains loss if labels are provided).
                     For Seq2Seq: BaseModelOutputWithPastAndCrossAttentions or Seq2SeqLMOutput
                     For CausalLM: CausalLMOutputWithPast
        """
        # Stage 2: Vision Encoding
        slice_features = self.vision_encoder(pixel_values) # (B, N, VisFeatDim)

        # Stage 3: Slice Aggregation
        scan_embedding = self.slice_aggregator(slice_features) # (B, AggFeatDim)

        # Stage 4: Bridging (Projection)
        conditioned_embedding = self.bridge(scan_embedding) # (B, LangModelDim)

        # Stage 5: Language Model Processing
        # Prepare inputs for the specific language model type

        if self.is_encoder_decoder_model:
            # Reshape visual embedding for encoder_outputs: (B, 1, LangModelDim)
            encoder_hidden_states = conditioned_embedding.unsqueeze(1)
            # Pass to language model
            # Provide labels for loss calculation during training/validation
            outputs = self.language_model(
                input_ids=input_ids,           # Often used as decoder_input_ids by HF
                attention_mask=attention_mask, # For decoder attention
                labels=labels,                 # Target IDs for loss
                encoder_outputs=(encoder_hidden_states,), # Visual context passed as encoder output
                return_dict=True
            )
        else: # CausalLM
            # Conditioning CausalLMs typically requires modifying input embeddings.
            # This is a placeholder showing how inputs might be passed; a proper
            # implementation would construct multimodal `inputs_embeds`.
            warnings.warn("Forward pass for CausalLM needs modification to incorporate visual embeddings correctly (e.g., via inputs_embeds). Current pass might only use text.", RuntimeWarning)

            # Ensure input_ids and attention_mask are valid for CausalLM training
            if input_ids is None:
                # Causal LMs typically need a starting point even for "unconditional" generation tasks
                # when training. Often, this is the BOS token.
                bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
                if bos_token_id is None: raise ValueError("Tokenizer needs bos_token_id or eos_token_id for CausalLM training.")
                input_ids = torch.full((pixel_values.shape[0], 1), bos_token_id, dtype=torch.long, device=self.device)
                attention_mask = torch.ones_like(input_ids)

            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels, # Target IDs for loss calculation
                return_dict=True
                # NOTE: Passing `encoder_hidden_states` here would likely be ignored by standard CausalLMs.
                # Correct method involves creating `inputs_embeds`.
            )

        return outputs # Contains loss, logits, etc.

    def _prepare_batch(self, batch):
        """Helper to tokenize text and prepare labels."""
        pixel_values = batch['pixel_values'].to(self.device) # Move pixels to device

        # Determine task type and prepare text
        if 'question' in batch and 'answer' in batch: # VQA Task
            # Format input prompt (model-specific, adjust as needed)
            # Example for T5-style VQA:
            input_texts = [f"question: {q} context: " for q in batch['question']]
            target_texts = batch['answer']
        elif 'report' in batch: # Report Generation Task
            # Format input prompt (can be empty or a prefix)
            # Example for T5: prefix indicating task
            input_texts = ["generate report: "] * len(batch['report']) # Simple prefix
            # For CausalLM, might leave input_texts empty/None and handle in forward
            # if not self.is_encoder_decoder_model:
            #     input_texts = None
            target_texts = batch['report']
        else:
            # Handle cases where keys might be missing or inconsistent
            warnings.warn(f"Batch keys ({list(batch.keys())}) do not match expected VQA ('question', 'answer') or Report ('report') format. Attempting fallback or skipping.")
            # Fallback or return None to skip batch
            return None, None, None, None # Indicate failure to prepare

        # --- Tokenize inputs ---
        input_ids, attention_mask = None, None
        if input_texts: # Only tokenize if input_texts are defined
            input_encoding = self.tokenizer(
                input_texts,
                return_tensors='pt',
                padding='longest', # Pad to longest sequence in batch
                truncation=True,
                max_length=512 # Max length for input prompts
            )
            input_ids = input_encoding.input_ids.to(self.device)
            attention_mask = input_encoding.attention_mask.to(self.device)

        # --- Tokenize targets (labels) ---
        if not target_texts:
             warnings.warn("Target texts are missing in the batch. Cannot compute loss.")
             labels = None
        else:
             target_encoding = self.tokenizer(
                 target_texts,
                 return_tensors='pt',
                 padding='longest',
                 truncation=True,
                 max_length=self.max_label_length # Use configured max length
             )
             labels = target_encoding.input_ids.to(self.device)

             # Replace padding token id in labels with -100 for CrossEntropyLoss
             # This is standard practice for Hugging Face model loss calculation
             labels[labels == self.tokenizer.pad_token_id] = -100

        return pixel_values, input_ids, attention_mask, labels


    def training_step(self, batch, batch_idx):
        pixel_values, input_ids, attention_mask, labels = self._prepare_batch(batch)

        if pixel_values is None: # Skip if batch preparation failed
            warnings.warn(f"Skipping training step for batch {batch_idx} due to preparation error.")
            return None

        if labels is None:
             warnings.warn(f"Skipping training step for batch {batch_idx} due to missing labels.")
             return None

        outputs = self(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss # The model calculates the loss if labels are provided

        if loss is None:
             warnings.warn(f"Loss is None for batch {batch_idx}. Check model output and label preparation.")
             return None

        # Log training loss
        # sync_dist=True ensures correct logging in distributed training (DDP)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, input_ids, attention_mask, labels = self._prepare_batch(batch)

        if pixel_values is None: # Skip if batch preparation failed
            warnings.warn(f"Skipping validation step for batch {batch_idx} due to preparation error.")
            return None
        if labels is None:
             warnings.warn(f"Skipping validation step for batch {batch_idx} due to missing labels.")
             return None

        outputs = self(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        if loss is None:
             warnings.warn(f"Validation loss is None for batch {batch_idx}. Check model output.")
             return None # Or return a tensor(0.0) if necessary for logging aggregation

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Optionally generate text and calculate metrics here (can be slow)
        # Example: Calculate BLEU, ROUGE, etc. on generated text vs ground truth
        # Or calculate metrics in validation_epoch_end / using a callback
        return loss

    def configure_optimizers(self):
        # If using QLoRA/PEFT, only adapter parameters and the bridge should be optimized.
        # If fine-tuning the whole model, optimize all parameters.
        # AdamW is a common choice.
        # The self.parameters() method of LightningModule correctly returns only parameters with requires_grad=True
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        print(f"Configured AdamW optimizer with LR: {self.learning_rate}")

        # Optional: Add learning rate scheduler
        # Example: ReduceLROnPlateau
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

        return optimizer

# --- Example Training Setup (PyTorch Lightning) ---
print("--- Stage 6 Example ---")

# Assume train_loader and val_loader from Stage 1 are available
# Create dummy loaders for structure demonstration if needed
train_loader, val_loader = None, None
try:
    if 'train_loader' not in locals() or 'val_loader' not in locals():
         warnings.warn("train_loader/val_loader not found. Creating dummy DataLoaders for Stage 6 structure example.")
         # Need dummy dataset first
         dummy_files = [f'dummy_path_{i}' for i in range(10)] # More samples for dummy loader
         dummy_labs = [{'question': f'Q{i}?', 'answer': f'A{i}.'} for i in range(10)] # VQA dummy data
         # dummy_labs = [f'Report {i}' for i in range(10)] # Report dummy data
         dummy_dataset = MRIDataset(dummy_files, dummy_labs) # Minimal dataset
         train_loader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE)
         val_loader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE)
         print(f"Created dummy DataLoaders with {len(dummy_dataset)} samples.")
    else:
         # If loaders exist from previous execution (e.g., in a notebook)
         print("Using existing train_loader and val_loader.")
         train_loader = train_loader # Already defined
         val_loader = val_loader   # Already defined

except NameError:
     warnings.warn("Could not find or create dummy DataLoaders. Training cannot proceed.")
     exit()
except Exception as e_load:
     warnings.warn(f"Error setting up DataLoaders: {e_load}. Training cannot proceed.")
     exit()


# Instantiate the main model
if train_loader and val_loader:
    try:
        neuro_report_model = NeuroReportModel(
            vision_encoder_instance,
            slice_aggregator_instance,
            bridge_instance,
            language_model, # Loaded/defined earlier
            tokenizer,      # Loaded/defined earlier
            is_encoder_decoder_model=is_encoder_decoder, # Flag from stage 5 or dummy
            learning_rate=LEARNING_RATE,
            max_label_length=MAX_LABEL_LENGTH
        )

        # Configure Trainer
        # Add callbacks like ModelCheckpoint, EarlyStopping as needed
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True) # Ensure checkpoint directory exists
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=MODEL_SAVE_PATH,
            filename='neuroreport-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,       # Save only the best model based on monitor
            monitor='val_loss', # Metric to monitor
            mode='min',         # Mode for the monitored metric ('min' for loss)
            save_last=True      # Optionally save the last checkpoint as well
        )

        # Optional: Early stopping
        # early_stopping_callback = pl.callbacks.EarlyStopping(
        #     monitor='val_loss',
        #     patience=3, # Stop if val_loss doesn't improve for 3 epochs
        #     mode='min'
        # )

        # Determine precision based on availability and QLoRA status
        precision_setting = '32-true' # Default to float32
        if torch.cuda.is_available():
             if USE_QLORA:
                  # QLoRA uses 4-bit weights internally, computations often in bf16/fp16.
                  # Trainer precision should generally be 32 for stability with BitsAndBytes.
                  precision_setting = '32-true'
                  warnings.warn("Using QLoRA (4-bit). Trainer precision set to 32-true for stability.")
             elif torch.cuda.is_bf16_supported():
                  precision_setting = 'bf16-mixed'
                  print("Using bfloat16 mixed precision.")
             else:
                  precision_setting = '16-mixed'
                  print("Using float16 mixed precision.")


        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto", # Use all available devices ("auto") or specify count e.g., [0, 1]
            # strategy="ddp" if torch.cuda.device_count() > 1 else "auto", # Enable DDP if multiple GPUs
            precision=precision_setting, # Set precision based on QLoRA and hardware
            callbacks=[checkpoint_callback], # Add callbacks: checkpoint_callback, early_stopping_callback
            # logger=... # Optional: Add TensorBoard or WandB logger
            log_every_n_steps=10, # How often to log metrics
            # gradient_clip_val=1.0 # Optional: Gradient clipping for stability
            # limit_train_batches=0.1, # Optional: Use fraction of training data for debugging
            # limit_val_batches=0.1,   # Optional: Use fraction of validation data for debugging
        )

        print(f"\nPyTorch Lightning Trainer configured:")
        print(f"  - Max Epochs: {NUM_EPOCHS}")
        print(f"  - Accelerator: {trainer.accelerator.__class__.__name__}")
        print(f"  - Devices: {trainer.num_devices}")
        # print(f"  - Strategy: {trainer.strategy.__class__.__name__}")
        print(f"  - Precision: {trainer.precision}")
        print(f"  - Checkpoint Path: {MODEL_SAVE_PATH}")

        print("\nStarting training (call to trainer.fit)...")
        # --- Start Training ---
        trainer.fit(neuro_report_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # print("Trainer.fit(...) call is commented out for this example.")
        # print("To run training, ensure valid DataLoaders and uncomment the trainer.fit line.")

    except Exception as e:
        import traceback
        print(f"\nError setting up or running training: {e}")
        traceback.print_exc()
else:
    print("Skipping Trainer setup because DataLoaders are not available.")

print("\nStage 6: Training setup complete.\n")
