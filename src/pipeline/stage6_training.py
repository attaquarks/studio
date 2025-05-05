# ========== Stage 6: Training and Fine-Tuning ==========
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader # For dummy loader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler # For dummy/scheduler
import warnings
import os
from typing import Optional

# --- Import Components from Previous Stages ---
pipeline_dir = os.path.dirname(__file__)

# Use try-except blocks for robustness if run standalone
try:
    from .stage1_data_acquisition import NeuroReportDataModule, MRIDataset, BATCH_SIZE, TARGET_SIZE, NUM_SLICES_PER_SCAN # Use DataModule
except ImportError:
    warnings.warn("Could not import from stage1. Defining dummy components for Stage 6 structure.")
    BATCH_SIZE = 2
    NUM_SLICES_PER_SCAN = 8
    TARGET_SIZE=(32, 32) # Small dummy size
    # Define minimal dummy Dataset and DataModule classes
    class MRIDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs): self.len = 10
        def __len__(self): return self.len
        def __getitem__(self, idx):
            return {'pixel_values': torch.randn(NUM_SLICES_PER_SCAN, 3, TARGET_SIZE[0], TARGET_SIZE[1]),
                    'question': f'Q{idx}', 'answer': f'A{idx}', 'report': f'R{idx}'} # Include all possible keys
    class NeuroReportDataModule(pl.LightningDataModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.batch_size = BATCH_SIZE
        def setup(self, stage=None):
            self.train_dataset = MRIDataset()
            self.val_dataset = MRIDataset()
        def train_dataloader(self): return DataLoader(self.train_dataset, batch_size=self.batch_size)
        def val_dataloader(self): return DataLoader(self.val_dataset, batch_size=self.batch_size)


try:
    from .stage2_vision_encoder import VisionEncoder, VISION_FEATURE_DIM as STAGE2_VISION_FEATURE_DIM
except ImportError:
    warnings.warn("Could not import VisionEncoder from stage2. Defining dummy.")
    STAGE2_VISION_FEATURE_DIM = 768
    class VisionEncoder(nn.Module):
        def __init__(self, **kwargs): super().__init__(); self.feature_dim = STAGE2_VISION_FEATURE_DIM; self.dummy = nn.Linear(10, self.feature_dim)
        def forward(self, x): b, s, c, h, w = x.shape; return torch.randn(b, s, self.feature_dim, device=x.device)

try:
    from .stage3_slice_aggregation import SliceAggregator, AGGREGATOR_OUTPUT_DIM as STAGE3_AGGREGATOR_OUTPUT_DIM
except ImportError:
    warnings.warn("Could not import SliceAggregator from stage3. Defining dummy.")
    STAGE3_AGGREGATOR_OUTPUT_DIM = STAGE2_VISION_FEATURE_DIM # Assume mean pooling
    class SliceAggregator(nn.Module):
        def __init__(self, **kwargs): super().__init__(); self.output_dim = STAGE3_AGGREGATOR_OUTPUT_DIM
        def forward(self, x, **kwargs): return x.mean(dim=1)

try:
    from .stage4_vision_language_bridge import CrossAttentionBridge, TARGET_LANGUAGE_MODEL_DIM as STAGE4_TARGET_DIM
except ImportError:
    warnings.warn("Could not import from stage4. Defining dummy CrossAttentionBridge.")
    STAGE4_TARGET_DIM = 768 # Must match dummy LM dim below
    class CrossAttentionBridge(nn.Module):
        def __init__(self, language_dim=STAGE4_TARGET_DIM, **kwargs): super().__init__(); self.dummy = nn.Linear(10, language_dim)
        def forward(self, vis, lang, **kwargs): return lang # Pass through dummy

try:
    from .stage5_language_decoder import LanguageDecoder, LANGUAGE_MODEL_NAME, model_type as STAGE5_LM_TYPE, USE_LORA as STAGE5_USE_LORA, LANGUAGE_MODEL_DIM as STAGE5_LM_DIM
    # Verify LM dim consistency
    if STAGE4_TARGET_DIM != STAGE5_LM_DIM:
         warnings.warn(f"Mismatch! Stage 4 target dim ({STAGE4_TARGET_DIM}) != Stage 5 LM dim ({STAGE5_LM_DIM}). Check configurations.")
         # Use Stage 5 dim as the definitive one if imported
         STAGE4_TARGET_DIM = STAGE5_LM_DIM

except ImportError as e:
    warnings.warn(f"Could not import components from stage5 ({e}). Defining dummy components for Stage 6 structure.")
    LANGUAGE_MODEL_NAME = 'google/flan-t5-base' # Use a consistent model name
    STAGE5_LM_TYPE = 'seq2seq'
    STAGE5_USE_LORA = False
    STAGE4_TARGET_DIM = 768 # Reset based on dummy T5
    STAGE5_LM_DIM = STAGE4_TARGET_DIM
    # Define dummy LanguageDecoder class (cannot easily load model here without full Stage 5 logic)
    class LanguageDecoder(nn.Module):
         def __init__(self, *args, **kwargs):
             super().__init__()
             self.model = None # Cannot instantiate easily here
             self.tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
             if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
             self.model_dim = STAGE5_LM_DIM
         def get_model_dim(self): return self.model_dim
         def prepare_inputs(self, *args, **kwargs): return {'input_ids': torch.randint(0, 100, (2, 10)), 'attention_mask': torch.ones(2, 10)}
         def forward(self, *args, **kwargs): return {'loss': torch.tensor(0.0)} # Dummy output with loss
         def generate(self, *args, **kwargs): return torch.randint(0, 100, (2, 5)) # Dummy generated IDs
         def decode(self, *args, **kwargs): return ["dummy output"] * 2


# --- Configuration ---
LEARNING_RATE = 2e-5 # Often lower for fine-tuning large models, esp. with LoRA
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
MODEL_SAVE_PATH = "./neuroreport_model_checkpoint" # Directory to save checkpoints
MAX_LABEL_LENGTH = 256 # Max length for target labels (reports/answers)
WARMUP_STEPS = 100 # Steps for learning rate warmup
GRADIENT_CLIP_VAL = 1.0 # Optional gradient clipping

# --- Instantiate components (assuming they are loaded/imported correctly) ---
# These should be the actual instances passed or created based on full config
try:
    # Assuming instances are created based on imported classes/configs
    vision_encoder_instance = VisionEncoder() # Uses defaults from stage2 or dummy
    slice_aggregator_instance = SliceAggregator() # Uses defaults from stage3 or dummy
    language_decoder_instance = LanguageDecoder() # Uses defaults from stage5 or dummy
    cross_attention_bridge_instance = CrossAttentionBridge(
        vision_dim=slice_aggregator_instance.output_dim, # Use actual output dim
        language_dim=language_decoder_instance.get_model_dim() # Use actual LM dim
    )
    tokenizer_instance = language_decoder_instance.tokenizer # Get tokenizer from the decoder instance

except Exception as e_inst:
    print(f"Error instantiating pipeline components for training: {e_inst}")
    print("Check component definitions and configurations in previous stages.")
    exit()


# --- Combined Model (PyTorch Lightning Module) ---
class NeuroReportModel(pl.LightningModule):
    """
    Combines all stages into a single model for end-to-end training using PyTorch Lightning.
    Uses Cross-Attention Bridge.
    """
    def __init__(self,
                 vision_encoder: VisionEncoder,
                 slice_aggregator: SliceAggregator,
                 cross_attention_bridge: CrossAttentionBridge,
                 language_decoder: LanguageDecoder,
                 mode: str = "vqa", # 'vqa' or 'report'
                 learning_rate: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 warmup_steps: int = WARMUP_STEPS,
                 max_label_length: int = MAX_LABEL_LENGTH):
        super().__init__()
        # Save hyperparameters - important for loading checkpoints
        # Ignore large components to avoid saving them directly in hparams.yaml
        self.save_hyperparameters(ignore=['vision_encoder', 'slice_aggregator', 'cross_attention_bridge', 'language_decoder'])

        self.vision_encoder = vision_encoder
        self.slice_aggregator = slice_aggregator
        self.cross_attention_bridge = cross_attention_bridge
        self.language_decoder = language_decoder # Contains tokenizer and model
        self.mode = mode

        # Store training config params needed for optimizer/scheduler setup
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_label_length = max_label_length

        # --- Parameter Freezing Logic (Example) ---
        # Freeze vision encoder backbone if needed (fine-tune only top layers or bridge/LM)
        # if self.hparams.freeze_vision_backbone: # Assumes freeze_vision_backbone is saved in hparams
        #     for param in self.vision_encoder.backbone.parameters():
        #         param.requires_grad = False

        # If using PEFT/LoRA, the LanguageDecoder's `get_peft_model` call
        # should have already frozen the base LM and made only adapters trainable.
        # We might only want to train adapters + bridge + aggregator (optionally).
        # Example: Freeze everything *except* LoRA adapters and the bridge
        # if self.language_decoder.use_lora:
        #     print("Freezing non-LoRA and non-bridge parameters...")
        #     for name, param in self.named_parameters():
        #         is_trainable = False
        #         if 'lora_' in name: is_trainable = True # Train LoRA adapters
        #         if 'cross_attention_bridge' in name: is_trainable = True # Train bridge
        #         # if 'slice_aggregator' in name: is_trainable = True # Optionally train aggregator
        #         # if 'vision_encoder.fc' in name: is_trainable = True # Optionally train final VE layer
        #         param.requires_grad = is_trainable

        print("\n--- Trainable Parameters in NeuroReportModel ---")
        self._log_trainable_parameters()


    def _log_trainable_parameters(self):
        total_trainable_params = 0
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                print(f"  - Trainable: {name} ({param.numel():,})")
                total_trainable_params += param.numel()
            # else:
            #     print(f"  - Frozen: {name} ({param.numel():,})") # Optional: log frozen params too
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {total_trainable_params:,}")
        if total_params > 0:
             print(f"Trainable Ratio: {total_trainable_params / total_params * 100:.4f}%")
        print("-" * 40 + "\n")


    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for training/evaluation (calculates loss).

        Args:
            pixel_values (torch.Tensor): Batch of image slices [B, S, C, H, W]
            input_ids (torch.Tensor): Tokenized input text (question/prefix) [B, L_in]
            attention_mask (torch.Tensor): Attention mask for input_ids [B, L_in]
            labels (torch.Tensor, optional): Tokenized target text (answer/report) [B, L_out]. Required for loss.

        Returns:
            Model output containing loss (e.g., Seq2SeqLMOutput or CausalLMOutputWithPast).
        """
        # 1. Extract visual features
        # Input shape [B, S, C, H, W] -> Output shape [B, S, D_v]
        visual_features_slices = self.vision_encoder(pixel_values)

        # 2. Aggregate slice features
        # Input shape [B, S, D_v] -> Output shape [B, D_agg] (D_agg = Stage 3 output dim)
        aggregated_visual_features = self.slice_aggregator(visual_features_slices)

        # 3. Get language embeddings/hidden states from input text
        # Input shapes [B, L_in] -> Output shape [B, L_in, D_l]
        # Use the underlying LM's embedding layer or full forward pass if needed
        # For Seq2Seq, encoder handles input_ids. For CausalLM, need embeddings.
        if self.language_decoder.model_type == 'seq2seq':
            # Pass visual features later via encoder_outputs in the decoder call
            # Process text inputs normally for the decoder start
            decoder_input_ids = input_ids
            decoder_attention_mask = attention_mask
            # We don't strictly need language_features here as encoder_outputs handles context
            # However, the Bridge requires language features. Let's get encoder's last hidden state?
            # This deviates slightly from pure cross-attention conditioning via encoder_outputs.
            # Alternative: Apply bridge *before* the decoder forward pass.
            # Let's try passing the *aggregated* visual features directly as `encoder_hidden_states`.
            # This requires the Bridge to output [B, 1, D_l] if projecting only visual.
            # Let's revert to using the Bridge to modify decoder inputs based on visual context.

            # Get decoder input embeddings
            if hasattr(self.language_decoder.model, 'get_input_embeddings'):
                language_embeds = self.language_decoder.model.get_input_embeddings()(decoder_input_ids)
            else: # Fallback (e.g., T5 might need shared embeddings)
                 language_embeds = self.language_decoder.model.shared(decoder_input_ids)
                 warnings.warn("Using model.shared for embeddings in Seq2Seq.")

            # Bridge visual and language features (using cross-attention inside bridge)
            # The bridge expects [B, D_agg] and [B, L_in, D_l] -> outputs [B, L_in, D_l]
            enhanced_language_features = self.cross_attention_bridge(
                 aggregated_visual_features, # [B, D_agg]
                 language_embeds,            # [B, L_in, D_l]
                 language_attention_mask=decoder_attention_mask.bool() if decoder_attention_mask is not None else None
            )
            # Use enhanced features as input embeddings for the decoder
            model_inputs = {
                "inputs_embeds": enhanced_language_features,
                "attention_mask": decoder_attention_mask,
                "labels": labels,
                # We *could* still pass original aggregated features as encoder_outputs
                # if the decoder is designed for it, but using inputs_embeds is common.
                # "encoder_outputs": (aggregated_visual_features.unsqueeze(1),), # Shape [B, 1, D_agg] - requires bridge inside decoder
                "return_dict": True
            }
            outputs = self.language_decoder(**model_inputs)


        elif self.language_decoder.model_type == 'causal_lm':
            # For Causal LMs, we typically modify the `inputs_embeds`.
            if input_ids is None: # Should have input_ids (at least BOS) for CausalLM
                raise ValueError("input_ids are required for CausalLM forward pass during training.")

            # Get text embeddings
            language_embeds = self.language_decoder.model.get_input_embeddings()(input_ids) # [B, L_in, D_l]

            # Bridge visual and language features
            enhanced_language_features = self.cross_attention_bridge(
                aggregated_visual_features, # [B, D_agg]
                language_embeds,            # [B, L_in, D_l]
                language_attention_mask=attention_mask.bool() if attention_mask is not None else None
            ) # Output shape [B, L_in, D_l]

            # Pass enhanced embeddings to the Causal LM
            # Note: `labels` need to align with the `inputs_embeds` sequence length.
            # Ensure label preparation shifts labels correctly if needed.
            outputs = self.language_decoder(
                inputs_embeds=enhanced_language_features,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        else:
            raise ValueError(f"Unsupported language model type: {self.language_decoder.model_type}")

        return outputs


    def _prepare_batch(self, batch):
        """Helper to tokenize text and prepare labels for loss calculation."""
        # Ensure pixel_values are on the correct device
        pixel_values = batch['pixel_values'].to(self.device)

        input_texts = None
        target_texts = None

        if self.mode == "vqa":
            if 'question' not in batch or 'answer' not in batch:
                 warnings.warn("VQA mode requires 'question' and 'answer' keys in batch.")
                 return None # Cannot proceed
            # Format for VQA (adjust based on LM expectations if needed)
            input_texts = [f"question: {q} context: " for q in batch['question']]
            target_texts = batch['answer']
        elif self.mode == "report":
            if 'report' not in batch:
                 warnings.warn("Report mode requires 'report' key in batch.")
                 return None # Cannot proceed
            # Format for Report Generation (prefix can guide the model)
            # For CausalLM, input might just be BOS, target is the full report.
            # For Seq2Seq, input is prefix, target is the report.
            if self.language_decoder.model_type == 'seq2seq':
                 input_texts = ["generate report: "] * len(batch['report'])
            else: # CausalLM - start with BOS token
                 # Input text is effectively handled by starting with BOS embedding
                 input_texts = [self.language_decoder.tokenizer.bos_token] * len(batch['report']) if self.language_decoder.tokenizer.bos_token else [""] * len(batch['report'])

            target_texts = batch['report']
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose 'vqa' or 'report'.")

        # --- Tokenize inputs ---
        input_ids, attention_mask = None, None
        try:
            # Use the tokenizer from the language_decoder instance
            tokenizer = self.language_decoder.tokenizer
            input_encoding = tokenizer(
                input_texts,
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=512 # Max length for input part
            )
            input_ids = input_encoding.input_ids.to(self.device)
            attention_mask = input_encoding.attention_mask.to(self.device)
        except Exception as e_tok_in:
            warnings.warn(f"Error tokenizing input texts: {e_tok_in}")
            return None # Cannot proceed without valid inputs

        # --- Tokenize targets (labels) ---
        labels = None
        try:
            tokenizer = self.language_decoder.tokenizer
            target_encoding = tokenizer(
                target_texts,
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=self.max_label_length # Use configured max length for labels
            )
            labels = target_encoding.input_ids.to(self.device)

            # Replace padding token id in labels with -100 for CrossEntropyLoss
            labels[labels == tokenizer.pad_token_id] = -100
        except Exception as e_tok_tgt:
             warnings.warn(f"Error tokenizing target texts: {e_tok_tgt}")
             # Allow proceeding without labels for eval steps, but training needs them
             labels = None


        # For CausalLM, labels often need to be the same as input_ids, shifted.
        # However, HF models typically handle this internally when `labels` are passed alongside `input_ids` or `inputs_embeds`.
        # Ensure the `labels` tensor corresponds correctly to the sequence passed (either via input_ids or inputs_embeds).
        # If we prepend visual features to embeds, labels might need masking for the visual part.
        # For simplicity with the current bridge modifying language embeds, we assume HF handles label alignment.


        return pixel_values, input_ids, attention_mask, labels

    def training_step(self, batch, batch_idx):
        prep_result = self._prepare_batch(batch)
        if prep_result is None:
            warnings.warn(f"Skipping training step {batch_idx}: Batch preparation failed.")
            return None # Skip step if batch prep failed
        pixel_values, input_ids, attention_mask, labels = prep_result

        if labels is None:
             warnings.warn(f"Skipping training step {batch_idx}: Labels are missing.")
             return None # Skip step if labels missing


        # Perform forward pass to get loss
        try:
            outputs = self(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        except Exception as e_fwd:
             warnings.warn(f"Error during training forward pass {batch_idx}: {e_fwd}")
             # Maybe return None or a zero tensor to avoid crashing trainer
             return torch.tensor(0.0, device=self.device, requires_grad=True) # Dummy loss


        if loss is None:
             warnings.warn(f"Loss is None for training batch {batch_idx}. Check model output and label preparation.")
             return None # Skip if loss calculation failed

        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        prep_result = self._prepare_batch(batch)
        if prep_result is None:
            warnings.warn(f"Skipping validation step {batch_idx}: Batch preparation failed.")
            return None
        pixel_values, input_ids, attention_mask, labels = prep_result

        if labels is None:
             warnings.warn(f"Skipping validation step {batch_idx}: Labels are missing (cannot calculate val_loss).")
             # If we still want to log generation metrics, we can proceed without loss
             # For now, return None if loss cannot be calculated.
             return None


        try:
            outputs = self(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        except Exception as e_fwd:
             warnings.warn(f"Error during validation forward pass {batch_idx}: {e_fwd}")
             loss = None # Ensure loss is None


        if loss is not None:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            warnings.warn(f"Validation loss is None for batch {batch_idx}. Check model output.")

        # Optionally generate text and calculate metrics here (can be slow)
        # Example: Log generated text samples periodically
        # if batch_idx % 10 == 0: # Log every 10 batches
        #     self.log_validation_samples(batch, outputs) # Implement this helper method

        # Return loss (or other metrics if calculated)
        return loss


    # Optional helper for logging validation samples
    # def log_validation_samples(self, batch, outputs):
    #     # ... logic to generate text using self.language_decoder.generate ...
    #     # ... decode predictions and references ...
    #     # ... log using self.logger.experiment.add_text ...
    #     pass


    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Filter parameters that require gradients
        # This automatically handles PEFT/LoRA cases where only adapters are trainable
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
             warnings.warn("No trainable parameters found! Check model freezing logic and LoRA setup.")
             # Add dummy parameter to optimizer to avoid crashing
             optimizer = optim.AdamW([nn.Parameter(torch.zeros(1))], lr=self.learning_rate)
        else:
             print(f"Configuring optimizer for {len(trainable_params)} trainable parameter tensors.")
             optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # Learning Rate Scheduler (Example: Linear warmup then decay)
        try:
            # Calculate total steps (required for some schedulers)
            # Handle cases where trainer/estimated_stepping_batches might not be available yet
            total_steps = self.trainer.estimated_stepping_batches if hasattr(self.trainer, 'estimated_stepping_batches') else 10000 # Estimate if needed
            print(f"Total estimated training steps: {total_steps}")
            print(f"Warmup steps: {self.warmup_steps}")

            scheduler = get_scheduler(
                name="linear", # Example: linear warmup and decay
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step", # Call scheduler step-wise
                "frequency": 1,
            }
            print(f"Configured 'linear' LR scheduler with {self.warmup_steps} warmup steps.")
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        except Exception as e_sched:
             warnings.warn(f"Could not configure LR scheduler: {e_sched}. Using optimizer only.")
             return optimizer


# --- Example Training Setup (PyTorch Lightning) ---
if __name__ == "__main__":
    print("--- Stage 6 Example ---")

    # --- Dummy DataModule Setup ---
    # Assume data_module is instantiated correctly (using dummy or real data)
    # This needs access to annotations file path and data directory path
    DUMMY_DATA_DIR_S6 = "./dummy_mri_data_s6" # Use different dir for clarity
    DUMMY_ANNOTATIONS_FILE_S6 = "./dummy_annotations_s6.csv"
    # Recreate dummy data if needed
    if not os.path.exists(DUMMY_ANNOTATIONS_FILE_S6):
         os.makedirs(DUMMY_DATA_DIR_S6, exist_ok=True)
         dummy_data_s6 = []
         for i in range(10): # Small dummy dataset for example
            scan_name = f"scan_{i+1}.nii.gz"
            file_path = os.path.join(DUMMY_DATA_DIR_S6, scan_name)
            dummy_volume = np.random.rand(32, 32, 10).astype(np.float32) # Very small volume
            nib.save(nib.Nifti1Image(dummy_volume, np.eye(4)), file_path)
            dummy_data_s6.append({'file_name': scan_name, 'question': f'Q{i}', 'answer': f'A{i}', 'report': f'R{i}'})
         pd.DataFrame(dummy_data_s6).to_csv(DUMMY_ANNOTATIONS_FILE_S6, index=False)
         print(f"Created dummy data for Stage 6 in {DUMMY_DATA_DIR_S6}")

    data_module_instance = None
    try:
        data_module_instance = NeuroReportDataModule(
            data_dir=DUMMY_DATA_DIR_S6,
            annotations_path=DUMMY_ANNOTATIONS_FILE_S6,
            batch_size=BATCH_SIZE,
            mode="vqa", # Set mode consistent with dummy data/model task
            target_size=TARGET_SIZE, # Use consistent target size
            num_workers=0, # Use 0 for debugging
            n_slices=NUM_SLICES_PER_SCAN # Use consistent slice count
        )
        # Prepare data to ensure datasets are created before trainer needs them
        data_module_instance.prepare_data()
        data_module_instance.setup('fit')
    except Exception as e_dm:
         print(f"Error creating dummy DataModule: {e_dm}")
         data_module_instance = None # Ensure it's None if setup fails

    # --- Instantiate the main model ---
    if data_module_instance:
        try:
            # Use the component instances created earlier
            neuro_report_model_instance = NeuroReportModel(
                vision_encoder=vision_encoder_instance,
                slice_aggregator=slice_aggregator_instance,
                cross_attention_bridge=cross_attention_bridge_instance,
                language_decoder=language_decoder_instance, # Contains tokenizer and model
                mode="vqa", # Should match DataModule mode
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                warmup_steps=WARMUP_STEPS,
                max_label_length=MAX_LABEL_LENGTH
            )

            # --- Configure Trainer ---
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=MODEL_SAVE_PATH,
                filename='neuroreport-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,       # Save only the best model based on monitor
                monitor='val_loss', # Metric to monitor
                mode='min',         # Mode for the monitored metric ('min' for loss)
                save_last=True      # Optionally save the last checkpoint
            )
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
            # early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

            # Determine precision based on QLoRA status
            precision_setting = '32-true' # Default
            if torch.cuda.is_available():
                 if hasattr(language_decoder_instance, 'use_qlora') and language_decoder_instance.use_qlora:
                      precision_setting = '32-true' # Recommended for stability with bitsandbytes
                      print("Using QLoRA (4-bit): Trainer precision set to 32-true.")
                 elif torch.cuda.is_bf16_supported():
                      precision_setting = 'bf16-mixed'
                      print("Using bfloat16 mixed precision.")
                 else:
                      precision_setting = '16-mixed'
                      print("Using float16 mixed precision.")


            trainer = pl.Trainer(
                max_epochs=NUM_EPOCHS,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices="auto",
                precision=precision_setting,
                callbacks=[checkpoint_callback, lr_monitor], # Add callbacks
                log_every_n_steps=10, # Log less frequently for faster example run
                gradient_clip_val=GRADIENT_CLIP_VAL,
                # limit_train_batches=5, # DEBUG: Use fraction of training data
                # limit_val_batches=2,   # DEBUG: Use fraction of validation data
            )

            print(f"\nPyTorch Lightning Trainer configured:")
            print(f"  - Mode: {neuro_report_model_instance.mode}")
            print(f"  - Max Epochs: {NUM_EPOCHS}")
            print(f"  - Precision: {trainer.precision}")
            print(f"  - Checkpoint Path: {MODEL_SAVE_PATH}")
            print("\nStarting training (fit call commented out for example)...")

            # --- Start Training ---
            # trainer.fit(neuro_report_model_instance, datamodule=data_module_instance)
            print("\nTrainer.fit(...) call is commented out.")
            print("To run training, ensure valid DataLoaders, dependencies, and uncomment the trainer.fit line.")

        except Exception as e:
            import traceback
            print(f"\nError setting up or running training: {e}")
            traceback.print_exc()
    else:
        print("Skipping Trainer setup because DataModule could not be initialized.")

    # Clean up dummy files (optional)
    # import shutil
    # try:
    #     if os.path.exists(DUMMY_DATA_DIR_S6): shutil.rmtree(DUMMY_DATA_DIR_S6)
    #     if os.path.exists(DUMMY_ANNOTATIONS_FILE_S6): os.remove(DUMMY_ANNOTATIONS_FILE_S6)
    #     print("Cleaned up dummy files for Stage 6.")
    # except Exception as e_clean: print(f"Error cleaning up Stage 6 dummy files: {e_clean}")

    print("\nStage 6: Training setup complete.\n")

# Expose key training config for potential use in evaluation/inference stages
# MODEL_SAVE_PATH, MAX_LABEL_LENGTH
